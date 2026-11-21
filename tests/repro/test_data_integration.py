"""End-to-end over the committed dataset: generation, env loading, validity, scoring, repro."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType

import pytest

from fintrace.constants import MAX_FINDINGS
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.matcher import match
from fintrace.evaluator.metrics import compute_metrics
from fintrace.generator.injectors.base import inconsistent_ids
from fintrace.generator.validate import validate_clean_history
from fintrace.models import Finding, PrivateEpisode
from fintrace.reward.official import official_reward

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"
GENERATION_BUDGET_S = 60.0


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "generate_data.py"
    spec = importlib.util.spec_from_file_location("generate_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_script()


@pytest.fixture(autouse=True)
def _cached_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build each (seed, config) once per module: the script is cheap, the generator is not."""
    real = GEN.build_episode
    store: dict[tuple[int, str], object] = {}

    def cached(seed: int, config: object = None) -> object:
        key = (seed, "" if config is None else config.model_dump_json())  # type: ignore[attr-defined]
        if key not in store:
            store[key] = real(seed, config)
        return store[key]

    monkeypatch.setattr(GEN, "build_episode", cached)


def _data_manifest() -> dict[str, object]:
    return json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


def _episode_dirs() -> list[Path]:
    dirs = [p for split in ("train", "dev") for p in sorted((DATA_DIR / split).glob("ep_*"))]
    assert len(dirs) == 2 * GEN.EPISODES_PER_SPLIT, "run `make data` first"
    return dirs


def _tree(root: Path) -> dict[str, bytes]:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return {str(p.relative_to(root)): p.read_bytes() for p in files}


def _canonical(episode: PrivateEpisode) -> str:
    return GEN._canonical_json(episode.model_dump(mode="json"))


def test_generate_data_actual_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "data"
    assert GEN.main(["--out-dir", str(out_dir)]) == 0
    report = json.loads(capsys.readouterr().out)
    written = (out_dir / "manifest.json").read_text(encoding="utf-8")
    assert written == GEN._canonical_json(report) + "\n"
    for split in ("train", "dev"):
        dirs = sorted((out_dir / split).glob("ep_*"))
        assert len(dirs) == 10
        for ep_dir in dirs:
            assert {p.name for p in ep_dir.iterdir()} == set(GEN.EPISODE_FILES)
            assert len(GEN.EPISODE_FILES) == 10  # 6 public + 2 private + seed + checksums
    assert GEN.main(["--verify-only", "--out-dir", str(out_dir)]) == 0
    assert capsys.readouterr().out.startswith("All checksums match.")
    # a stale episode directory from an earlier run is removed by the regeneration
    stale = out_dir / "train" / "ep_000000000000"
    stale.mkdir()
    (stale / "manifest.json").write_text("{}")
    assert GEN.main(["--split", "train", "--out-dir", str(out_dir)]) == 0
    assert not stale.exists() and len(list((out_dir / "train").glob("ep_*"))) == 10


def test_env_can_load_generated_episode_via_seed() -> None:
    """The committed episode and env.reset(seed) are the same object: same id, same truth."""
    manifest = _data_manifest()
    splits = manifest["splits"]
    assert isinstance(splits, list)
    train_report = splits[0]
    first = train_report["episodes"][0]
    config = GEN.load_config(CONFIG_DIR / "train.yaml")
    env = FintraceEnv()
    observation, _ = env.reset(seed=first["seed"], config=config)
    assert env._private is not None
    issues = env._private.truth.issues
    assert len(issues) == config.errors.total() == 8
    assert env.episode.manifest.episode_id == first["episode_id"]
    on_disk = GEN.read_episode(DATA_DIR / first["dir"])
    assert on_disk == env._private and _canonical(on_disk) == _canonical(env._private)
    assert observation.result["manifest"]["episode_id"] == first["episode_id"]


def test_all_generated_episodes_validate() -> None:
    manifest = _data_manifest()
    assert manifest["config_hash"] == GEN.config_hash(CONFIG_DIR)
    recorded = {
        e["episode_id"]: e["seed"] for split in manifest["splits"] for e in split["episodes"]
    }
    for ep_dir in _episode_dirs():
        episode = GEN.read_episode(ep_dir)
        assert isinstance(episode, PrivateEpisode)
        assert ep_dir.name == episode.public.manifest.episode_id == episode.truth.episode_id
        assert GEN.read_seed(ep_dir) == recorded[ep_dir.name]
        assert GEN.verify_episode(ep_dir)
        validate_clean_history(episode, ignore_ids=inconsistent_ids(episode))
        assert [i.type for i in episode.truth.issues].count("duplicate_invoice") == 2
        assert len(episode.truth.issues) == 8
    assert len(recorded) == 20  # 20 distinct episode ids across both splits


def test_mark_everything_scores_negative() -> None:
    """Flagging the first 30 payments as misallocated loses on every committed episode."""
    for ep_dir in _episode_dirs():
        episode = GEN.read_episode(ep_dir)
        public = episode.public
        assert len(public.payments) >= MAX_FINDINGS, f"{ep_dir.name}: too few payments to flood"
        invoice_by_id = {i.invoice_id: i for i in public.invoices}
        findings = []
        for pay in public.payments[:MAX_FINDINGS]:
            invoice = invoice_by_id[pay.invoice_refs[0].invoice_id] if pay.invoice_refs else None
            findings.append(
                Finding(
                    type="payment_allocation",
                    affected_ids=[pay.payment_id, *([invoice.invoice_id] if invoice else [])],
                    claimed_amount_minor=pay.amount_minor,
                    currency=pay.currency,
                    evidence_ids=[pay.doc_id, *([invoice.doc_id] if invoice else [])],
                    explanation="looks suspicious",
                )
            )
        assert len(findings) == MAX_FINDINGS
        tp, fp, fn = match(findings, episode.truth.issues)
        _, _, f1 = compute_metrics(tp, fp, fn)
        assert tp <= 2 and fp >= MAX_FINDINGS - 2  # at most the two payment_allocation issues
        assert official_reward(f1, 0) < 0, ep_dir.name


@pytest.mark.slow
def test_generate_data_timing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Full regeneration fits the budget and is byte-identical to the committed dataset."""
    out_dir = tmp_path / "data"
    started = time.monotonic()
    assert GEN.main(["--out-dir", str(out_dir)]) == 0
    elapsed = time.monotonic() - started
    capsys.readouterr()
    assert elapsed < GENERATION_BUDGET_S, f"generation took {elapsed:.1f}s"
    regenerated = _tree(out_dir)
    committed = {k: v for k, v in _tree(DATA_DIR).items() if not k.endswith(".gitkeep")}
    assert set(regenerated) == set(committed)
    differing = sorted(k for k in committed if committed[k] != regenerated[k])
    assert differing == [], f"committed data drifted from the generator: {differing[:5]}"
