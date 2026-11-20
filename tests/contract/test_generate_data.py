"""Contract tests for scripts/generate_data.py: configs, seed ranges, on-disk format, checksums."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from fintrace.generator.build import build_episode
from fintrace.generator.injectors.base import InjectorError
from fintrace.models import EpisodeConfig, PrivateEpisode

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script() -> ModuleType:
    path = REPO_ROOT / "scripts" / "generate_data.py"
    spec = importlib.util.spec_from_file_location("generate_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_script()
CONFIG_DIR = REPO_ROOT / "config"


def _canonical(episode: object) -> str:
    """The one on-disk JSON form; the only fair comparison of two dumps (keys are re-sorted)."""
    return GEN._canonical_json(episode.model_dump(mode="json"))  # type: ignore[attr-defined]


def test_dry_run_generates_10_each_split(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """--dry-run builds both splits in memory, prints one report JSON and writes nothing."""
    out_dir = tmp_path / "data"
    assert GEN.main(["--dry-run", "--out-dir", str(out_dir)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"config_hash", "splits"}
    assert report["config_hash"] == GEN.config_hash(CONFIG_DIR)
    assert [s["split"] for s in report["splits"]] == ["train", "dev"]
    for split in report["splits"]:
        assert split["generated_count"] == GEN.EPISODES_PER_SPLIT == 10
        assert len(split["episodes"]) == 10
        seeds = [e["seed"] for e in split["episodes"]]
        assert seeds == sorted(seeds) and all(
            split["seed_base"] <= s < split["seed_base"] + GEN.MAX_CANDIDATE_SEEDS for s in seeds
        )
        assert not set(seeds) & {k["seed"] for k in split["skipped"]}
    assert not out_dir.exists()  # dry run never touches the disk


def test_both_configs_parse() -> None:
    train = GEN.load_config(CONFIG_DIR / "train.yaml")
    dev = GEN.load_config(CONFIG_DIR / "dev.yaml")
    assert isinstance(train, EpisodeConfig) and isinstance(dev, EpisodeConfig)
    assert train == dev  # the split is out-of-distribution by seed, not by config
    assert train.errors.total() == 8  # 4 injectors + 2 issues per scenario
    assert train.decoys.total() == 5
    assert train.n_ledger_rows_range == (500, 1500) and train.n_documents_range == (30, 150)
    assert str(train.split_payment_rate) == "0.15"  # parsed as Decimal from the quoted string


def test_train_dev_disjoint_seeds() -> None:
    ranges = {
        split: range(base, base + GEN.MAX_CANDIDATE_SEEDS)
        for split, base in GEN.SPLIT_SEED_BASES.items()
    }
    assert set(ranges) == {"train", "dev"}
    assert not set(ranges["train"]) & set(ranges["dev"])
    assert GEN.SPLIT_SEED_BASES == {"train": 1000, "dev": 10000}


def test_write_episode_roundtrip_identical(tmp_path: Path) -> None:
    """write -> read gives the identical PrivateEpisode; a second write is byte-identical."""
    train = GEN.load_config(CONFIG_DIR / "train.yaml")
    for name, episode, seed in (
        ("clean", build_episode(42), 42),
        ("full", build_episode(1001, train), 1001),
    ):
        ep_dir = tmp_path / name
        GEN.write_episode(episode, ep_dir, seed)
        assert {p.name for p in ep_dir.iterdir()} == set(GEN.EPISODE_FILES)
        loaded = GEN.read_episode(ep_dir)
        assert loaded == episode  # field-by-field; dict key order inside provenance is irrelevant
        assert _canonical(loaded) == _canonical(episode)
        assert GEN.read_seed(ep_dir) == seed
        first = {p.name: p.read_bytes() for p in ep_dir.iterdir()}
        GEN.write_episode(episode, ep_dir, seed)
        assert {p.name: p.read_bytes() for p in ep_dir.iterdir()} == first
    assert len(GEN.read_episode(tmp_path / "full").truth.issues) == 8


def test_verify_episode_detects_corruption(tmp_path: Path) -> None:
    ep_dir = tmp_path / "ep"
    GEN.write_episode(build_episode(42), ep_dir, 42)
    assert GEN.verify_episode(ep_dir)
    ledger = ep_dir / "ledger.csv"
    raw = bytearray(ledger.read_bytes())
    raw[-2] ^= 0x01  # flip one bit in the last data byte
    ledger.write_bytes(bytes(raw))
    assert not GEN.verify_episode(ep_dir)
    GEN.write_episode(build_episode(42), ep_dir, 42)  # repaired
    assert GEN.verify_episode(ep_dir)
    (ep_dir / "policy.json").unlink()  # a listed public file is missing
    assert not GEN.verify_episode(ep_dir)
    GEN.write_episode(build_episode(42), ep_dir, 42)
    (ep_dir / "truth.json").write_bytes(b"{}\n")  # private files are not covered by the checksums
    assert GEN.verify_episode(ep_dir)
    (ep_dir / GEN.CHECKSUMS_FILE).unlink()
    assert not GEN.verify_episode(ep_dir)


def _fake_builder(
    infeasible: set[int], broken: set[int]
) -> Callable[[int, EpisodeConfig | None], PrivateEpisode]:
    """Stand-in for build_episode: InjectorError on ``infeasible``, ValueError on ``broken``."""

    def build(seed: int, config: EpisodeConfig | None = None) -> PrivateEpisode:
        if seed in infeasible:
            raise InjectorError(f"seed {seed} infeasible")
        if seed in broken:
            raise ValueError(f"seed {seed} broke the generator")
        return build_episode(seed)  # a real clean episode: cheap and unique per seed

    return build


def test_build_split_skips_only_injector_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    base = GEN.SPLIT_SEED_BASES["train"]
    monkeypatch.setattr(GEN, "EPISODES_PER_SPLIT", 3)
    monkeypatch.setattr(GEN, "MAX_CANDIDATE_SEEDS", 6)
    monkeypatch.setattr(GEN, "build_episode", _fake_builder({base, base + 2}, set()))
    report, built = GEN.build_split("train", EpisodeConfig())
    assert [seed for seed, _ in built] == [base + 1, base + 3, base + 4]
    assert [k["seed"] for k in report["skipped"]] == [base, base + 2]
    assert report["skipped"][0]["reason"] == f"seed {base} infeasible"
    assert report["generated_count"] == 3 and report["candidates_tried"] == 5
    assert [e["seed"] for e in report["episodes"]] == [base + 1, base + 3, base + 4]
    # any other exception is a generator bug and must propagate untouched
    monkeypatch.setattr(GEN, "build_episode", _fake_builder({base}, {base + 1}))
    with pytest.raises(ValueError, match="broke the generator"):
        GEN.build_split("train", EpisodeConfig())
    # exhausting the pool names every skipped seed
    monkeypatch.setattr(GEN, "build_episode", _fake_builder(set(range(base, base + 6)), set()))
    with pytest.raises(RuntimeError, match=rf"only 0 feasible seeds.*{base}, {base + 1}"):
        GEN.build_split("train", EpisodeConfig())


def test_write_split_is_atomic_and_keeps_gitkeep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "data"
    train_dir = out_dir / "train"
    train_dir.mkdir(parents=True)
    (train_dir / ".gitkeep").write_bytes(b"")
    stale = train_dir / "ep_000000000000"
    stale.mkdir()
    (stale / "truth.json").write_bytes(b"{}\n")
    monkeypatch.setattr(GEN, "EPISODES_PER_SPLIT", 2)
    monkeypatch.setattr(GEN, "MAX_CANDIDATE_SEEDS", 4)
    base = GEN.SPLIT_SEED_BASES["train"]
    monkeypatch.setattr(GEN, "build_episode", _fake_builder(set(), set()))
    _, built = GEN.build_split("train", EpisodeConfig())
    GEN.write_split(out_dir, "train", built)
    names = sorted(p.name for p in train_dir.iterdir())
    assert names == sorted([".gitkeep", *(ep.public.manifest.episode_id for _, ep in built)])
    assert not stale.exists() and not list(out_dir.glob(".staging_*"))
    assert [GEN.read_seed(train_dir / ep.public.manifest.episode_id) for _, ep in built] == [
        base,
        base + 1,
    ]
    # a failing build leaves the previous dataset untouched (nothing is written before building)
    before = {p: p.read_bytes() for p in train_dir.rglob("*") if p.is_file()}
    monkeypatch.setattr(GEN, "build_episode", _fake_builder(set(), {base + 1}))
    with pytest.raises(ValueError):
        GEN.main(["--out-dir", str(out_dir)])
    assert {p: p.read_bytes() for p in train_dir.rglob("*") if p.is_file()} == before
    assert not (out_dir / "dev").exists() and not (out_dir / "manifest.json").exists()


def test_jsonl_roundtrip_survives_unicode_line_separators(tmp_path: Path) -> None:
    """ensure_ascii=False writes U+2028 / U+0085 raw; the reader must split on "\\n" only."""
    desc = f"Consulting{chr(0x2028)}retainer {chr(0x85)} Q1"  # LINE SEPARATOR and NEXT LINE
    episode = build_episode(42)
    first = episode.public.invoices[0]
    line = first.line_items[0].model_copy(update={"desc": desc})
    invoice = first.model_copy(update={"line_items": (line, *first.line_items[1:])})
    public = episode.public.model_copy(update={"invoices": (invoice, *episode.public.invoices[1:])})
    patched = episode.model_copy(update={"public": public})
    GEN.write_episode(patched, tmp_path, 42)
    loaded = GEN.read_episode(tmp_path)
    assert loaded.public.invoices[0].line_items[0].desc == desc
    assert len(loaded.public.invoices) == len(episode.public.invoices)
    assert chr(0x2028) in (tmp_path / "invoices.jsonl").read_text(encoding="utf-8")


def test_canonical_json_byte_stable() -> None:
    canon = GEN._canonical_json
    assert canon({"b": 1, "a": 2}) == canon({"a": 2, "b": 1}) == '{"a":2,"b":1}'
    assert canon({"k": [1, {"z": None, "y": "é"}]}) == '{"k":[1,{"y":"é","z":null}]}'  # no \\u
    with pytest.raises(ValueError):
        canon({"nan": float("nan")})
