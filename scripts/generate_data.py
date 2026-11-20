"""Generate the public dataset: 10 train + 10 dev episodes under ``data/`` (``make data``).

Usage::

    python scripts/generate_data.py [--split {train,dev,all}] [--dry-run] [--verify-only]
                                    [--out-dir DIR] [--config-dir DIR]

Each split reads its ``config/<split>.yaml`` into ``EpisodeConfig`` and walks a disjoint seed range
(``SPLIT_SEED_BASES``); seeds that the injectors cannot satisfy (``InjectorError``) are skipped
deterministically and listed in the report, and the first ``EPISODES_PER_SPLIT`` feasible seeds are
written as ``data/<split>/ep_<episode_id>/``:

    manifest.json ledger.csv invoices.jsonl payments.jsonl fx.csv policy.json   public, hashed
    truth.json provenance.json                                                  private (evaluator)
    checksums.sha256                                                   sha256 of the public files
    seed.txt                                                           the seed, human-readable

Every file is byte-stable: canonical JSON (sorted keys, compact separators, ``ensure_ascii=False``),
CSV via ``csv.DictWriter`` with sorted field names and LF line ends, one record per JSONL line.
``--dry-run`` builds everything in memory and only prints the report; ``--verify-only`` recomputes
every checksum (and the config hash) and exits non-zero on drift. The report printed to stdout is
the same JSON object written to ``data/manifest.json``: ``{"config_hash": ..., "splits": [...]}``.
The script never imports the evaluator or the reward: it writes truth files, it never scores them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Final, TypeAlias

import yaml

from fintrace.constants import CONFIG_DIR, DATA_DIR
from fintrace.exporters.episode_files import (
    CHECKSUMS_FILE,
    EPISODE_FILES,
    PRIVATE_FILES,
    PUBLIC_FILES,
    SEED_FILE,
    canonical_json,
    read_episode,
    read_seed,
    verify_episode,
    write_episode,
    write_text,
)
from fintrace.generator.build import build_episode
from fintrace.generator.injectors.base import InjectorError
from fintrace.models import EpisodeConfig, PrivateEpisode

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
# disjoint seed ranges: the dev split is out-of-distribution by seed, not by config
SPLIT_SEED_BASES: Final[dict[str, int]] = {"train": 1000, "dev": 10000}
EPISODES_PER_SPLIT: Final[int] = 10
MAX_CANDIDATE_SEEDS: Final[int] = 100  # ample: the full config rejected 3 of 13 seeds per split
CONFIG_FILES: Final[dict[str, str]] = {"train": "train.yaml", "dev": "dev.yaml"}
DATA_MANIFEST_FILE: Final[str] = "manifest.json"
EPISODE_DIR_PREFIX: Final[str] = "ep_"
CONFIG_HASH_HEX: Final[int] = 12
SKIP_REASON_CHARS: Final[int] = 120
STAGING_PREFIX: Final[str] = ".staging_"  # out_dir/.staging_<split>, swapped in once complete

SplitReport: TypeAlias = dict[str, Any]
SplitBuild: TypeAlias = tuple[SplitReport, list[tuple[int, PrivateEpisode]]]


_canonical_json = canonical_json  # legacy name kept for the contract tests


# ---------------------------------------------------------------- config
def load_config(path: Path) -> EpisodeConfig:
    """Parse a YAML episode config into a validated ``EpisodeConfig``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return EpisodeConfig.model_validate(raw)


def config_hash(config_dir: Path) -> str:
    """First ``CONFIG_HASH_HEX`` hex digits of sha256 over train.yaml + dev.yaml bytes."""
    digest = hashlib.sha256()
    for split in SPLIT_SEED_BASES:
        digest.update((config_dir / CONFIG_FILES[split]).read_bytes())
    return digest.hexdigest()[:CONFIG_HASH_HEX]


# ---------------------------------------------------------------- splits
def _episode_dirs(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        return []
    return sorted(
        p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith(EPISODE_DIR_PREFIX)
    )


def _clear_split_dir(split_dir: Path) -> None:
    """Remove stale ``ep_*`` directories only; ``.gitkeep`` and anything else stay."""
    for stale in _episode_dirs(split_dir):
        shutil.rmtree(stale)


def build_split(split: str, config: EpisodeConfig) -> SplitBuild:
    """Build the first ``EPISODES_PER_SPLIT`` feasible seeds of ``split`` in memory.

    Only ``InjectorError`` marks a seed infeasible (skipped, reported); any other exception is a
    bug and propagates. Exhausting the candidate pool raises ``RuntimeError`` listing the skips.
    """
    seed_base = SPLIT_SEED_BASES[split]
    built: list[tuple[int, PrivateEpisode]] = []
    skipped: list[dict[str, Any]] = []
    candidates_tried = 0
    for seed in range(seed_base, seed_base + MAX_CANDIDATE_SEEDS):
        candidates_tried += 1
        try:
            episode = build_episode(seed, config)
        except InjectorError as exc:
            reason = str(exc)[:SKIP_REASON_CHARS]
            skipped.append({"seed": seed, "reason": reason})
            print(f"[{split}] seed {seed}: skipped ({reason})", file=sys.stderr)
            continue
        built.append((seed, episode))
        if len(built) == EPISODES_PER_SPLIT:
            break
    if len(built) < EPISODES_PER_SPLIT:
        skipped_seeds = [entry["seed"] for entry in skipped]
        last_seed = seed_base + MAX_CANDIDATE_SEEDS - 1
        raise RuntimeError(
            f"{split}: only {len(built)} feasible seeds in {seed_base}..{last_seed}"
            f" (need {EPISODES_PER_SPLIT}); skipped: {skipped_seeds}"
        )
    report: SplitReport = {
        "split": split,
        "seed_base": seed_base,
        "config": CONFIG_FILES[split],
        "generated_count": len(built),
        "candidates_tried": candidates_tried,
        "episodes": [
            {
                "seed": seed,
                "episode_id": ep.public.manifest.episode_id,
                "dir": f"{split}/{ep.public.manifest.episode_id}",
            }
            for seed, ep in built
        ],
        "skipped": skipped,
    }
    return report, built


def write_split(out_dir: Path, split: str, built: list[tuple[int, PrivateEpisode]]) -> None:
    """Replace ``out_dir/<split>/ep_*`` with ``built``: stage everything first, then swap.

    All files are written into ``.staging_<split>`` before the old ``ep_*`` directories are
    removed, so a write failure leaves the previous dataset untouched; the swap itself is a set
    of same-filesystem renames.
    """
    staging = out_dir / f"{STAGING_PREFIX}{split}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        for seed, episode in built:
            write_episode(episode, staging / episode.public.manifest.episode_id, seed)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)  # never leave a half-written staging dir
        raise
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    _clear_split_dir(split_dir)
    for staged in _episode_dirs(staging):
        staged.rename(split_dir / staged.name)
    staging.rmdir()


def generate_split(
    split: str, config: EpisodeConfig, out_dir: Path, *, dry_run: bool
) -> SplitReport:
    """Build one split and, unless ``dry_run``, write it; returns the split report."""
    report, built = build_split(split, config)
    if not dry_run:
        write_split(out_dir, split, built)
    return report


def write_data_manifest(out_dir: Path, report: dict[str, Any]) -> None:
    """``data/manifest.json``: the same report object the CLI prints."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(out_dir / DATA_MANIFEST_FILE, canonical_json(report) + "\n")


def _verify_split(out_dir: Path, split: str, recorded: dict[str, int]) -> list[str]:
    """Failures of one split: directory set vs the manifest, file set, checksums, seeds."""
    failures: list[str] = []
    dirs = _episode_dirs(out_dir / split)
    if len(dirs) != EPISODES_PER_SPLIT:
        failures.append(f"{split}: expected {EPISODES_PER_SPLIT} episode dirs, got {len(dirs)}")
    on_disk = {f"{split}/{d.name}" for d in dirs}
    listed = {key for key in recorded if key.startswith(f"{split}/")}
    if on_disk != listed:
        failures.append(f"{split}: directories differ from {DATA_MANIFEST_FILE}")
    for ep_dir in dirs:
        rel = ep_dir.relative_to(out_dir).as_posix()  # manifest keys are always "<split>/<id>"
        if {p.name for p in ep_dir.iterdir()} != EPISODE_FILES:
            failures.append(f"unexpected file set: {rel}")
        elif not verify_episode(ep_dir):
            failures.append(f"checksum mismatch: {rel}")
        elif recorded.get(rel) != read_seed(ep_dir):
            failures.append(f"seed differs from {DATA_MANIFEST_FILE}: {rel}")
    return failures


def verify_all(out_dir: Path, config_dir: Path) -> int:
    """Re-check every split against ``data/manifest.json`` and the configs; 0 iff no drift."""
    failures: list[str] = []
    recorded: dict[str, int] = {}
    manifest_path = out_dir / DATA_MANIFEST_FILE
    if not manifest_path.is_file():
        failures.append(f"missing {DATA_MANIFEST_FILE}")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = config_hash(config_dir)
        recorded_hash = manifest.get("config_hash")
        if recorded_hash != current:
            failures.append(f"config_hash drift: manifest {recorded_hash!r} vs configs {current!r}")
        try:
            for split_report in manifest["splits"]:
                for entry in split_report["episodes"]:
                    recorded[entry["dir"]] = entry["seed"]
        except (KeyError, TypeError):
            failures.append(f"malformed {DATA_MANIFEST_FILE}: splits[].episodes[].dir/seed")
    checked = 0
    for split in SPLIT_SEED_BASES:
        failures.extend(_verify_split(out_dir, split, recorded))
        checked += len(_episode_dirs(out_dir / split))
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print(f"All checksums match. ({checked} episodes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Generate or verify the FINTRACE dataset.")
    parser.add_argument("--split", choices=("train", "dev", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="build in memory, write nothing")
    parser.add_argument("--verify-only", action="store_true", help="re-hash the written episodes")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / DATA_DIR)
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / CONFIG_DIR)
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    config_dir: Path = args.config_dir

    if args.verify_only:
        return verify_all(out_dir, config_dir)

    splits = list(SPLIT_SEED_BASES) if args.split == "all" else [args.split]
    # build every split in memory first: a failure anywhere leaves the disk untouched
    builds = [build_split(split, load_config(config_dir / CONFIG_FILES[split])) for split in splits]
    report = {"config_hash": config_hash(config_dir), "splits": [built[0] for built in builds]}
    if not args.dry_run:
        if args.split == "all":
            (out_dir / DATA_MANIFEST_FILE).unlink(missing_ok=True)  # stale until rewritten below
        for split, (_, built) in zip(splits, builds, strict=True):
            write_split(out_dir, split, built)
        if args.split == "all":
            write_data_manifest(out_dir, report)
        else:
            print(f"{DATA_MANIFEST_FILE} not updated: regenerate with --split all", file=sys.stderr)
    print(canonical_json(report))
    return 0


__all__ = [
    "CHECKSUMS_FILE",
    "CONFIG_FILES",
    "DATA_MANIFEST_FILE",
    "EPISODES_PER_SPLIT",
    "EPISODE_FILES",
    "MAX_CANDIDATE_SEEDS",
    "PRIVATE_FILES",
    "PUBLIC_FILES",
    "REPO_ROOT",
    "SEED_FILE",
    "SPLIT_SEED_BASES",
    "STAGING_PREFIX",
    "build_split",
    "config_hash",
    "generate_split",
    "load_config",
    "main",
    "read_episode",
    "read_seed",
    "verify_all",
    "verify_episode",
    "write_data_manifest",
    "write_episode",
    "write_split",
]

if __name__ == "__main__":
    raise SystemExit(main())
