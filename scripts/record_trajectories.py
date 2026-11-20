"""Record the baselines' trajectories as JSONL and leak-scan every file (``make trajectories``).

Usage::

    python scripts/record_trajectories.py [--split {train,dev}] [--policies all|a,b,...]
                                          [--out-dir trajectories]

One file per (policy, split, seed): ``<out-dir>/<policy>_<split>_<seed>.jsonl``, one canonical
JSON line per ``env.step()``. After recording, every file is scanned for private tokens; a hit
is printed and the exit code is 1. Timings go to stdout only, never into the files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Final

from fintrace.agents.base import BASELINE_NAMES
from fintrace.evaluator.runner import BASELINE_FACTORIES, REPO_ROOT, run_split, summarize
from fintrace.logging.trajectory import TRAJECTORY_SUFFIX, scan_trajectory

TRAJECTORIES_DIR: Final[Path] = REPO_ROOT / "trajectories"
SPLITS: Final[tuple[str, ...]] = ("train", "dev")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 when every recorded file passes the leak scan."""
    parser = argparse.ArgumentParser(description="Record baseline trajectories as JSONL.")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--policies", default="all", help="all or a comma-separated list")
    parser.add_argument("--seeds", type=int, default=None, help="first N episodes (default all)")
    parser.add_argument("--out-dir", type=Path, default=TRAJECTORIES_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    names = list(BASELINE_NAMES) if args.policies == "all" else args.policies.split(",")
    names = [name.strip() for name in names if name.strip()]
    unknown = [name for name in names if name not in BASELINE_FACTORIES]
    if unknown:
        print(f"unknown policies: {unknown} (known: {sorted(BASELINE_FACTORIES)})", file=sys.stderr)
        return 2
    out_dir: Path = args.out_dir
    started = time.monotonic()
    written: list[Path] = []
    for name in names:
        policy_started = time.monotonic()
        results = run_split(
            BASELINE_FACTORIES[name],
            args.split,
            limit=args.seeds,
            data_dir=args.data_dir,
            config_dir=args.config_dir,
            trajectory_dir=out_dir,
        )
        files = [Path(str(result["trajectory"])) for result in results]
        written.extend(files)
        lines = sum(path.read_text(encoding="utf-8").count("\n") for path in files)
        size = sum(path.stat().st_size for path in files)
        stats = summarize(results)
        print(
            f"{name:16s} episodes={len(files):2d} mean_reward={float(stats['mean_reward']):+.3f} "
            f"lines={lines:5d} bytes={size:8d} ({time.monotonic() - policy_started:.2f}s)"
        )
    hits = [hit for path in written for hit in scan_trajectory(path)]
    if hits:
        for hit in hits:
            print(hit, file=sys.stderr)
        print(f"leak scan: {len(hits)} hits in {len(written)} files", file=sys.stderr)
        return 1
    total = sum(path.stat().st_size for path in written)
    print(
        f"leak scan: {len(written)} files clean ({total} bytes, suffix {TRAJECTORY_SUFFIX}); "
        f"{time.monotonic() - started:.2f}s total"
    )
    return 0


__all__ = ["SPLITS", "TRAJECTORIES_DIR", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
