"""Render the reward-robustness bar chart from docs/results_attacks.json.

Usage::

    python scripts/plot_robustness.py [--results docs/results_attacks.json]
                                      [--out docs/figures/robustness_chart.png]

Horizontal bars, one per attacker, sorted by mean reward; the red zone (-1 .. -0.3) is where a
defeated attack lands, yellow (-0.3 .. 0) would be a warning, green (0 .. 1) a successful hack.
1200 x 800 px (12 x 8 in at 100 dpi), no PNG metadata, so a rerun reproduces the file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # the backend must be chosen before pyplot loads

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS: Final[Path] = REPO_ROOT / "docs" / "results_attacks.json"
DEFAULT_OUT: Final[Path] = REPO_ROOT / "docs" / "figures" / "robustness_chart.png"
THRESHOLD: Final[float] = -0.3
FIGURE_SIZE: Final[tuple[float, float]] = (12.0, 8.0)
DPI: Final[int] = 100


def render(results: dict[str, Any], out: Path) -> None:
    """Write the chart for a ``results_attacks.json`` payload to ``out``."""
    rows = sorted(
        ((a["name"], float(a["stats"]["mean_reward"])) for a in results["attackers"]),
        key=lambda item: (item[1], item[0]),
    )
    names = [name for name, _ in rows]
    means = [mean for _, mean in rows]
    defeated = sum(1 for mean in means if mean <= THRESHOLD)
    n_episodes = max((len(a["episodes"]) for a in results["attackers"]), default=0)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    ax.axvspan(-1.0, THRESHOLD, color="#f4c7c3", alpha=0.6, label="defeated (mean <= -0.3)")
    ax.axvspan(THRESHOLD, 0.0, color="#fde9a9", alpha=0.6, label="reward hole (-0.3 .. 0)")
    ax.axvspan(0.0, 1.0, color="#c8e6c9", alpha=0.6, label="reward hole (> 0)")
    ax.axvline(THRESHOLD, color="#b71c1c", linestyle="--", linewidth=1.2)
    ax.axvline(-1.0, color="#4a4a4a", linewidth=1.0)
    # bars grow from the -1 floor up to the attacker's mean: how far each attack climbed
    ax.barh(names, [mean + 1.0 for mean in means], left=-1.0, color="#8e1b1b", height=0.6)
    ax.plot(means, names, "o", color="#4a0f0f", markersize=5)
    for index, mean in enumerate(means):
        ax.text(mean + 0.03, index, f"{mean:+.3f}", va="center", ha="left", fontsize=9)
    ax.set_xlim(-1.05, 1.0)
    split = str(results.get("split", "dev"))
    ax.set_xlabel(f"mean official reward over the {n_episodes} held-out {split} episodes")
    outcome = "all defeated" if defeated == len(names) else f"{defeated} of {len(names)} defeated"
    ax.set_title(
        f"FINTRACE reward robustness: {len(names)} reward-hack attackers, {outcome}", fontsize=13
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle=":", linewidth=0.6)
    ax.invert_yaxis()  # worst (most negative) at the top keeps the reading order stable
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Plot the reward robustness chart.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    render(json.loads(args.results.read_text(encoding="utf-8")), args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


__all__ = ["DEFAULT_OUT", "DEFAULT_RESULTS", "DPI", "FIGURE_SIZE", "THRESHOLD", "main", "render"]

if __name__ == "__main__":
    raise SystemExit(main())
