"""Render the oracle-gap heatmap from docs/oracle_gap_results.json.

Usage::

    python scripts/plot_oracle_gap.py [--results docs/oracle_gap_results.json]
                                      [--out docs/figures/oracle_gap_matrix.png]

Rows: the policies in display order (4 baselines above a separator, then the 20 attackers).
Columns: the five diagnostic episode types. Cell: oracle gap = mean(oracle) - mean(policy) on
that type, green (0, matches the oracle) to red (2, the -1 floor); the colour scale is fixed to
the theoretical range so reruns stay comparable. 1400 x 1000 px (14 x 10 in at 100 dpi), no
PNG metadata, so a rerun reproduces the file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # the backend must be chosen before pyplot loads
from matplotlib.colors import Normalize

from fintrace.diagnostics.oracle_gap import BASELINE_ORDER

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS: Final[Path] = REPO_ROOT / "docs" / "oracle_gap_results.json"
DEFAULT_OUT: Final[Path] = REPO_ROOT / "docs" / "figures" / "oracle_gap_matrix.png"
FIGURE_SIZE: Final[tuple[float, float]] = (14.0, 10.0)
DPI: Final[int] = 100
GAP_MIN: Final[float] = 0.0  # a policy that matches the oracle
GAP_MAX: Final[float] = 2.0  # the -1 floor against a +1 oracle
COLORMAP: Final[str] = "RdYlGn_r"
TEXT_LUMINANCE_SPLIT: Final[float] = 0.55


def _cell_text(gap: float) -> str:
    text = f"{gap:.2f}"
    return "0.00" if text == "-0.00" else text


def _text_colour(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _ = rgba
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "black" if luminance >= TEXT_LUMINANCE_SPLIT else "white"


def render(results: dict[str, Any], out: Path) -> None:
    """Write the heatmap for an ``oracle_gap_results.json`` payload to ``out``."""
    types = [str(t) for t in results["types"]]
    policies = [str(p) for p in results["policies"]]
    matrix = [[float(results["gap"][policy][t]) for t in types] for policy in policies]
    n_per_type = int(results["n_per_type"])
    cmap = matplotlib.colormaps[COLORMAP]
    norm = Normalize(GAP_MIN, GAP_MAX)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    fig.subplots_adjust(left=0.20, right=0.88, top=0.93, bottom=0.07)  # fixed: exact pixel size
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for row, gaps in enumerate(matrix):
        for col, gap in enumerate(gaps):
            colour = _text_colour(cmap(norm(gap)))
            ax.text(col, row, _cell_text(gap), ha="center", va="center", fontsize=10, color=colour)
    ax.set_xticks(range(len(types)), labels=types, fontsize=10)
    ax.set_yticks(range(len(policies)), labels=policies, fontsize=9)
    ax.set_xticks([i - 0.5 for i in range(1, len(types))], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, len(policies))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    n_baselines = sum(1 for policy in policies if policy in BASELINE_ORDER)
    if 0 < n_baselines < len(policies):
        ax.axhline(n_baselines - 0.5, color="black", linewidth=2.0)  # baselines | attackers
    ax.set_xlabel("diagnostic episode type (exactly one injected issue; clean = none)", fontsize=10)
    ax.set_title(
        f"FINTRACE oracle gap: {len(policies)} policies x {len(types)} error types, "
        f"{n_per_type} paired episodes per type",
        fontsize=13,
    )
    cax = fig.add_axes((0.90, 0.07, 0.02, 0.86))
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(
        "oracle gap = mean(oracle reward) - mean(policy reward)   [0 = oracle, 2 = floor]",
        fontsize=10,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Plot the oracle-gap heatmap.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    render(json.loads(args.results.read_text(encoding="utf-8")), args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


__all__ = [
    "COLORMAP",
    "DEFAULT_OUT",
    "DEFAULT_RESULTS",
    "DPI",
    "FIGURE_SIZE",
    "GAP_MAX",
    "GAP_MIN",
    "main",
    "render",
]

if __name__ == "__main__":
    raise SystemExit(main())
