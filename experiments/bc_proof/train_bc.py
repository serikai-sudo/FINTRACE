"""CPU behaviour-cloning proof: a small MLP imitates the rule-based auditor from its trajectories.

Usage::

    experiments/bc_proof/.venv/bin/python experiments/bc_proof/train_bc.py [--epochs 40]
        [--extra-episodes 30] [--work-dir DIR] [--out-dir experiments/bc_proof]

1. Records expert trajectories with the existing runner: the committed train seeds plus extra
   train-config seeds (infeasible seeds skipped) for training, the committed dev seeds (held out)
   for validation. Trajectories go to a temporary work dir and are never committed.
2. Featurises every step with the torch-free ``featurize.py`` (37 public-only features) and
   trains a two-head MLP (tool: 5 classes, fine target: 8 classes) with torch on the CPU, seeded.
3. Reports per-epoch loss/accuracy, final held-out accuracy against chance and the majority
   baseline, and writes ``bc_proof_report.md``, ``bc_results.json`` and (when matplotlib is
   available) ``bc_learning_curve.png``. Exit 0 = PASS (thresholds below), 1 = FAIL, 2 = setup
   (torch or numpy missing).

Deliberately no environment rollout: the expert's ``submit`` carries findings (amounts, ids,
evidence) that are the product of its reasoning, not an action label; a rollout would need
hand-written argument selection and finding construction and would measure those heuristics,
not the learned policy. This proof is the honest core: the environment's public observations
carry enough signal for a learner to predict the expert's next action far above chance.
torch is a dependency of this directory only (see requirements.txt); never of src/ or the image.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Final

try:
    import numpy as np
    import torch
except ImportError as exc:  # exit 2 = setup: the dedicated venv is missing or incomplete
    print(f"{exc.name} is not installed: pip install -r experiments/bc_proof/requirements.txt", file=sys.stderr)
    raise SystemExit(2) from None

from fintrace.agents.rule_based import RuleBasedReconciliation
from fintrace.environment.env import FintraceEnv
from fintrace.evaluator.runner import BASELINE_FACTORIES, load_split_config, run_baseline, run_split
from fintrace.generator.injectors.base import InjectorError
from fintrace.logging.trajectory import TrajectoryLogger, trajectory_path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # featurize.py lives next to this file
from featurize import FEATURE_NAMES, FINE_CLASSES, TOOLS, featurize_file  # noqa: E402

HERE: Final[Path] = Path(__file__).resolve().parent
EXPERT: Final[str] = "rule_based"
SEED: Final[int] = 0
EXTRA_SEED_BASE: Final[int] = 2000
EXTRA_SEED_SPAN: Final[int] = 200
DEFAULT_EXTRA_EPISODES: Final[int] = 30
DEFAULT_EPOCHS: Final[int] = 40
BATCH_SIZE: Final[int] = 64
HIDDEN: Final[int] = 64
LEARNING_RATE: Final[float] = 1e-3
TOOL_ACC_MIN: Final[float] = 0.80  # chance is 1/5 = 0.20
FINE_ACC_MIN: Final[float] = 0.75  # chance is 1/8 = 0.125
MARGIN_OVER_MAJORITY: Final[float] = 0.15
REPORT_FILE: Final[str] = "bc_proof_report.md"
RESULTS_FILE: Final[str] = "bc_results.json"
FIGURE_FILE: Final[str] = "bc_learning_curve.png"


# ---------------------------------------------------------------- data
def record_expert(work: Path, extra_episodes: int) -> tuple[list[Path], list[Path], list[int]]:
    """Expert trajectories: (train files, held-out dev files, skipped extra seeds)."""
    train_results = run_split(BASELINE_FACTORIES[EXPERT], "train", trajectory_dir=work / "train")
    train_files = [Path(str(r["trajectory"])) for r in train_results]
    dev_results = run_split(BASELINE_FACTORIES[EXPERT], "dev", trajectory_dir=work / "dev")
    dev_files = [Path(str(r["trajectory"])) for r in dev_results]
    config = load_split_config("train")
    env = FintraceEnv()
    skipped: list[int] = []
    seed = EXTRA_SEED_BASE
    while len(train_files) < len(train_results) + extra_episodes and seed < EXTRA_SEED_BASE + EXTRA_SEED_SPAN:
        path = trajectory_path(work / "train", EXPERT, "extra", seed)
        try:
            result = run_baseline(RuleBasedReconciliation(), env, seed, config, logger=TrajectoryLogger(path))
        except InjectorError:
            skipped.append(seed)
            seed += 1
            continue
        if not (result["terminated"] or result["truncated"]):
            raise RuntimeError(f"extra seed {seed}: episode did not end")
        train_files.append(path)
        seed += 1
    if len(train_files) < len(train_results) + extra_episodes:
        raise RuntimeError("not enough feasible extra seeds for the requested training set")
    return train_files, dev_files, skipped


def load_dataset(files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    tools: list[int] = []
    fine: list[int] = []
    for path in files:
        x, yt, yf = featurize_file(path)
        rows += x
        tools += yt
        fine += yf
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(tools, dtype=np.int64),
        np.asarray(fine, dtype=np.int64),
    )


def distribution(labels: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    counts = np.bincount(labels, minlength=len(names))
    return {name: float(counts[i] / max(1, len(labels))) for i, name in enumerate(names)}


# ---------------------------------------------------------------- model
def train(
    x_train: np.ndarray,
    y_tool: np.ndarray,
    y_fine: np.ndarray,
    x_val: np.ndarray,
    v_tool: np.ndarray,
    v_fine: np.ndarray,
    *,
    epochs: int,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray, str]:
    """Seeded CPU training; returns the per-epoch history, final val predictions, torch version."""
    from torch import nn

    torch.manual_seed(SEED)
    torch.set_num_threads(min(2, torch.get_num_threads()))  # the jury box has 2 CPUs
    rng = random.Random(SEED)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # train-constant features (e.g. calculator counts) pass through
    xt = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    xv = torch.tensor((x_val - mean) / std, dtype=torch.float32)
    yt_tool, yt_fine = torch.tensor(y_tool), torch.tensor(y_fine)
    yv_tool, yv_fine = torch.tensor(v_tool), torch.tensor(v_fine)

    class TwoHeadMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(xt.shape[1], HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU()
            )
            self.tool = nn.Linear(HIDDEN, len(TOOLS))
            self.fine = nn.Linear(HIDDEN, len(FINE_CLASSES))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.shared(x)
            return self.tool(h), self.fine(h)

    model = TwoHeadMLP()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    indices = list(range(len(xt)))

    def accuracy(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        model.eval()
        with torch.no_grad():
            pa, pb = model(x)
        model.train()
        return (
            float((pa.argmax(1) == a).float().mean()),
            float((pb.argmax(1) == b).float().mean()),
        )

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        total_loss = 0.0
        for start in range(0, len(indices), BATCH_SIZE):
            batch = torch.tensor(indices[start : start + BATCH_SIZE])
            optimiser.zero_grad()
            out_tool, out_fine = model(xt[batch])
            loss = loss_fn(out_tool, yt_tool[batch]) + loss_fn(out_fine, yt_fine[batch])
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach()) * len(batch)
        train_tool, train_fine = accuracy(xt, yt_tool, yt_fine)
        val_tool, val_fine = accuracy(xv, yv_tool, yv_fine)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(indices),
                "train_tool_acc": train_tool,
                "train_fine_acc": train_fine,
                "val_tool_acc": val_tool,
                "val_fine_acc": val_fine,
            }
        )
    model.eval()
    with torch.no_grad():
        pred_tool, pred_fine = model(xv)
    return history, pred_tool.argmax(1).numpy(), pred_fine.argmax(1).numpy(), torch.__version__


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, names: tuple[str, ...]) -> dict[str, float | None]:
    recall: dict[str, float | None] = {}
    for index, name in enumerate(names):
        mask = y_true == index
        recall[name] = float((y_pred[mask] == index).mean()) if mask.any() else None
    return recall


# ---------------------------------------------------------------- outputs
def render_report(results: dict[str, Any]) -> str:
    m = results["metrics"]
    data = results["data"]
    lines = [
        "# BC proof: a CPU-trained MLP imitates the rule-based auditor",
        "",
        f"Status: **{results['status']}** (thresholds: held-out tool accuracy >= {TOOL_ACC_MIN:.2f}, "
        f"fine accuracy >= {FINE_ACC_MIN:.2f}, both >= majority baseline + {MARGIN_OVER_MAJORITY:.2f}).",
        "",
        "Claim: FINTRACE's public observations carry enough signal for a learner to predict the "
        "expert's next action far above chance, on an ordinary CPU, in seconds. This is the honest "
        "core of 'the environment is trainable'; there is no rollout (see Limits).",
        "",
        "## Setup",
        "",
        f"- Expert: `{EXPERT}` baseline, recorded with the standard runner.",
        f"- Train: {data['train_episodes']} episodes / {data['train_steps']} steps = the "
        f"{data['committed_train_episodes']} committed train seeds + {data['extra_episodes']} extra "
        f"train-config seeds from {EXTRA_SEED_BASE} (skipped as infeasible: {data['skipped_extra_seeds']}).",
        f"- Validation (held out): {data['dev_episodes']} committed dev episodes / {data['dev_steps']} steps.",
        f"- Features: {len(FEATURE_NAMES)} public-only prefix features (`featurize.py`, torch-free).",
        f"- Model: MLP {len(FEATURE_NAMES)} -> {HIDDEN} -> {HIDDEN} -> two heads ({len(TOOLS)} tools, "
        f"{len(FINE_CLASSES)} fine targets); Adam lr {LEARNING_RATE}, batch {BATCH_SIZE}, "
        f"{results['epochs']} epochs, seed {SEED} (torch.manual_seed + random.Random).",
        f"- Platform: {results['platform']['python']}, torch {results['platform']['torch']}, "
        f"{results['platform']['threads']} CPU thread(s), {results['platform']['machine']}; "
        f"wall time {results['wall_time_s']:.1f} s (recording {results['record_time_s']:.1f} s, "
        f"training {results['train_time_s']:.1f} s).",
        "",
        "## Result (held-out dev seeds)",
        "",
        "| Target | Classes | Chance | Majority baseline | BC accuracy | Threshold |",
        "|---|---:|---:|---:|---:|---:|",
        f"| tool | {len(TOOLS)} | {1 / len(TOOLS):.3f} | {m['majority_tool']:.3f} | **{m['val_tool_acc']:.3f}** | {TOOL_ACC_MIN:.2f} |",
        f"| tool + table | {len(FINE_CLASSES)} | {1 / len(FINE_CLASSES):.3f} | {m['majority_fine']:.3f} | **{m['val_fine_acc']:.3f}** | {FINE_ACC_MIN:.2f} |",
        "",
        "Per-class recall (tool head, held out):",
        "",
        "| Tool | Share of expert steps | Recall |",
        "|---|---:|---:|",
    ]
    for name in TOOLS:
        share = data["dev_tool_distribution"][name]
        recall = m["tool_recall"][name]
        lines.append(f"| {name} | {share:.3f} | {'n/a' if recall is None else f'{recall:.3f}'} |")
    lines += ["", "## Learning curve", "", "| Epoch | Train loss | Train tool acc | Val tool acc | Val fine acc |", "|---:|---:|---:|---:|---:|"]
    shown = [h for h in results["history"] if h["epoch"] in (1, 2, 5, 10, 20, 30, results["epochs"])]
    for h in shown:
        lines.append(
            f"| {int(h['epoch'])} | {h['train_loss']:.3f} | {h['train_tool_acc']:.3f} | "
            f"{h['val_tool_acc']:.3f} | {h['val_fine_acc']:.3f} |"
        )
    if results.get("figure"):
        lines += ["", f"![learning curve]({results['figure']})"]
    lines += [
        "",
        "## Limits (read before quoting)",
        "",
        "- No rollout reward: the expert's `submit` carries findings (amounts, ids, evidence) produced by "
        "its reasoning, not an action label. Driving the environment would need hand-written argument "
        "selection and finding construction, and the reward would measure those heuristics rather than "
        "the learned policy. The claim here is action prediction only.",
        "- The expert is deterministic and procedural, so its action sequence is highly learnable; the "
        "proof shows the observations expose that structure, not that a hard auditing problem is solved. "
        "The two progress features (`step_fraction`, `remaining_fraction`) let the model learn the "
        "expert's fixed budget schedule; `submit` has the lowest recall because its timing depends on "
        "the expert's finding construction, not on the public prefix alone.",
        "- torch is not bit-reproducible across machines; the numbers above are from one seeded run and "
        "the thresholds are regression guards with margin, not pins.",
        "- Trajectories are regenerated on every run and never committed; everything in this directory "
        "is outside the jury path, the main test suite and the Docker image.",
        "",
        f"Regenerate: `make bc-proof` (results in `{RESULTS_FILE}`).",
        "",
    ]
    return "\n".join(lines)


def render_figure(history: list[dict[str, float]], out: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    ax_loss.plot(epochs, [h["train_loss"] for h in history], color="#4a4a4a")
    ax_loss.set_title("train loss (sum of two heads)")
    ax_loss.set_xlabel("epoch")
    ax_acc.plot(epochs, [h["val_tool_acc"] for h in history], label="val tool accuracy")
    ax_acc.plot(epochs, [h["val_fine_acc"] for h in history], label="val tool+table accuracy")
    ax_acc.axhline(1 / len(TOOLS), color="#b71c1c", linestyle="--", linewidth=1, label="chance (tool)")
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_title("held-out accuracy")
    ax_acc.legend(loc="lower right", fontsize=8)
    fig.suptitle("FINTRACE BC proof: CPU MLP imitating the rule-based auditor")
    fig.tight_layout()
    fig.savefig(out, dpi=100, metadata={"Software": None})
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU behaviour-cloning proof.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--extra-episodes", type=int, default=DEFAULT_EXTRA_EPISODES)
    parser.add_argument("--work-dir", type=Path, default=None, help="trajectory scratch dir")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args(argv)
    started = time.monotonic()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="fintrace_bc_"))
    train_files, dev_files, skipped = record_expert(work, args.extra_episodes)
    record_time = time.monotonic() - started
    x_train, t_train, f_train = load_dataset(train_files)
    x_val, t_val, f_val = load_dataset(dev_files)
    print(f"recorded {len(train_files)} train / {len(dev_files)} dev episodes "
          f"({len(x_train)} / {len(x_val)} steps) in {record_time:.1f}s; skipped extra seeds {skipped}")
    train_started = time.monotonic()
    history, pred_tool, pred_fine, torch_version = train(
        x_train, t_train, f_train, x_val, t_val, f_val, epochs=args.epochs
    )
    train_time = time.monotonic() - train_started
    for h in history:
        if h["epoch"] in (1, 2, 5, 10, 20, 30, args.epochs):
            print(f"epoch {int(h['epoch']):3d}  loss {h['train_loss']:.3f}  train tool {h['train_tool_acc']:.3f}"
                  f"  val tool {h['val_tool_acc']:.3f}  val fine {h['val_fine_acc']:.3f}")
    val_tool_acc = float((pred_tool == t_val).mean())
    val_fine_acc = float((pred_fine == f_val).mean())
    majority_tool = float(np.bincount(t_val).max() / len(t_val))
    majority_fine = float(np.bincount(f_val).max() / len(f_val))
    passed = (
        val_tool_acc >= TOOL_ACC_MIN
        and val_fine_acc >= FINE_ACC_MIN
        and val_tool_acc >= majority_tool + MARGIN_OVER_MAJORITY
        and val_fine_acc >= majority_fine + MARGIN_OVER_MAJORITY
    )
    results: dict[str, Any] = {
        "status": "PASS" if passed else "FAIL",
        "epochs": args.epochs,
        "seed": SEED,
        "thresholds": {
            "tool_acc_min": TOOL_ACC_MIN,
            "fine_acc_min": FINE_ACC_MIN,
            "margin_over_majority": MARGIN_OVER_MAJORITY,
        },
        "data": {
            "committed_train_episodes": len(train_files) - args.extra_episodes,
            "extra_episodes": args.extra_episodes,
            "train_episodes": len(train_files),
            "train_steps": int(len(x_train)),
            "dev_episodes": len(dev_files),
            "dev_steps": int(len(x_val)),
            "skipped_extra_seeds": skipped,
            "dev_tool_distribution": distribution(t_val, TOOLS),
            "dev_fine_distribution": distribution(f_val, FINE_CLASSES),
            "n_features": len(FEATURE_NAMES),
        },
        "metrics": {
            "val_tool_acc": val_tool_acc,
            "val_fine_acc": val_fine_acc,
            "majority_tool": majority_tool,
            "majority_fine": majority_fine,
            "chance_tool": 1 / len(TOOLS),
            "chance_fine": 1 / len(FINE_CLASSES),
            "tool_recall": per_class_recall(t_val, pred_tool, TOOLS),
        },
        "history": history,
        "platform": {
            "python": platform.python_version(),
            "torch": torch_version,
            "threads": torch.get_num_threads(),
            "machine": platform.machine(),
        },
        "record_time_s": record_time,
        "train_time_s": train_time,
        "wall_time_s": time.monotonic() - started,
    }
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_written = render_figure(history, out_dir / FIGURE_FILE)
    results["figure"] = FIGURE_FILE if figure_written else None
    (out_dir / RESULTS_FILE).write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / REPORT_FILE).write_text(render_report(results), encoding="utf-8")
    print(f"held-out tool accuracy {val_tool_acc:.3f} (chance {1 / len(TOOLS):.2f}, majority {majority_tool:.3f}); "
          f"tool+table accuracy {val_fine_acc:.3f} (chance {1 / len(FINE_CLASSES):.3f}, majority {majority_fine:.3f})")
    print(f"wrote {out_dir / REPORT_FILE}, {out_dir / RESULTS_FILE}"
          + (f", {out_dir / FIGURE_FILE}" if figure_written else " (matplotlib missing: no figure)"))
    print(f"bc-proof: {results['status']} (wall {results['wall_time_s']:.1f}s, torch {torch_version}, CPU)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
