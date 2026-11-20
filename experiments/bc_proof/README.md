# BC proof: the environment is trainable on a CPU

A small MLP learns to imitate the rule-based auditor from its recorded trajectories, on an
ordinary CPU, in seconds, and predicts the expert's next action far above chance on held-out
episodes. Results: [`bc_proof_report.md`](bc_proof_report.md) (numbers, seeds, thresholds, limits),
[`bc_results.json`](bc_results.json), [`bc_learning_curve.png`](bc_learning_curve.png).

## Run

```bash
make bc-proof
```

That target creates a dedicated virtual environment under `experiments/bc_proof/.venv`, installs
the root `requirements.txt` (the same pinned runtime the image and the jury use, so the recorded
episodes are the pinned generator's), `fintrace` in editable mode and this directory's
`requirements.txt`, and runs `train_bc.py`. `torch==2.15.0` matches `2.15.0+cpu` on the PyTorch
CPU index and `2.15.0` on PyPI; pip prefers the local-version build, so the CPU wheel is taken
whenever that index is reachable and the PyPI wheel otherwise. It prints the
learning curve, the held-out accuracies and `PASS`/`FAIL`; exit 1 when the model does not beat
chance and the majority baseline with margin, exit 2 when torch or numpy is missing. Behind a proxy that blocks the
PyTorch index, use `make bc-proof BC_INDEX_URL=https://pypi.org/simple` (PyPI's torch wheel carries
CUDA libraries but runs on the CPU all the same).

Manual equivalent:

```bash
python3.11 -m venv experiments/bc_proof/.venv
experiments/bc_proof/.venv/bin/pip install -r requirements.txt -e . \
    -r experiments/bc_proof/requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
experiments/bc_proof/.venv/bin/python experiments/bc_proof/train_bc.py
```

## What it does

1. Records expert trajectories with the standard runner: the 10 committed train seeds plus 30
   extra train-config seeds (infeasible seeds skipped) for training; the 10 committed dev seeds,
   never seen in training, for validation. Trajectories go to a temporary directory and are
   never committed.
2. `featurize.py` (pure Python, no torch, no numpy) turns every step into 37 features computed
   from the public observations seen so far: listing progress per table, what was already
   inspected, the last action and its outcome, and the candidate structure an auditor would track
   (duplicate-looking invoices, foreign-currency payments, unread payment documents).
3. `train_bc.py` trains a two-head MLP (tool: 5 classes; tool + table: 8 classes) with torch on
   the CPU, seeded, and reports held-out accuracy against chance (0.20 / 0.125) and the majority
   baseline. Thresholds: tool accuracy >= 0.80, fine accuracy >= 0.75, both >= majority + 0.15.

## Isolation

- torch and numpy are dependencies of this directory only (`experiments/bc_proof/requirements.txt`).
  They are not in the root `requirements.txt` (the only thing the Docker image installs), not in
  `pyproject.toml`, and nothing under `src/` or `judge_adapter.py` imports them.
- The main test suite stays torch-free and guards this: `tests/contract/test_bc_proof_isolation.py`
  scans every source file, the dependency files, the lint scope and a fresh interpreter's
  `sys.modules`, and pins the featurizer's layout and determinism without importing torch.
- `make bc-proof` is not part of `make test`, `make test-all` or `make lint`.

## Limits

No rollout reward is reported. The expert's `submit` carries findings (amounts, ids, evidence)
produced by its reasoning, not an action label; a rollout would need hand-written argument
selection and finding construction and would measure those heuristics rather than the learned
policy. The expert is procedural: the two progress features let the model learn its fixed budget
schedule, and the hardest step is `submit` (lowest per-class recall in the report), whose timing
depends on the expert's finding construction rather than on the public prefix alone. torch is
not bit-reproducible across machines, so the thresholds are regression guards with margin, not
pins; the numbers in the report come from one seeded run.
