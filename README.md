# FINTRACE

Reproducible, offline RL environment for training and evaluating financial-audit agents.
Built for Genesis Hackathon 2026 (FinTech track).

## Quick Start

```bash
make install   # pinned runtime deps + package (editable) + dev tools
make test      # fast tests (pytest -m "not slow")
make test-slow # seed sweeps and other slow tests (pytest -m slow)
make test-all  # everything
make schemas   # regenerate the model-derived JSON Schemas after modifying src/fintrace/models.py
make demo      # end-to-end demo episode (TODO)
```

Contributor and agent rules: [CLAUDE.md](CLAUDE.md).

## Why FINTRACE

Audit agents fail on exactly the cases that matter — duplicate invoices, partial payments, FX errors
and period misclassification hidden among clean look-alikes — and there is no reproducible
environment that trains on them with a reward the agent cannot game. FINTRACE generates seeded
ledger-and-document episodes with hidden ground truth, exposes them only through audited tools, and
scores submissions in an isolated evaluator, so that every reward is deterministic and leak-free.

## Results

- Dev-split results: _TBD — `docs/results_dev.md`_
- Reward robustness report: _TBD — `docs/reward_robustness_report.md`_

## License

MIT — see [LICENSE](LICENSE).
