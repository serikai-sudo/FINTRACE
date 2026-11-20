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
make data      # regenerate data/{train,dev}/ep_*/ (10 + 10 episodes) from config/{train,dev}.yaml
make data-verify # re-hash the committed episodes against checksums.sha256 (drift detection)
make data-dry  # build both splits in memory and print the report without writing
make demo      # end-to-end demo episode (TODO)
```

Contributor and agent rules: [CLAUDE.md](CLAUDE.md).


## Jury adapter

`judge_adapter.py` (repository root, `/app/judge_adapter.py` in the image) is the connection protocol the
Genesis jury harness calls. It is a thin translation layer over the existing environment, generator and
evaluator; it adds no logic of its own.

Entry points:

```python
generate_episode(seed: int, config: dict, output_dir: str) -> None
make_env()  # -> object with reset(seed, config) and step(action)
score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict
```

- `generate_episode` builds the episode for `seed` under `config["team_config"]` (an `EpisodeConfig`,
  see `config/dev.yaml`) and writes `manifest.json`, `ledger.csv`, `invoices.jsonl`, `payments.jsonl`,
  `fx.csv`, `policy.json` (public) plus `truth.json`, `provenance.json` (closed), `seed.txt` and
  `checksums.sha256` into `output_dir`. The same seed and config give byte-identical files.
- `make_env().reset(seed, config)` with `config = {"mode": "closed", "episode_dir": "<dir>"}` loads that
  episode exactly as written (same `episode_id`, nothing regenerated; a second reset starts it over);
  `{"mode": "train", "team_config": {...}}` generates from the seed. The first observation is
  `{"task": str, "manifest": object, "policy": object, "remaining_steps": 100}`; `info` is `{}`.
- `step(action)` takes `{"name": <action>, "args": {...}}` and returns
  `(observation, reward, terminated, truncated, info)` with
  `observation = {"result": value_or_null, "error": null_or_{"code", "message"}, "remaining_steps": int}`.
  In closed mode the reward returned to the policy is `0` on every step, including the terminal one;
  the true return is available only through `score_episode`.
- `score_episode` loads the closed truth from `episode_dir` and returns
  `{"tp", "fp", "fn", "precision", "recall", "f1", "return"}`; a structurally invalid submission gives
  `null` metrics and `"return": -1`.

Actions and result shapes (canonical table names: `ledger`, `invoices`, `payments`, `fx`):

| name | args | result |
|---|---|---|
| `list_records` | `table`, `cursor` (first page: `null`), `limit` (≤ 50) | `{"records": [object], "next_cursor": null_or_string}` |
| `get_record` | `table`, `id` | the record object |
| `get_document` | `doc_id` | the invoice or payment document |
| `calculator` | `expression` | decimal string, e.g. `"0.3"` |
| `submit` | `{"findings": <findings.json>}` | `{"accepted": true}` or a structural error |

Error codes are `INVALID_ACTION`, `INVALID_TABLE`, `NOT_FOUND`, `INVALID_CURSOR`, `CALC_ERROR`,
`EPISODE_OVER`, `INVALID_SUBMISSION`; `message` is a fixed generic text (no paths, seeds or closed data).
Issue types in findings: `duplicate_invoice`, `payment_allocation`, `fx_error`, `period_misclassification`
(see `schemas/submission.schema.json`). Budgets: 100 actions per episode (submit counts as an action but
not as an inspection step; 100 actions without submit truncate the episode) and 60 s of wall clock from
reset to the terminal step (exceeding it truncates with empty findings). The clock starts inside `reset`
and is checked as each action arrives, so time spent before `reset` or inside a single policy call is not
interrupted; a harness that counts from process start leaves correspondingly less for the actions.

Run inside the container (working directory `/app`, `PYTHONPATH=/app/src:/app`):

```bash
docker run --rm --network none --cpus 2 --memory 4g --entrypoint python fintrace:0.1.0 -c "import judge_adapter"
```

```python
import judge_adapter

judge_adapter.generate_episode(10000, {"team_config": {...}}, "/tmp/ep")
env = judge_adapter.make_env()
observation, info = env.reset(0, {"mode": "closed", "episode_dir": "/tmp/ep"})
observation, reward, terminated, truncated, info = env.step(
    {"name": "list_records", "args": {"table": "invoices", "cursor": None, "limit": 50}}
)
judge_adapter.score_episode("/tmp/ep", {"findings": [...]}, inspection_steps=1)
```

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
