# FINTRACE

Reproducible, offline RL environment for training and evaluating financial-audit agents.
Built for Genesis Hackathon 2026 (FinTech track).

## Quick Start

```bash
make install   # pinned runtime deps + package (editable) + dev tools
make test      # fast tests (pytest -m "not slow"), 469 tests
make test-slow # seed sweeps and other slow tests (pytest -m slow), 20 tests
make test-all  # everything
make lint      # ruff over src/ tests/ scripts/ judge_adapter.py (zero findings)
make schemas   # regenerate the model-derived JSON Schemas after modifying src/fintrace/models.py
make data      # regenerate data/{train,dev}/ep_*/ (10 + 10 episodes) from config/{train,dev}.yaml
make data-verify # re-hash the committed episodes against checksums.sha256 (drift detection)
make data-dry  # build both splits in memory and print the report without writing
make baselines # 4 baselines on the dev split -> docs/results_dev.md
make attacks   # 20 reward-hack attackers on the dev split -> docs/reward_robustness_report.md (+ chart)
make oracle-gap # 24 policies x 5 error types -> docs/oracle_gap_report.md (+ heatmap)
make ocel      # sample investigation as an OCEL 2.0 log -> docs/sample_ocel_log.json
make docker-verify # build the image and run the jury protocol inside it offline (needs a Docker daemon)
make demo      # end-to-end demo episode (TODO)
```

Contributor and agent rules: [CLAUDE.md](CLAUDE.md). Every claim below maps to a test, a `make`
target or a committed document: [docs/compliance_matrix.md](docs/compliance_matrix.md).

## Judging criteria → evidence

| Criterion (points) | Evidence in this repository |
|---|---|
| Contract / isolation (25) | 17 jury-protocol tests in `tests/contract/test_judge_adapter.py` (reset/step envelopes, closed-mode reward 0, step and time budgets, 7-field scoring, invalid submissions); error contract `tests/contract/test_models.py::test_error_is_code_only` and `tests/contract/test_env_fuzz.py::test_fuzz_action_envelopes_never_crash`; schema drift `tests/contract/test_schemas.py::test_generated_schemas_match_committed`; leak scans `tests/security/test_isolation.py::test_no_truth_in_observations_via_100_random_steps` and `tests/security/test_leak_scan_full.py::test_leak_scan_20_seeds_100_steps_each` (slow); import firewall `tests/contract/test_runner.py::test_oracle_not_accessible_from_agents`; a real investigation through the adapter `tests/security/test_adapter_isolation.py::test_adapter_no_evaluator_leak_to_policy` |
| Data / ground truth (25) | consistency checks `tests/contract/test_validate.py`; per-type injector suites `tests/contract/test_injector_duplicate.py`, `test_injector_payment.py`, `test_injector_fx_error.py`, `test_injector_period.py` (each with `test_independent_check_passes_on_injected` and a 200-seed sweep); interacting scenarios `tests/contract/test_interacting_scenarios.py`; decoys `tests/contract/test_decoys.py::test_similar_invoices_not_caught_as_duplicate`; the committed dataset `tests/integration/test_data_integration.py::test_all_generated_episodes_validate` and `make data-verify`; Benford and 700-seed clean sweeps in `tests/contract/test_generator.py` (slow) |
| Reward reliability (25) | matcher `tests/reward/test_matcher.py` and `tests/reward/test_matcher_adversarial.py` (one-to-one maximum matching, tolerance edges, evidence supersets); formula and clipping `tests/reward/test_official_reward.py`, `tests/reward/test_reward_adversarial.py`; 20 attackers `tests/contract/test_attackers.py::test_attacker_scores_negative_on_dev` and `tests/integration/test_run_attacks.py::test_run_attacks_full_dev_split` (slow) → [reward robustness report](docs/reward_robustness_report.md); per-type floor `tests/contract/test_oracle_gap.py::test_attackers_uniform_floor`; transform invariance `tests/contract/test_robustness.py` |
| Reproducibility / generalization (15) | `tests/contract/test_generator.py::test_deterministic_regeneration`; byte-identical episode files `tests/contract/test_judge_adapter.py::test_generate_episode_writes_all_files`; disjoint train/dev seeds `tests/contract/test_generate_data.py::test_train_dev_disjoint_seeds`; full dataset regeneration `tests/integration/test_data_integration.py::test_generate_data_timing` (slow); trajectory rerun identity `tests/integration/test_record_trajectories.py::test_record_trajectories_dev_with_rerun_check` (slow); held-out invariance `tests/contract/test_robustness.py::test_remap_ids_rule_based_f1_matches`, `::test_shift_time_rule_based_f1_matches`; held-out dev results [docs/results_dev.md](docs/results_dev.md) and the [oracle-gap report](docs/oracle_gap_report.md) |
| Engineering (10) | CI `.github/workflows/ci.yml` (lint + fast suite, then the slow suite); `make lint` (ruff with mandatory annotations, no wall clock, no global RNG, no `print` in `src/`); offline guarantee `tests/integration/test_jury_protocol_offline.py::test_jury_protocol_runs_offline`, lean runtime `::test_runtime_requirements_lean`; `Dockerfile` (digest-pinned base, non-root, `src/` + `judge_adapter.py` only), `make docker-verify`, [docs/docker_offline_verification.md](docs/docker_offline_verification.md); 489 tests, 469 fast / 20 slow |

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

## Differentiators

- **D1 — Reward robustness against 20 purpose-built attackers.** Twenty policies in
  `src/fintrace/agents/attacks/` each probe one defence of the reward (flooding, empty or neighbouring
  evidence, tolerance edges, currency swaps, wrong or misspelt types, look-alike ids, foreign episode
  ids, budget burning). All 20 are defeated on the held-out dev split; the report is regenerated and
  byte-compared in CI. An auditor gets a reward that measures audits, not reward engineering.
- **D2 — OCEL 2.0 export.** `make ocel` turns an investigation (public episode + trajectory) into an
  object-centric event log (invoices, payments, FX rates, the episode and the agent's findings as
  objects; tool calls as events), schema-validated and leak-scanned, so the trace opens in
  process-mining tools such as PM4Py or Celonis.
- **D3 — Oracle-gap diagnostic.** `make oracle-gap` isolates one error type per episode and measures how
  far each of 24 policies is from the truth-reading oracle: the rule-based auditor is 4.1× further
  from the oracle on `payment_allocation` than on `duplicate_invoice`, while every attacker stays at
  the floor on every type. The environment yields a gradient of difficulty, not a pass/fail bit.
- **D4 — CPU behaviour-cloning proof (planned).** `experiments/bc_proof/` holds only its own
  requirements file today; a small imitation learner trained on recorded trajectories, on CPU, is the
  next step and will never enter the shipped image.
- **D5 — Environment Card (planned).** A one-page card (task, observation and action spaces, reward,
  budgets, known limitations, intended use) is planned; until then this README and
  [docs/design_note.md](docs/design_note.md) serve that purpose.

## Enterprise pillars

| Pillar | Mechanism in FINTRACE |
|---|---|
| P1 Ground-truth integrity | seeded generator, one injector per error type, independent re-derivation of every issue, provenance log |
| P2 Isolation and leak-proof scoring | `PublicEpisode` / `PrivateEpisode` split, evaluator-side `TruthStore`, import firewall, leak scanner over every surface |
| P3 Reproducibility and auditability | one `random.Random(seed)`, canonical serialisation, checksummed dataset, replayable trajectories |
| P4 Adversarial robustness | one-to-one matcher with exact tolerance and evidence rules, 20 attackers, transform invariance, oracle gap |
| P5 Interoperability with audit tooling | jury adapter, OCEL 2.0 export, offline digest-pinned Docker image |

The engineering narrative behind each pillar: [docs/design_note.md](docs/design_note.md).

## Why FINTRACE

Audit agents fail on exactly the cases that matter — duplicate invoices, partial payments, FX errors
and period misclassification hidden among clean look-alikes — and there is no reproducible
environment that trains on them with a reward the agent cannot game. FINTRACE generates seeded
ledger-and-document episodes with hidden ground truth, exposes them only through audited tools, and
scores submissions in an isolated evaluator, so that every reward is deterministic and leak-free.

## Results

- Baselines on the held-out dev split ([docs/results_dev.md](docs/results_dev.md)): oracle +0.975,
  rule_based +0.678, random −1.000, mark_everything −1.000 (10 episodes, config hash `aa6f0ac73b16`).
- D1 reward robustness ([report](docs/reward_robustness_report.md),
  [chart](docs/figures/robustness_chart.png), [data](docs/results_attacks.json)): 20 of 20 attackers
  defeated, mean rewards from −1.000 to −0.782, 15 of 20 on the −1.000 floor; the strongest no-audit
  heuristic, run as a reference, reaches −0.540.
- D3 oracle gap ([report](docs/oracle_gap_report.md), [heatmap](docs/figures/oracle_gap_matrix.png),
  [data](docs/oracle_gap_results.json)): rule_based gap 0.19 / 0.79 / 0.19 / 0.20 across
  duplicate_invoice / payment_allocation / fx_error / period_misclassification; attacker gaps on the
  four error types all in [1.33, 2.00]; 24 policies × 5 types × 3 paired episodes.
- D2 OCEL 2.0 export ([docs/sample_ocel_log.json](docs/sample_ocel_log.json)): the committed
  rule-based investigation of dev seed 10000 as 100 events over 343 objects (1 episode, 86 invoices,
  66 payments, 182 FX rates, 8 findings), valid against `schemas/ocel.schema.json`.
- Offline guarantee ([docs/docker_offline_verification.md](docs/docker_offline_verification.md)): the
  complete jury protocol runs under a socket guard on every CI run; the image audit and the
  `make docker-verify` command are documented for machines with a Docker daemon.
- Dataset: 10 train + 10 dev episodes (`data/manifest.json`; seeds 1001–1012 and 10000–10012 with the
  infeasible seeds listed as skipped), 8 injected issues per episode, verified by `make data-verify`.

## License

MIT — see [LICENSE](LICENSE).
