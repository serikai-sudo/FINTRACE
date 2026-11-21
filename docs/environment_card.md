# FINTRACE Environment Card

Version `0.1.0`, card date 2026-11-20 (fixed build date; every number below is read from a committed artefact named in the last section, none is typed by hand).

## Environment details

- Task: audit a seeded synthetic ledger against its source documents (invoices, payments, FX rates) and submit the injected errors as findings; a hidden evaluator scores them against the exact ground truth.
- Observation space (jury protocol): reset returns `{task, manifest, policy, remaining_steps}`; every step returns `{result, error, remaining_steps}` with `error` null or `{code, message}` (jury adapter envelope; the internal environment returns the error code only).
- Action space: 5 tools, `list_records`, `get_record`, `get_document`, `calculator`, `submit`, sent as `{"name", "args"}`; `list_records` pages hold at most 50 records; a submission holds at most 30 findings over 4 issue types.
- Reward: `clip(2 * F1 - 1 - 0.002 * inspection_steps, -1, 1)`; every step before the terminal submit is an inspection step; in the jury's closed mode the policy sees reward 0 and the return is computed by `score_episode` only.
- Budgets: 100 actions per episode, 60 s of wall clock per episode, 2 CPU / 4 GB RAM, no network (jury machine).

## Data

- Generator: seeded synthetic double-entry books (`random.Random(seed)` only), validated for balance, totals, allocations and period bounds before any error is injected; the first-digit distribution is checked against Benford's law by chi-squared in the test suite.
- Injected error types (count per committed dev episode): duplicate_invoice (1), payment_allocation (1), fx_error (1), period_misclassification (1); interacting scenarios: scenario_a (1), scenario_b (1); issues per dev episode: 8.
- Decoys (legitimate look-alikes, per dev episode): repeated_payment_amounts (2), split_payments (1), refunds (1), similar_invoices (1).
- Split: 10 train + 10 dev committed episodes (seed bases 1000 / 10000; infeasible seeds skipped: train [1000, 1003, 1004], dev [10009, 10010, 10011]); config hash `aa6f0ac73b16`; every public file checksummed.

## Metrics

| Metric | Value | Evidence |
|---|---|---|
| Determinism | byte-identical regeneration of all 20 committed episodes and of every report | `tests/repro/test_data_integration.py::test_generate_data_timing`, `tests/repro/test_determinism.py::test_deterministic_regeneration` |
| Data leakage | enforced by tests scanning 20 episodes x 100 surfaces (2000) and 10000 fuzzed actions; any hit fails the suite | `tests/security/test_leak_scan_full.py::test_leak_scan_20_seeds_100_steps_each`, `tests/contract/test_env_fuzz.py::test_fuzz_10000_random_actions_no_crash_no_leak` |
| Adversarial robustness | 20/20 attackers defeated on 10 dev episodes (threshold -0.3); worst mean -0.782 (undersubmit); 16/20 on the -1.000 floor; reference pairs_only -0.540 | `docs/results_attacks.json`, `docs/reward_robustness_report.md` |
| Differential signal | rule_based oracle gap duplicate_invoice: 0.19, payment_allocation: 0.79, fx_error: 0.19, period_misclassification: 0.20; spread 0.60 (payment_allocation vs duplicate_invoice); attackers in [1.33, 2.00] on every error type; 24 policies x 5 types x 3 paired episodes | `docs/oracle_gap_results.json`, `docs/oracle_gap_report.md` |
| Baselines (dev split) | oracle +0.975, rule_based +0.678, mark_everything -1.000, random -1.000 (mean reward) | `docs/results_dev.md` |
| Test coverage | 398 test functions in 53 files (21 slow); parametrisation expands the collected count (`make test` / `make test-slow`) | `tests/` |

Baseline detail (mean reward, std, min, max, mean steps):

| Policy | Mean | Std | Min | Max | Steps |
|---|---:|---:|---:|---:|---:|
| oracle | +0.975 | 0.001 | +0.974 | +0.976 | 13.3 |
| rule_based | +0.678 | 0.099 | +0.516 | +0.802 | 100.0 |
| mark_everything | -1.000 | 0.000 | -1.000 | -1.000 | 5.0 |
| random | -1.000 | 0.000 | -1.000 | -1.000 | 34.0 |

## Intended use

- Training and evaluating audit agents (RL fine-tuning, behaviour cloning, prompted LLM agents) on reconciliation errors with an exact-amount, leak-free, reproducible reward.
- Benchmarking reward robustness: the 20 attack policies and the oracle-gap diagnostic are part of the environment and rerun with one command each.

## Out of scope

- Auditing real books: the ledgers are synthetic and the error taxonomy is fixed; the environment certifies agents against this taxonomy, not against a company's accounts.
- Production deployment: the shipped image is a jury/evaluation artefact (offline, CPU-only).

## Limitations

- Synthetic data: 4 error classes plus 2 interacting scenarios; realistic rates and structure, but no document images, free text or human noise.
- No learned policy is shipped: the baselines are a rule-based auditor, a random investigator, a mark-everything attacker and a truth-reading oracle; the CPU behaviour-cloning proof (D4, `experiments/bc_proof/`) predicts the expert's actions and claims no rollout reward.
- The rule-based baseline spends its whole step budget, so late allocation issues can be missed by design (its mean on the dev split is +0.678 against the oracle's +0.975).
- Reward conventions: amounts must match exactly (the published per-type tolerance `AMOUNT_TOLERANCE_MINOR` is 0 minor units); evidence ids must exist in the public episode and cover one acceptable set; a finding with a wrong value is one false positive, while a wrong JSON type, a missing field or a size limit ends the episode at -1 and is recorded in the evaluator journal. The reward resists the shipped attackers; it is not a proof against every possible exploit.
- Isolation is in-process: the environment object holds the hidden truth in an evaluator-side store and the tools read the public episode only; the harness must run the policy in a separate process for a process boundary.

## Reproducibility

- Same seed and config give byte-identical episode files and identical rewards; all randomness flows from one seeded generator; wall-clock and global-RNG APIs are banned by lint.
- Offline by test: the complete jury protocol runs under a socket guard on every CI run; the image is built from `python:3.11-slim` pinned by digest `sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535`, runs as a non-root user and installs the pinned runtime requirements only.
- Regenerate everything: `make data`, `make baselines`, `make attacks`, `make oracle-gap`, `make ocel`, `make card`; verify: `make data-verify`, `make test-all`.

## Provenance of the numbers

| Number | Value | Source |
|---|---|---|
| attackers defeated | 20/20 | `docs/results_attacks.json: attackers[].defeated` |
| worst attacker mean reward | -0.782 (undersubmit) | `docs/results_attacks.json: max(attackers[].stats.mean_reward)` |
| attackers on the -1.000 floor | 16/20 | `docs/results_attacks.json: attackers[].stats.max_reward <= -1` |
| reference heuristic mean reward | -0.540 (pairs_only) | `docs/results_attacks.json: reference.stats.mean_reward` |
| defeat threshold | -0.3 | `docs/results_attacks.json: threshold` |
| rule_based oracle-gap spread | 0.60 (payment_allocation 0.79 vs duplicate_invoice 0.19) | `docs/oracle_gap_results.json: gap.rule_based[type] over the error types` |
| attacker oracle gap range | [1.33, 2.00] | `docs/oracle_gap_results.json: gap[attacker][error type]` |
| oracle on clean books | +1.000 | `docs/oracle_gap_results.json: mean_reward.oracle.clean` |
| oracle-gap design | 24 policies x 5 types x 3 paired episodes | `docs/oracle_gap_results.json: policies, types, n_per_type` |
| baseline oracle mean reward | +0.975 | `docs/results_dev.md: Summary table` |
| baseline rule_based mean reward | +0.678 | `docs/results_dev.md: Summary table` |
| baseline mark_everything mean reward | -1.000 | `docs/results_dev.md: Summary table` |
| baseline random mean reward | -1.000 | `docs/results_dev.md: Summary table` |
| dataset | 10 train + 10 dev episodes | `data/manifest.json: splits[].episodes` |
| skipped seeds | train [1000, 1003, 1004], dev [10009, 10010, 10011] | `data/manifest.json: splits[].skipped` |
| config hash | aa6f0ac73b16 | `data/manifest.json: config_hash` |
| leak-scan surfaces | 20 episodes x 100 = 2000 | `tests/security/test_leak_scan_full.py: EPISODES_REQUIRED, STEPS_PER_EPISODE (+ reset)` |
| fuzzed actions | 10000 | `tests/contract/test_env_fuzz.py: FUZZ_ITERATIONS` |
| test inventory | 398 test functions in 53 files (21 slow) | `tests/**/test_*.py (AST)` |
| image base digest | sha256:07d4824036af339f... | `Dockerfile: ARG PYTHON_BASE` |
| jury compute budget | 2 CPU / 4 GB RAM, no network | `CONTRIBUTING.md section 1 (declared constants JURY_CPUS, JURY_MEMORY_GB)` |
| step / time budgets | 100 actions, 60 s | `fintrace.constants: MAX_STEPS, TIME_BUDGET_SECONDS` |
