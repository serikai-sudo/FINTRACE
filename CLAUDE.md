# CLAUDE.md — FINTRACE

Instructions for every human and AI agent working in this repository. Read fully before any change.
If a rule here conflicts with a convenience, the rule wins.

## 1. Project context

FINTRACE is a reinforcement-learning environment for fine-tuning and evaluating financial-audit
agents, built for **Genesis Hackathon 2026 — FinTech (online)**; deadline **Nov 22, 12:00 Astana**,
500+ competing teams. Each episode is a seeded synthetic ledger plus source documents (invoices,
payments, FX rates) with injected errors — duplicate invoices, partial payments, FX errors, period
misclassification — hidden among clean decoys. The agent investigates through a fixed tool API and
submits findings; an isolated evaluator scores them against hidden ground truth.

### Compute budget on the jury machine (hard limits)

| Resource            | Limit                                  |
|---------------------|----------------------------------------|
| CPU / RAM           | 2 CPU, 4 GB RAM                        |
| Network             | none (offline)                         |
| Wall clock          | ≤ 60 s per episode                     |
| Steps               | ≤ 100 steps per episode                |
| Repository size     | ≤ 1 GB                                 |
| GPU / torch         | not available, not shipped             |

`torch` is allowed **only** in `experiments/bc_proof/` (own `requirements.txt`). It is never
installed in the Docker image and never imported from `src/` (ruff `banned-api` enforces this).

## 2. Repository map

```
config/            YAML episode configs (default/train/dev) + versions.json (pinned versions)
schemas/           JSON Schema 2020-12. manifest/invoice/payment/submission are GENERATED from
                   models.py by `make schemas`; trajectory/ocel are hand-written (tests pin them
                   to constants.py)
data/{train,dev}/  10 + 10 committed episodes, ep_<episode_id>/ each (`make data` regenerates them
                   from config/{train,dev}.yaml; data/manifest.json lists seeds and skipped seeds)
src/fintrace/
  constants.py     project-wide Final constants
  models.py        Pydantic v2 models (PublicEpisode / PrivateEpisode, records, submissions)
  generator/       seeded episode generator; injectors/ = one module per error type
  environment/     step/reset loop, observation building, step & time budgets
  tools/           agent-facing tools; operate on PublicEpisode ONLY
  graph/           NetworkX document/ledger graph
  exporters/       JSONL / OCEL exporters; episode_files.py = episode directory read/write
                   (the single serializer behind `make data` and the jury adapter)
  evaluator/       scoring against PrivateEpisode (truth + provenance)
  reward/          reward shaping on top of evaluator output
  agents/          baseline agents; attacks/ = adversarial reward-hacking agents
  diagnostics/     oracle-gap and robustness diagnostics
  logging/         structured trajectory logging
  transforms/      observation / data transforms
experiments/bc_proof/  behaviour-cloning proof (torch; NOT part of the package or image)
tests/             contract/, reward/, repro/, robustness/, integration/, test_smoke.py
scripts/           CLI tooling: generate_schemas.py (`make schemas`)
judge_adapter.py   jury connection protocol (generate_episode / make_env / score_episode);
                   the only Python file at the repository root, copied to /app in the image
trajectories/ docs/ demo/
```

Do not add top-level directories or new modules without explicit approval. The tree above is the
architecture; extend existing files first.

## 3. Commands

| Command             | What it does                                                    |
|---------------------|-----------------------------------------------------------------|
| `make install`      | `pip install -r requirements.txt -e ".[dev]"` (pinned runtime + dev) |
| `make test`         | `pytest -m "not slow"` — the fast suite (< 40 s; config in `pyproject.toml`; warnings are errors) |
| `make test-slow`    | `pytest -m slow` — seed sweeps, fuzz, full leak scan (≈ 2 min)  |
| `make test-all`     | `pytest -m ""` — both suites                                    |
| `make lint`         | `ruff check src/ tests/ scripts/ judge_adapter.py` — must be zero findings |
| `make schemas`      | regenerate `schemas/{manifest,invoice,payment,submission}.schema.json` from `models.py`; run after every model change, under the pinned deps from `make install` (drift test fails otherwise) |
| `make data`         | regenerate `data/{train,dev}/ep_*/` + `data/manifest.json` from `config/{train,dev}.yaml` (skips infeasible seeds; commit the result separately) |
| `make data-verify`  | re-hash every episode's public files against `checksums.sha256` and the config hash; non-zero on drift |
| `make data-dry`     | build both splits in memory and print the report JSON without writing |
| `make demo`         | end-to-end demo episode (TODO)                                  |
| `make baselines`    | run the 4 baselines (random, rule_based, mark_everything, oracle) on the dev split and write `docs/results_dev.md` |
| `make trajectories` | record the 4 baselines × dev split as `trajectories/<policy>_dev_<seed>.jsonl` (one canonical JSON line per step, thoughts kept) and leak-scan every file; only `rule_based_dev_10000.jsonl` is committed as the demo sample |
| `make attacks`      | run the 20 reward-hack attackers on the dev split and write `docs/reward_robustness_report.md`, `docs/results_attacks.json`, `docs/figures/robustness_chart.png`; non-zero if any attacker beats −0.3 |
| `make oracle-gap`   | build 5 x 3 paired single-error diagnostic episodes in memory (clean + one type each, seeds 30000+), run the 24 policies (4 baselines + 20 attackers) and write `docs/oracle_gap_report.md`, `docs/oracle_gap_results.json`, `docs/figures/oracle_gap_matrix.png`; non-zero if the oracle drops below +0.9 or an attacker earns a positive reward on erroneous books |
| `make ocel`         | export the committed sample trajectory + its public dev episode as an OCEL 2.0 log (`docs/sample_ocel_log.json`; objects = episode/invoices/payments/fx_rates/findings, events = tool calls); validates structure + `schemas/ocel.schema.json` and leak-scans before writing; non-zero on any failure |
| `make clean`        | remove caches, build artifacts, `*.egg-info` (keeps `.venv`)    |
| `make docker-build` | `docker build -t fintrace:0.1.0 .`                              |
| `make docker-run`   | run the image offline with jury limits (`--network none --cpus 2 --memory 4g`) |

Python interpreter: **3.11 only** (`requires-python = ">=3.11,<3.12"`). Create the venv with
`python3.11 -m venv .venv`.

## 4. Style rules

- **Python 3.11.** Use 3.11 syntax (`X | None`, `typing.Self`, `StrEnum`); nothing newer.
- **Type hints everywhere**: every function argument, return value and module-level constant.
  Function signatures are enforced by ruff `ANN`; variable/constant annotations
  (`MAX_STEPS: Final[int] = 100`) are checked in review — ruff does not lint them.
- **Pydantic v2** for every data model, always with `model_config = ConfigDict(extra="forbid")`.
  Prefer `frozen=True` for records that must not mutate after generation.
- **Money is `decimal.Decimal`** — never `float`. Quantize explicitly; parse from `str`, never from
  `float` (`Decimal("10.10")`, not `Decimal(10.1)`).
- **Reuse the contract field types** from `fintrace.models` for these roles: identifiers →
  `OpaqueId`, money in minor units → `MinorUnits`, sizes/step counters → `NonNegInt` / `StepCount`,
  free-form JSON payloads → `JsonObject`. A bare `str` / `int` / `dict` is wrong for those roles
  (bounded config values such as `qty` or `minor_unit_digits` stay plain `int` with `Field` limits).
- **Schemas follow models**: after any change to `models.py` run `make schemas` and commit the
  regenerated files. `tests/contract/test_schemas.py` compares them byte-for-byte and fails on drift.
- **Timestamps are ISO-8601 with an explicit offset** (`2026-03-31T23:59:59+05:00`). Naive
  datetimes are bugs (ruff `DTZ`).
- **Named exports**: every module declares `__all__`; import by name
  (`from fintrace.models import PublicEpisode`), no star imports, no default-export patterns.
- **File names in `snake_case`**; one responsibility per module.
- **No `print()` in `src/`** (ruff `T20`). Structured logging arrives with `fintrace.logging`.
- **Deterministic serialization**: `json.dumps(..., sort_keys=True)`; stable ordering of all
  emitted collections.

## 5. Anti-patterns — DO NOT

| Don't                                                                 | Why / instead                                              | Enforced by |
|-----------------------------------------------------------------------|------------------------------------------------------------|-------------|
| `float` for money                                                     | rounding drift breaks ground truth → `Decimal`             | review, tests |
| global `random.*` (`random.random()`, `random.seed()`, …), `random.SystemRandom`, `os.urandom`, `secrets`, `numpy.random` | hidden state / OS entropy breaks repro → one `rng = random.Random(seed)` passed explicitly | ruff `TID251` (names listed in `pyproject.toml`; anything not listed → review) |
| `time.time()`/`time_ns()`, `datetime.now()`/`utcnow()`/`today()`, `uuid4()`/`uuid1()` | non-deterministic → derive IDs/times from the seeded RNG; measure budgets with `time.monotonic()` | ruff `TID251` |
| iterating a `set` (or `frozenset`) over episode content               | order varies with hash seed → `sorted(...)` first          | review, repro tests |
| `eval()` / `exec()` in the safe calculator                            | code execution → AST-whitelisted evaluator                 | ruff `S307` / `S102` |
| creating a new file when an existing one can be edited                | tree sprawl → extend the existing module                   | review |
| hand-editing `schemas/{manifest,invoice,payment,submission}.schema.json` | they are generated → edit `models.py`, run `make schemas` | schema drift test |
| importing `fintrace.evaluator` from `fintrace.agents.*`               | agents would see the scorer → keep isolation               | review (import-graph test planned) |
| echoing the seed in `observation.info`                                | lets agents regenerate truth → never expose the seed       | leak-scan test |
| stack traces / exception text in environment error responses          | leaks internals → only `{"error": {"code": "<CODE>"}}`     | contract tests |
| `torch` (or any GPU/network dependency) in `src/` or the image        | jury box is CPU-only and offline                           | ruff `TID251` |

## 6. Architecture invariants

1. **PublicEpisode vs PrivateEpisode.** The generator produces both. `PublicEpisode` is the only
   object the environment and `tools/` may read. `PrivateEpisode` (ground truth, injection
   provenance) is consumed exclusively by `evaluator/`.
2. **`truth.json` and `provenance.json` never reach the agent surface** — not in observations,
   tool outputs, `info`, error messages, logs shown to the agent, or file paths the agent can open.
3. **Evaluator isolation.** Today `evaluator/` is a separate module; later it becomes a separate
   process. `agents/*` (including `agents/attacks/*`) must never import `evaluator/*`, directly or
   transitively.
4. **Error contract.** Every environment error is exactly `{"error": {"code": "<CODE>"}}`. No
   messages, no stack traces, no echoed input.
5. **Determinism.** Same seed + same config ⇒ byte-identical episode files and identical rewards.
   All randomness flows from one `random.Random(seed)` instance.
6. **Budgets.** Environment enforces ≤ 100 steps and ≤ 60 s per episode; exceeding either
   terminates the episode with an error code, not an exception. The 100-step budget is the
   deterministic bound; the ≤ 60 s wall-clock is a safety timeout (`time.monotonic`, stored off-state)
   that fires only on a hung policy and yields the empty-submission truncation reward, so §6.5
   determinism holds for all well-behaved runs.
7. **Versioning.** `config/versions.json` and `schema_version` in configs are the source of truth
   for artifact versions; `fintrace.__version__` is the package version.

## 7. Judging criteria (100 points) — optimize for these

| Criterion                         | Points | What it means for code here                                  |
|-----------------------------------|-------:|--------------------------------------------------------------|
| Contract / isolation              | 25     | Public/Private split, no leaks, strict error contract, schemas |
| Data / ground truth               | 25     | Realistic generator, exact injected-error truth, decoys       |
| Reward reliability                | 25     | Deterministic reward, resistant to attack agents              |
| Reproducibility / generalization  | 15     | Seeded repro, train/dev split, held-out generalization        |
| Engineering                       | 10     | CI green, Docker offline, typed, tested, documented           |

## 8. Testing

- **Every feature ships with a test** in the matching folder: `contract/` (schemas, API, error
  contract, fuzz), `reward/` (reward correctness), `repro/` (seed determinism), `robustness/`
  (attack agents), `security/` (isolation, leak scans), `integration/` (end-to-end episode).
- Run `make test` (and `make lint`) **after every change**. Red tests or lint findings block commit.
  Tests slower than ~10 s (seed sweeps, fuzz, full leak scans) carry `@pytest.mark.slow`, are
  excluded from `make test` and run with `make test-slow`; CI runs both jobs.
- **Leak-scan test gates all observations**: every observation, tool output and error payload is
  scanned for ground-truth values, provenance fields and the seed; any hit fails the suite.
- Pytest runs with `filterwarnings = error`, `--strict-markers`, `xfail_strict`.

## 9. Agent workflow

1. Plan first (list of files to touch + one line each), get approval, then implement.
2. Edit existing files; do not create files outside the tree in §2 without approval.
3. Keep diffs minimal and scoped to the task; no drive-by refactors.
4. Before finishing: `make lint && make test` green; report outputs verbatim.
5. Never commit trajectories (`trajectories/*.jsonl`) or secrets. The dataset under `data/` is
   committed, but only as the output of `make data` and in its own `data:` commit — never hand-edit
   an episode file (`make data-verify` and the integration tests catch drift).
