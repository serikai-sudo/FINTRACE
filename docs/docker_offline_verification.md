# Docker offline verification of the jury protocol

The jury scores FINTRACE by importing `judge_adapter` inside the Docker image, run with
`--network none --cpus 2 --memory 4g` (CONTRIBUTING.md §1). This page records how that offline
guarantee is verified, what was executed in the authoring environment and what the jury runs.

## 1. What is verified automatically on every CI run (no Docker needed)

`tests/integration/test_jury_protocol_offline.py` (fast suite, `make test`):

- installs a hard socket guard for the duration of the test: every IPv4/IPv6 `socket.socket`,
  `socket.create_connection`, `getaddrinfo`, `gethostbyname`, `gethostbyname_ex` and `getnameinfo`
  call raises and is recorded; a positive control first proves the guard blocks
  `create_connection`, `getaddrinfo`, `socket(AF_INET)`, `socket()` and `urllib.request.urlopen`;
- then runs the complete protocol through the public `judge_adapter` interface and nothing else:
  `generate_episode` → `make_env` → `reset` (closed mode) → the real rule-based baseline until the
  episode ends (≤ 100 steps) → `score_episode`;
- asserts the episode ended, every policy-facing reward was 0 (closed mode), the seven score
  fields are consistent (`precision`, `recall`, `f1`, `return = clip(2·f1 − 1 − 0.002·steps, −1, 1)`)
  and that **no network attempt was recorded** during the protocol;
- `test_runtime_requirements_lean` keeps `requirements.txt` (the only thing the image installs) and
  `[project].dependencies` free of chart, numeric and HTTP libraries;
- `test_judge_adapter_imports_no_network_or_heavy_modules` imports `judge_adapter` in a fresh
  interpreter and checks that none of `matplotlib numpy pandas requests httpx urllib3 torch pm4py
  streamlit transformers` is loaded.

The same driver the test uses, `scripts/jury_protocol_smoke.py`, is what `make docker-verify`
pipes into the container, so the in-container run and the CI test exercise one code path.

## 2. Static audit of the image (`Dockerfile`, `.dockerignore`)

| Property | Value |
|---|---|
| Base image | `python:3.11-slim` pinned by digest `sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535` |
| Build | two stages; stage 1 installs `requirements.txt` into `/opt/venv`, removes pip/setuptools and bundled test suites; stage 2 copies only the venv, `src/` and `judge_adapter.py` |
| Runtime dependencies | `annotated-types, pydantic, pydantic-core, pyyaml, typing-extensions, typing-inspection` (matplotlib and its stack are a dev extra: charts are rendered by `scripts/`, which is not in the image) |
| Build context | whitelist: `requirements.txt`, `src/`, `judge_adapter.py`; `data/` (truth files), `trajectories/`, `config/`, `docs/`, `.git` never enter the image |
| User | non-root `fintrace`, uid 10001, `nologin` shell |
| Interpreter | `PYTHONPATH=/app/src:/app`, `PYTHONDONTWRITEBYTECODE=1`, `python -m compileall` at build time so cold start pays no compile cost |
| Network | nothing in `src/` or `judge_adapter.py` imports `socket`, `urllib`, `http`, `requests` or `httpx`; the protocol test above proves the whole path runs with sockets blocked |
| Entry point | `python -m fintrace`: the live demo episode (built in memory from the embedded config, investigated by the rule-based auditor, scored; byte-stable, no files, no network), so `docker run --rm --network none fintrace:0.1.0` shows an investigation; the jury overrides it with `--entrypoint python` or imports `judge_adapter` from `/app` |
| `MPLBACKEND=Agg` | inert since matplotlib left the image; harmless |

## 3. Status in the authoring environment

The Docker CLI (29.10.0) is present but no daemon is reachable (`/var/run/docker.sock` does not
exist), so **`make docker-build` / `make docker-verify` were not executed here and no container
output is recorded on this page**. The protocol driver itself was executed locally with the same
code the image ships (`python scripts/jury_protocol_smoke.py`, see the commit message and the
test), and the offline guarantee is covered by the CI test in section 1.

## 4. Commands for the jury (or any machine with a Docker daemon)

```bash
make docker-build
```

```bash
make docker-verify
```

`docker-verify` builds the image, then runs the protocol driver inside it with the jury limits,
piping the script on stdin because the image contains no `scripts/`:

```bash
docker run --rm -i --network none --cpus 2 --memory 4g --entrypoint python fintrace:0.1.0 - \
  < scripts/jury_protocol_smoke.py
```

Expected output: one JSON line with `seed`, `episode_id`, `steps`, `terminated`, `truncated`,
`policy_rewards_zero: true`, `info_empty: true`, `n_findings` and `score` (`tp fp fn precision
recall f1 return`), a `protocol wall time` line, then `PASS` (exit 0). The Makefile target also
prints the total wall time of the `docker run` including container cold start, which must stay
far below the 60 s per-episode budget. Any failure prints `FAIL: <reason>` and exits non-zero.

To use the jury's own episode configuration instead of the embedded one, mount a config and pass
it: `docker run ... -v "$PWD/config:/cfg:ro" --entrypoint python fintrace:0.1.0 - --team-config
/cfg/dev.yaml < scripts/jury_protocol_smoke.py` (the image includes `pyyaml`).

To record a transcript, run `make docker-verify 2>&1 | tee docs/docker_offline_transcript.txt`
and note `docker image inspect fintrace:0.1.0 --format '{{.Id}}'` next to it.
