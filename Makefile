PYTHON ?= python
IMAGE  ?= fintrace:0.1.0
BC_VENV ?= experiments/bc_proof/.venv
BC_INDEX_URL ?= https://download.pytorch.org/whl/cpu

# `data` is also a directory in the tree: it must stay .PHONY or make reports it "up to date".
.PHONY: install test test-slow test-all lint schemas data data-verify data-dry demo baselines trajectories attacks oracle-gap ocel card bc-proof clean docker-build docker-run docker-verify

install:
	$(PYTHON) -m pip install -r requirements.txt -e ".[dev]"

test:
	$(PYTHON) -m pytest -m "not slow"

test-slow:
	$(PYTHON) -m pytest -m slow

test-all:
	$(PYTHON) -m pytest -m ""

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/ judge_adapter.py

schemas:
	$(PYTHON) scripts/generate_schemas.py

data:
	$(PYTHON) scripts/generate_data.py

data-verify:
	$(PYTHON) scripts/generate_data.py --verify-only

data-dry:
	$(PYTHON) scripts/generate_data.py --dry-run

demo:
	$(PYTHON) -m fintrace

baselines:
	$(PYTHON) scripts/run_baselines.py --split dev

trajectories:
	$(PYTHON) scripts/record_trajectories.py --split dev

attacks:
	$(PYTHON) scripts/run_attacks.py --split dev

oracle-gap:
	$(PYTHON) scripts/run_oracle_gap.py

ocel:
	$(PYTHON) scripts/export_ocel.py --trajectory trajectories/rule_based_dev_10000.jsonl --out docs/sample_ocel_log.json

card:
	$(PYTHON) scripts/build_environment_card.py

# D4: CPU behaviour-cloning proof in its own venv (torch never enters .venv, the suite or the image).
# The CPU wheel index is tried first and PyPI is the fallback; behind a proxy that blocks it, run
# `make bc-proof BC_INDEX_URL=https://pypi.org/simple`. torch==2.15.0 matches 2.15.0+cpu on the
# CPU index and 2.15.0 on PyPI; pip prefers the local-version build, so the CPU wheel wins
# whenever that index is reachable. Not part of test / test-all / lint.
bc-proof:
	@test -x $(BC_VENV)/bin/python || python3.11 -m venv $(BC_VENV)
	$(BC_VENV)/bin/python -m pip install -q --retries 2 --timeout 40 -r requirements.txt -e . -r experiments/bc_proof/requirements.txt --index-url $(BC_INDEX_URL) --extra-index-url https://pypi.org/simple
	$(BC_VENV)/bin/python experiments/bc_proof/train_bc.py

clean:
	find . \( -path ./.venv -o -path ./experiments/bc_proof/.venv -o -path ./.git \) -prune -o \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name '*.egg-info' -o -name htmlcov -o -name .coverage -o -path ./build -o -path ./dist \) -prune -exec rm -rf {} +

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --network none --cpus 2 --memory 4g $(IMAGE)

# Jury-protocol smoke inside the image, offline, with the jury limits; the driver script is piped
# on stdin because the image ships no scripts/. Prints PASS/FAIL and the wall time incl. cold start.
docker-verify: docker-build
	@start=$$(date +%s); \
	if docker run --rm -i --network none --cpus 2 --memory 4g --entrypoint python $(IMAGE) - < scripts/jury_protocol_smoke.py; then \
		echo "docker-verify: PASS ($$(( $$(date +%s) - start ))s wall incl. cold start, --network none)"; \
	else \
		echo "docker-verify: FAIL"; exit 1; \
	fi
