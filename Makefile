PYTHON ?= python
IMAGE  ?= fintrace:0.1.0

# `data` is also a directory in the tree: it must stay .PHONY or make reports it "up to date".
.PHONY: install test test-slow test-all lint schemas data data-verify data-dry demo baselines trajectories attacks oracle-gap ocel clean docker-build docker-run

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
	@echo "TODO"

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

clean:
	find . \( -path ./.venv -o -path ./.git \) -prune -o \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name '*.egg-info' -o -name htmlcov -o -name .coverage -o -path ./build -o -path ./dist \) -prune -exec rm -rf {} +

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --network none --cpus 2 --memory 4g $(IMAGE)
