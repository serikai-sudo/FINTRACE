PYTHON ?= python
IMAGE  ?= fintrace:0.1.0

# `data` is also a directory in the tree: it must stay .PHONY or make reports it "up to date".
.PHONY: install test test-slow test-all lint schemas data data-verify data-dry demo baselines attacks oracle-gap clean docker-build docker-run

install:
	$(PYTHON) -m pip install -r requirements.txt -e ".[dev]"

test:
	$(PYTHON) -m pytest -m "not slow"

test-slow:
	$(PYTHON) -m pytest -m slow

test-all:
	$(PYTHON) -m pytest -m ""

lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

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
	@echo "TODO"

attacks:
	@echo "TODO"

oracle-gap:
	@echo "TODO"

clean:
	find . \( -path ./.venv -o -path ./.git \) -prune -o \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name '*.egg-info' -o -name htmlcov -o -name .coverage -o -path ./build -o -path ./dist \) -prune -exec rm -rf {} +

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm --network none --cpus 2 --memory 4g $(IMAGE)
