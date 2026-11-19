PYTHON ?= python
IMAGE  ?= fintrace:0.1.0

.PHONY: install test lint demo baselines attacks oracle-gap clean docker-build docker-run

install:
	$(PYTHON) -m pip install -r requirements.txt -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src/ tests/

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
