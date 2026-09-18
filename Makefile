VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
PYTEST ?= $(VENV)/bin/pytest
RUFF ?= $(VENV)/bin/ruff
MYPY ?= $(VENV)/bin/mypy

.PHONY: help setup test lint format typecheck gen-kb validate bench-replay bench report docker

help:
	@echo "Available commands:"
	@echo "  make setup         Install dependencies and package in editable mode"
	@echo "  make test          Run pytest suite"
	@echo "  make lint          Run ruff linter and format check"
	@echo "  make format        Run ruff formatter and fix lint errors"
	@echo "  make typecheck     Run mypy type checker on src/"
	@echo "  make gen-kb        Generate base synthetic knowledge base (seed 42)"
	@echo "  make validate      Validate suites and distractor hardness"
	@echo "  make bench-replay  Run offline replay benchmark"
	@echo "  make bench         Run live/scripted benchmark on smoke suite"
	@echo "  make report        Generate report for runs/latest"
	@echo "  make docker        Build non-root Docker container"

setup:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) tests/

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

typecheck:
	$(MYPY) src

gen-kb:
	$(PYTHON) -m ragstress.cli gen-kb --seed 42 --out data/kb

validate:
	$(PYTHON) -m ragstress.cli validate-suite data/suites/smoke.jsonl
	$(PYTHON) -m ragstress.cli hardness data/suites/smoke.jsonl

bench-replay:
	$(PYTHON) -m ragstress.cli run --generator replay:gpt-4o-mini --suite data/suites/smoke.jsonl --run-id bench_replay --out runs/bench_replay

bench:
	$(PYTHON) -m ragstress.cli run --generator scripted:oracle --suite data/suites/smoke.jsonl --out runs/latest

report:
	$(PYTHON) -m ragstress.cli report runs/bench_replay

docker:
	docker build -t ragstress:latest .
