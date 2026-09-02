VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
PYTEST ?= $(VENV)/bin/pytest
RUFF ?= $(VENV)/bin/ruff
MYPY ?= $(VENV)/bin/mypy

.PHONY: help setup test lint format typecheck gen-kb validate bench-replay bench report

help:
	@echo "Available commands:"
	@echo "  make setup         Install dependencies and package in editable mode"
	@echo "  make test          Run pytest suite"
	@echo "  make lint          Run ruff linter and format check"
	@echo "  make format        Run ruff formatter and fix lint errors"
	@echo "  make typecheck     Run mypy type checker on src/"
	@echo "  make gen-kb        Generate base synthetic knowledge base (seed 42)"

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
	@echo "validate target placeholder (Phase 6)"

bench-replay:
	@echo "bench-replay target placeholder (Phase 6)"

bench:
	@echo "bench target placeholder (Phase 6)"

report:
	@echo "report target placeholder (Phase 7)"
