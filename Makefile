.PHONY: help install install-dev lint format test test-cov typecheck clean smoke-test

PYTHON ?= python

help:
	@echo "Common tasks:"
	@echo "  make install       Install the package"
	@echo "  make install-dev   Install with development dependencies"
	@echo "  make lint          Run ruff lint checks"
	@echo "  make format        Auto-format with ruff"
	@echo "  make typecheck     Run mypy"
	@echo "  make test          Run pytest"
	@echo "  make test-cov      Run pytest with coverage report"
	@echo "  make smoke-test    Run a tiny training job end-to-end"
	@echo "  make clean         Remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install || true

lint:
	ruff check src tests scripts

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts

typecheck:
	mypy src

test:
	pytest

test-cov:
	pytest --cov=diversity_tsc --cov-report=term-missing --cov-report=html

smoke-test:
	$(PYTHON) -m diversity_tsc.cli train \
		--datasets ECGFiveDays --ensemble-size 1 --epochs 2 \
		--output-directory runs/_smoke

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
