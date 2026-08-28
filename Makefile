.PHONY: test lint pipeline api

test:
	python -m pytest -q

lint:
	python -m ruff check src tests

pipeline:
	python -m mlops_platform.cli run-all

api:
	uvicorn mlops_platform.api:app --reload
