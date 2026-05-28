.PHONY: info ingest test lint format

info:
	power-forecast info

ingest:
	power-forecast ingest --config configs/data.yaml

validate:
	power-forecast validate --config configs/data.yaml

features:
	power-forecast features --config configs/data.yaml

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests
