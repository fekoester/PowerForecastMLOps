.PHONY: info ingest validate features backtest train predict forecast monitor test lint format run-all

info:
	power-forecast info

ingest:
	power-forecast ingest --config configs/data.yaml

validate:
	power-forecast validate --config configs/data.yaml

features:
	power-forecast features --config configs/data.yaml

backtest:
	power-forecast backtest --config configs/data.yaml

train:
	power-forecast train --config configs/data.yaml $(if $(MODELS),--models $(MODELS),) $(if $(WINDOWS),--windows $(WINDOWS),)

predict:
	power-forecast predict --config configs/data.yaml
	
forecast:
	power-forecast forecast --config configs/data.yaml

monitor:
	power-forecast monitor --config configs/data.yaml

run-all:
	make ingest
	make validate
	make features
	make backtest
	make train
	make predict
	make forecast
	make monitor

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests
