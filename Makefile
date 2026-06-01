.PHONY: info ingest validate features backtest train predict forecast monitor test lint format run-all

CONFIG ?= configs/data.yaml

info:
	power-forecast info

ingest:
	power-forecast ingest --config $(CONFIG)

validate:
	power-forecast validate --config $(CONFIG)

features:
	power-forecast features --config $(CONFIG)

backtest:
	power-forecast backtest --config $(CONFIG)

train:
	power-forecast train --config $(CONFIG) $(if $(MODELS),--models $(MODELS),) $(if $(WINDOWS),--windows $(WINDOWS),)

predict:
	power-forecast predict --config $(CONFIG)
	
forecast:
	power-forecast forecast --config $(CONFIG)

monitor:
	power-forecast monitor --config $(CONFIG)

run-all:
	make ingest
	make validate
	make features
	make backtest
	make train
	make predict
	make forecast
	make monitor

run-ci:
	$(MAKE) ingest CONFIG=configs/data_ci.yaml
	$(MAKE) validate CONFIG=configs/data_ci.yaml
	$(MAKE) features CONFIG=configs/data_ci.yaml
	$(MAKE) backtest CONFIG=configs/data_ci.yaml
	$(MAKE) train CONFIG=configs/data_ci.yaml MODELS=lightgbm,mlp,esn WINDOWS=30,90
	$(MAKE) predict CONFIG=configs/data_ci.yaml
	$(MAKE) forecast CONFIG=configs/data_ci.yaml
	$(MAKE) monitor CONFIG=configs/data_ci.yaml

test:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests
