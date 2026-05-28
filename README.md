# PowerForecastMLOps

Live electricity-demand forecasting platform using public EIA demand data and Open-Meteo weather data.

This project demonstrates an end-to-end production-style machine learning workflow: automated data ingestion, validation, leakage-safe feature engineering, walk-forward backtesting, model training, batch prediction, monitoring, CI/CD, and GitHub Pages deployment.

## Live dashboard

**Dashboard:** https://fekoester.github.io/PowerForecastMLOps/

The dashboard is generated automatically by the pipeline and shows the latest prediction performance, model health status, baseline comparison, feature importance, and monitoring diagnostics.

## What this project demonstrates

This is not a notebook-only forecasting experiment. The goal is to build the core structure of an operational ML forecasting system.

The pipeline includes:

* automated ingestion from public APIs
* raw data manifests
* schema, timestamp, missingness, and value-range validation
* leakage-safe lag and rolling-window feature engineering
* walk-forward time-series backtesting
* strong naive baseline comparison
* LightGBM model training
* model artifact serialization with feature schema metadata
* batch prediction pipeline
* monitoring and degradation checks
* machine-readable JSON reports
* human-readable HTML dashboard
* GitHub Actions CI/CD
* scheduled daily runs
* GitHub Pages deployment

## Current result

On the current dataset, the LightGBM model improves substantially over the strongest naive baseline.

Typical recent result:

| Model / baseline                          |      MAE | RMSE |      MAPE |
| ----------------------------------------- | -------: | ---: | --------: |
| Best naive baseline: previous hour demand |     ~787 | ~959 |    ~3.25% |
| Walk-forward LightGBM average             | ~420–430 | ~580 | ~1.7–1.8% |

The exact numbers change slightly as the live API data updates.

## Data sources

The project uses:

* **EIA Open Data API** for hourly electricity demand
* **Open-Meteo API** for hourly weather data

The current configuration uses California ISO demand data and Los Angeles weather as a simple weather proxy.

## Pipeline

```text
EIA + Open-Meteo APIs
        ↓
Raw data ingestion
        ↓
Data validation
        ↓
Leakage-safe feature engineering
        ↓
Walk-forward baseline backtesting
        ↓
LightGBM model training
        ↓
Batch prediction
        ↓
Monitoring report
        ↓
GitHub Actions + GitHub Pages
```

## Main commands

Run the full local pipeline:

```bash
make run-all
```

Run tests:

```bash
make test
```

Run individual steps:

```bash
make ingest
make validate
make features
make backtest
make train
make predict
make monitor
```

## Project structure

```text
configs/
  data.yaml

src/power_forecast/
  data/
    eia.py
    weather.py
    validate.py
    manifest.py

  features/
    build_features.py

  models/
    backtest.py
    metrics.py
    train_lightgbm.py
    predict.py

  monitoring/
    report.py

  pipelines/
    ingest_pipeline.py
    validation_pipeline.py
    feature_pipeline.py
    backtest_pipeline.py
    train_pipeline.py
    predict_pipeline.py
    monitor_pipeline.py

tests/
  test_validation.py
  test_features.py
  test_backtest.py
  test_train_lightgbm.py
  test_predict.py
  test_monitoring.py
```

## MLOps concepts covered

### Data validation

Before feature engineering or training, the pipeline validates:

* required columns
* timestamp parsing
* duplicate timestamps
* hourly coverage
* missing-value rates
* physically plausible value ranges

If critical checks fail, the pipeline stops.

### Leakage-safe forecasting features

Lag and rolling-window features are shifted so the model only sees information that would have been available at prediction time.

Example:

```python
shifted = demand_mwh.shift(1)
rolling_mean_24h = shifted.rolling(24).mean()
```

This prevents the current target from leaking into the input features.

### Walk-forward validation

Forecasting performance is evaluated with time-respecting walk-forward validation rather than random train/test splits.

This simulates the operational setting:

```text
train on past data → validate on future data
```

### Baseline comparison

The model is compared against strong naive baselines, including:

* previous hour demand
* same hour yesterday
* same hour last week
* rolling 24-hour mean
* rolling 7-day mean

A model is only useful if it beats the relevant operational baseline.

### Model artifact

The trained model artifact contains:

* fitted LightGBM model
* feature column order
* target column
* timestamp column
* training timestamp
* model configuration
* backtest metrics
* baseline comparison

This makes inference safer and more reproducible.

### Monitoring

The monitoring report compares the latest realized prediction error against:

* the model’s walk-forward validation performance
* the strongest naive baseline

It assigns a health status:

* `healthy`
* `watch`
* `degraded`
* `unknown`

## CI/CD

GitHub Actions runs the pipeline on push, pull request, manual trigger, and daily schedule.

The workflow:

1. installs the project in a clean Linux environment
2. runs the test suite
3. executes the full forecasting pipeline
4. uploads reports and predictions as artifacts
5. deploys the latest dashboard to GitHub Pages

## Current limitations

This project currently predicts the latest known feature rows for monitoring and demonstration purposes. The next planned extension is true next-24-hour forecasting using Open-Meteo forecast data and recursive lag construction.

Other planned extensions:

* proper future forecast mode
* model registry / promotion rules
* MLflow experiment tracking
* richer drift monitoring
* multiple weather locations
* multiple balancing authorities
* optional Docker deployment
* more polished dashboard design

## Why this project exists

The purpose of this project is to demonstrate practical ML platform and forecasting engineering beyond notebook-based modeling. It focuses on reproducibility, validation, backtesting discipline, monitoring, and automated deployment — the pieces needed to turn a forecasting model into an operational ML workflow.
