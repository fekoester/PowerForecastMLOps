# PowerForecastMLOps

Live electricity-demand forecasting platform using public EIA demand data and Open-Meteo weather data.

This project demonstrates an end-to-end production-style machine learning workflow for time-series forecasting: automated data ingestion, validation, forecast-safe feature engineering, walk-forward backtesting, model comparison, batch prediction, true next-24h forecasting, monitoring, CI/CD, and GitHub Pages deployment.

## Live dashboard

**Dashboard:** https://fekoester.github.io/PowerForecastMLOps/

The dashboard is generated automatically by the pipeline and shows:

* latest 24h realized prediction performance
* next 24h future forecast
* model health status
* production model selection
* latest-window winner
* model comparison across different training windows
* baseline comparison
* feature importance
* monitoring diagnostics

## What this project demonstrates

This is not a notebook-only forecasting experiment. The goal is to build the core structure of an operational ML forecasting system.

The pipeline includes:

* automated ingestion from public APIs
* paginated EIA demand ingestion for multi-year hourly data
* raw data manifests
* schema, timestamp, missingness, and value-range validation
* forecast-safe lag and historical-demand feature engineering
* walk-forward time-series backtesting
* fair day-ahead baseline comparison
* model comparison across LightGBM, MLP, and Echo State Network models
* multiple recent-training-window variants from the same 3-year feature table
* recency-weighted LightGBM training
* model artifact serialization with feature schema metadata
* latest 24h batch prediction with realized actuals
* true next 24h future forecasting using weather forecast data
* monitoring and degradation checks
* machine-readable JSON reports
* human-readable HTML dashboard
* GitHub Actions CI/CD
* scheduled daily runs
* GitHub Pages deployment

## Current result

The current setup uses roughly three years of hourly California ISO demand data and Los Angeles weather data.

A typical recent result:

| Model / baseline                      | Walk-forward MAE | Walk-forward RMSE | Walk-forward MAPE |
| ------------------------------------- | ---------------: | ----------------: | ----------------: |
| Best fair baseline: lag_24h           |           ~1,284 |            ~1,792 |             ~4.9% |
| Best production model: LightGBM 1095d |             ~866 |            ~1,194 |             ~3.3% |

The exact numbers change as the live API data updates.

The monitoring report also evaluates the latest known 24h window separately. This is intentionally shown separately from walk-forward model selection, because current-regime performance and historical-average performance can disagree.

Example latest-24h ranking from one run:

| Model          | Latest 24h MAE | Latest 24h MAPE | Role              |
| -------------- | -------------: | --------------: | ----------------- |
| lightgbm_30d   |            ~52 |          ~0.21% | latest 24h winner |
| lightgbm_90d   |           ~110 |          ~0.45% |                   |
| lightgbm_180d  |           ~158 |          ~0.64% |                   |
| lightgbm_1095d |           ~235 |          ~0.94% | production model  |

This exposes an important forecasting tradeoff:

* long training windows are usually more robust on average
* short training windows may adapt better to the current regime
* operational monitoring should show both

## Data sources

The project uses:

* **EIA Open Data API** for hourly electricity demand
* **Open-Meteo Historical Weather API** for hourly historical weather
* **Open-Meteo Forecast API** for next-24h weather forecast inputs

The current configuration uses:

* California ISO demand data
* Los Angeles weather as a simple weather proxy

This is intentionally simple and reproducible. A more production-grade regional model would use multiple weather locations and more detailed grid/market features.

## Forecasting task

The project now separates two different tasks:

### 1. Latest 24h monitoring prediction

This predicts the latest 24 known hourly demand values.

Actual demand is already known, so the system can compute:

* MAE
* RMSE
* MAPE
* Bias

This answers:

> How well did the model perform on the latest realized day?

### 2. Next 24h future forecast

This predicts the next 24 future hourly demand values.

Actual demand is not known yet, so no realized error metrics are shown.

This answers:

> What is the model forecasting for the next 24 hours?

## Forecast-safe features

The model is designed for direct next-24h forecasting. Therefore, it avoids features that would require unknown future demand.

Unsafe for direct next-24h forecasting:

* demand_lag_1h
* demand_lag_2h
* demand_lag_3h
* rolling means immediately before each future timestamp, unless recursively constructed

Safe features include:

* weather forecast at target hour
* calendar features at target hour
* demand_lag_24h
* demand_lag_48h
* demand_lag_168h
* same-hour historical demand statistics
* origin-shifted rolling statistics

Example safe demand-history features:

```text
demand_lag_24h
demand_lag_48h
demand_lag_168h

demand_origin_roll_mean_24h
demand_origin_roll_std_24h
demand_origin_roll_min_24h
demand_origin_roll_max_24h

demand_same_hour_mean_7d
demand_same_hour_mean_14d
demand_same_hour_mean_30d
demand_same_hour_mean_60d
demand_same_hour_mean_90d
demand_same_hour_mean_180d
demand_same_hour_mean_365d
```

For a target timestamp `t`, same-hour features use historical demand from earlier days:

```text
demand(t - 24h)
demand(t - 48h)
demand(t - 72h)
...
```

This makes them valid for future forecasting.

## Model candidates

The pipeline trains and compares several model families:

* LightGBM
* MLP neural network
* Echo State Network / reservoir computer

Each model can be trained on several recent training windows from the same multi-year feature table:

```text
30 days
90 days
180 days
365 days
1095 days
```

Example candidate names:

```text
lightgbm_30d
lightgbm_90d
lightgbm_180d
lightgbm_365d
lightgbm_1095d

mlp_30d
mlp_90d
...

esn_30d
esn_90d
...
```

This allows the pipeline to explicitly compare current-regime models against longer-history models.

## Production model vs latest-window winner

The dashboard distinguishes between:

### Production model

The model selected by average walk-forward validation MAE.

This is the historically robust model.

### Latest 24h winner

The model with the lowest realized MAE on the latest known 24h monitoring window.

This is the model that best fit the most recent day.

These are intentionally not always the same model.

Example:

```text
Production model:      lightgbm_1095d
Latest 24h winner:     lightgbm_30d
```

This distinction is important because selecting only by the latest day can overfit to one unusually easy or unusual window, while selecting only by long-run validation can under-adapt to the current regime.

## Recency weighting

The LightGBM model can use exponential recency weighting.

Example:

```text
weight = 0.5 ** (age_days / half_life_days)
```

With a half-life of 180 days:

* current rows have weight near 1.0
* rows 180 days old have weight around 0.5
* rows 360 days old have weight around 0.25
* older rows can be clipped to a minimum weight

This keeps the seasonal information from multi-year data while reducing the influence of stale regimes.

## Pipeline

```text
EIA + Open-Meteo APIs
        ↓
Raw data ingestion
        ↓
Data validation
        ↓
Forecast-safe feature engineering
        ↓
Fair baseline backtesting
        ↓
Walk-forward model comparison
        ↓
Train final model candidates
        ↓
Latest 24h batch prediction
        ↓
Next 24h future forecast
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
make forecast
make monitor
```

Useful inspection commands:

```bash
cat reports/metrics/train_lightgbm.json
cat reports/metrics/prediction_summary.json
cat reports/metrics/future_forecast_summary.json
cat reports/metrics/monitoring_summary.json
```

Open the local dashboard:

```bash
xdg-open reports/monitoring/latest_report.html
```

## Project structure

```text
configs/
  data.yaml

src/power_forecast/
  data/
    eia.py
    weather.py
    weather_forecast.py
    validate.py
    manifest.py

  features/
    build_features.py
    future_features.py

  models/
    backtest.py
    esn.py
    forecast.py
    metrics.py
    model_zoo.py
    predict.py
    train_compare.py

  monitoring/
    report.py

  pipelines/
    ingest_pipeline.py
    validation_pipeline.py
    feature_pipeline.py
    backtest_pipeline.py
    train_pipeline.py
    predict_pipeline.py
    forecast_pipeline.py
    monitor_pipeline.py

tests/
  test_validation.py
  test_features.py
  test_backtest.py
  test_train_lightgbm.py
  test_model_zoo.py
  test_predict.py
  test_monitoring.py
```

## MLOps concepts covered

### Data ingestion

The ingestion step pulls hourly demand and weather data from public APIs.

The EIA demand fetcher supports pagination, which allows multi-year hourly demand ingestion instead of being limited to the first 5,000 rows.

### Data validation

Before feature engineering or training, the pipeline validates:

* required columns
* timestamp parsing
* duplicate timestamps
* hourly coverage
* missing-value rates
* physically plausible value ranges

If critical checks fail, the pipeline stops.

### Feature engineering

Feature engineering creates weather, calendar, lag, same-hour, and origin-shifted rolling features.

The forecasting features are designed to be safe for next-24h prediction. The model does not use future demand values.

### Walk-forward validation

Forecasting performance is evaluated with time-respecting walk-forward validation rather than random train/test splits.

This simulates the operational setting:

```text
train on past data → validate on future data
```

### Baseline comparison

The model is compared against fair day-ahead baselines, including:

* same hour yesterday
* same hour two days ago
* same hour last week
* same-hour historical averages

The strongest baseline is reported automatically.

A model is only useful if it beats the relevant operational baseline.

### Model comparison

The training step compares:

* LightGBM
* MLP
* Echo State Network

and several recent training-window variants of each.

This turns the non-stationarity problem into an explicit experiment:

```text
Do we want robust long-history behavior or current-regime adaptation?
```

### Model artifact

The trained model artifact contains:

* selected production model
* all trained candidate models
* feature column order
* target column
* timestamp column
* training timestamp
* model configuration
* candidate training windows
* walk-forward metrics
* baseline comparison
* feature schema metadata

This makes inference safer and more reproducible.

### Prediction

The prediction step scores the latest known 24h window for all model candidates.

Because actual demand is known for this window, the pipeline computes realized per-model metrics.

### Future forecast

The forecast step scores the next 24 future hours.

It uses:

* Open-Meteo weather forecast data
* calendar features
* demand-history features available before the forecast horizon

Actual future demand is not known yet, so the dashboard shows only the forecast curves without error metrics.

### Monitoring

The monitoring report compares the latest realized prediction error against:

* the selected model’s walk-forward validation performance
* the strongest fair baseline
* all other candidate models on the latest 24h window

It assigns a health status:

* `healthy`
* `watch`
* `degraded`
* `unknown`

The local monitor output also prints a ranked latest-24h table for all candidate models.

Example:

```text
Production model:  lightgbm_1095d
Latest 24h winner: lightgbm_30d
```

## CI/CD

GitHub Actions runs the pipeline on:

* push
* pull request
* manual trigger
* daily schedule

The workflow:

1. installs the project in a clean Linux environment
2. runs the test suite
3. executes the full forecasting pipeline
4. uploads reports and predictions as artifacts
5. deploys the latest dashboard to GitHub Pages

## GitHub secrets

The pipeline requires an EIA API key.

Add this repository secret:

```text
EIA_API_KEY
```

The Open-Meteo API does not require an API key for the current usage.

## Configuration

The main configuration lives in:

```text
configs/data.yaml
```

Important settings include:

```yaml
eia:
  lookback_days: 1095

weather:
  lookback_days: 1095

features:
  same_hour_windows_days:
    - 7
    - 14
    - 30
    - 60
    - 90
    - 180
    - 365

train:
  training_windows_days:
    - 30
    - 90
    - 180
    - 365
    - 1095

  recency_weighting:
    enabled: true
    half_life_days: 180
    min_weight: 0.10
    apply_to:
      - lightgbm
```

## Current limitations

This project is intentionally compact and runs on GitHub Actions. Current limitations include:

* single demand region
* single weather proxy location
* no holiday/calendar-event feature yet
* no probabilistic prediction intervals yet
* no formal model registry
* no automatic model promotion rule beyond current selection logic
* no MLflow or experiment database
* dashboard is static HTML generated by the pipeline
* weather proxy is simplistic for a large balancing authority

## Planned extensions

Potential next improvements:

* adaptive production selection rule using both walk-forward and latest-window metrics
* ensemble of strong recent-window and long-window models
* prediction intervals / quantile regression
* multiple weather locations
* holiday and special-day features
* weather forecast error monitoring
* drift metrics for feature distributions
* explicit model registry and promotion rules
* MLflow experiment tracking
* Docker deployment
* support for multiple balancing authorities
* richer dashboard filtering and model comparison controls

## Why this project exists

The purpose of this project is to demonstrate practical ML platform and forecasting engineering beyond notebook-based modeling.

It focuses on reproducibility, validation, backtesting discipline, monitoring, automation, and deployment — the pieces needed to turn a forecasting model into an operational ML workflow.

