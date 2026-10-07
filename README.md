# Retail Demand MLOps Platform

An end-to-end demand-forecasting system whose main goal is to demonstrate the ML lifecycle around a model: reproducible data layers, time-aware features, training, promotion, serving, monitoring, and retraining decisions.

```text
raw demand data
  -> bronze / silver / gold tables
  -> time-aware features
  -> training and experiment tracking
  -> registry quality gate
  -> batch or FastAPI inference
  -> drift and error monitoring
  -> retraining plan
```

The example data includes seasonality, promotions, prices, stockouts, and store/SKU effects. A gradient-boosted regressor is compared with a simple baseline using forecasting metrics such as WAPE and sMAPE. The modeling problem stays small so the data and deployment workflow are easy to follow.

## Included

- deterministic synthetic demand generation and validation
- bronze, silver, and gold table builders, with an optional Spark/Delta path
- leakage-safe historical aggregates, time splits, and rolling-origin backtests
- local experiment tracking with optional MLflow integration
- conformal forecast intervals and candidate/champion regression gates
- batch scoring and FastAPI serving
- prediction/feature drift checks, bias alerts, segment errors, and prioritized retraining plans
- pytest coverage, Docker packaging, and GitHub Actions CI

## Setup

```bash
git clone https://github.com/amir-sbg/Retail-Demand-MLOps-Platform.git
cd Retail-Demand-MLOps-Platform
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m pytest -q
```

Run the complete local workflow:

```bash
mlops-demand run-all
```

Individual stages are also available:

```bash
mlops-demand make-data
mlops-demand build-lake
mlops-demand build-features
mlops-demand train
mlops-demand batch-predict
mlops-demand monitor
mlops-demand retrain-plan
```

The run writes generated data, model artifacts, predictions, registry files, and monitoring reports under `data/`, `artifacts/`, `models/registry/`, `predictions/`, and `reports/`. To start the API after a champion has been registered, install `requirements-api.txt` and run `uvicorn mlops_platform.api:app --reload`.

## Project layout

```text
src/mlops_platform/
├── data.py          raw demand generation and validation
├── lake.py          bronze and silver table builders
├── features.py      forecasting feature engineering
├── training.py      model training and experiment output
├── registry.py      candidate/champion promotion
├── inference.py     batch and record-level scoring
├── monitoring.py    PSI and quality monitoring
├── retraining.py    retraining decision plan
├── api.py           FastAPI serving layer
└── spark_jobs.py    optional Spark feature job
tests/
```
