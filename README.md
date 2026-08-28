# Retail Demand MLOps Platform

An end-to-end MLOps project for demand forecasting. The model is intentionally simple; the focus is the production workflow around it: data layers, feature engineering, training, experiment tracking, model promotion, inference, monitoring, and retraining decisions.

```text
raw demand data
  -> bronze / silver / gold tables
  -> time-aware features
  -> training + experiment tracking
  -> model registry gate
  -> batch and FastAPI inference
  -> drift / quality monitoring
  -> retraining plan
```

## What is inside

- Synthetic retail demand data generator with seasonality, promotions, prices, stockouts, and store/SKU effects
- Bronze, silver, and gold feature-table pipeline, with an optional Spark/Delta implementation path
- Time-based train/validation/test split for forecasting instead of random leakage-prone splitting
- Gradient-boosted demand model with baseline comparison, residual summaries, and promotion gates
- Local experiment tracking that can also log to MLflow when MLflow is installed
- Lightweight model registry with champion/candidate stages
- Batch scoring, FastAPI serving, prediction logging, drift checks, and retraining-plan generation
- Pytest coverage, Docker packaging, and GitHub Actions CI

## Tech stack

Python, pandas, scikit-learn, optional Spark/Delta Lake, optional MLflow, FastAPI, Docker, pytest, ruff, and GitHub Actions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .

python -m pytest -q
mlops-demand run-all
```

The full run writes generated data, run artifacts, registry files, predictions, and monitoring reports into ignored local folders:

```text
data/
artifacts/
models/registry/
predictions/
reports/
```

## Useful commands

```bash
mlops-demand make-data
mlops-demand build-lake
mlops-demand build-features
mlops-demand train
mlops-demand batch-predict
mlops-demand monitor
mlops-demand retrain-plan
```

Run the API after a champion model has been registered:

```bash
python -m pip install -r requirements-api.txt
uvicorn mlops_platform.api:app --reload
```

## Project structure

```text
src/mlops_platform/
  data.py          # raw demand generation and validation
  lake.py          # bronze and silver table builders
  features.py      # forecasting feature engineering
  training.py      # model training and experiment output
  registry.py      # candidate/champion model promotion
  inference.py     # batch and record-level scoring
  monitoring.py    # PSI drift and quality checks
  retraining.py    # retraining decision plan
  api.py           # FastAPI serving layer
  spark_jobs.py    # optional Spark feature job
```

The project is designed around the kind of lifecycle work that matters in production ML: keeping data transformations reproducible, comparing against a baseline, promoting models only when gates pass, and monitoring whether the deployed model still behaves like the model that was validated.
