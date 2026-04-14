<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/MLflow-2.13-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/pytest-tested-22c55e?style=for-the-badge&logo=pytest&logoColor=white"/>

<br/><br/>

# 🔁 MLOps Churn Prediction Pipeline

> A production-grade, end-to-end machine learning system that predicts which telecom customers are likely to cancel their service — and stays monitored in production.

</div>


## What This Project Does

Most ML tutorials stop at the notebook. This project answers what comes after:

- How do you **track and compare** dozens of experiments systematically?
- How do you **deploy** a model as a reliable REST API that handles bad inputs gracefully?
- How do you **know** when the model starts degrading in production?
- How do you make all of this **reproducible** on any machine?

This pipeline solves all of that using a real open-source dataset, production-grade tooling, and clean, modular code you can actually extend.



## The Business Problem

A telecom company loses significant revenue every time a customer churns (cancels their subscription). Acquiring a new customer costs 5–7× more than retaining an existing one.

**Goal:** Given a customer's profile and usage data, predict the probability they will churn in the next billing cycle — so the retention team can intervene before it happens.

**Dataset:** IBM Telco Customer Churn — 7,043 real customer records, 21 features, ~26% churn rate.


## What Gets Built

```
Raw CSV  →  Feature Engineering  →  Experiment Tracking  →  Best Model
                                                                  ↓
Production Traffic  →  Drift Monitor              REST API (real-time predictions)
        ↓                    ↓
    Alert triggered     HTML Report
```

### 1. Data Pipeline
Downloads the IBM Telco CSV directly from GitHub and produces clean, ML-ready splits. Handles a real data quality issue in the dataset: `TotalCharges` is stored as a string with blank values for new customers — this gets caught and fixed programmatically.

Three new features are engineered from the raw data:

| Feature | Formula | Why it matters |
|---|---|---|
| `charges_per_tenure` | `TotalCharges / (tenure + 1)` | Reveals price sensitivity — a customer paying a lot relative to their tenure is more likely to feel it's not worth it |
| `num_services` | Count of active add-ons (0–8) | Stickiness indicator — the more services a customer uses, the harder it is to leave |
| `is_long_term` | `1 if tenure ≥ 24 months` | Long-term customers churn at dramatically lower rates |

### 2. Feature Engineering Pipeline (no train/serve skew)

All transformations are wrapped in a single `sklearn.Pipeline`. This is a critical production decision: the exact same preprocessing logic runs during training AND at inference time. Without this, subtle differences in how data is scaled or encoded between training and serving cause silent, hard-to-debug accuracy degradation.

Different column types get appropriate treatment:

| Type | Strategy | Columns | Reason |
|---|---|---|---|
| Numeric | Median impute → StandardScaler | tenure, charges, etc. | Robust to outliers |
| Ordinal | OrdinalEncoder with explicit order | Contract | Month-to-month < One year < Two year — preserving this order helps trees find better splits |
| Binary Yes/No | OrdinalEncoder (No=0, Yes=1) | Partner, Dependents, etc. | Compact; avoids unnecessary dummy variables |
| Multi-class | OneHotEncoder (drop first) | PaymentMethod, InternetService, etc. | No implicit ordinal assumption; avoids multicollinearity |
| Passthrough | No transformation | SeniorCitizen | Already 0/1 |

### 3. Experiment Tracking with MLflow

Three GradientBoosting models are trained with different learning rates. Every run automatically logs:

- All hyperparameters (n_estimators, learning_rate, max_depth, subsample, etc.)
- 5-fold stratified cross-validation scores (mean ± std) for ROC-AUC, F1, Precision, Recall
- Final test set metrics
- Three diagnostic plots: ROC curve, confusion matrix, feature importance (top 20)
- The full sklearn Pipeline as a versioned artifact in the MLflow Model Registry
- A local `.pkl` backup so the API works even without a running MLflow server

You can compare all runs side-by-side in the MLflow UI and see exactly which hyperparameter combination gave the best generalization.

### 4. Model Promotion

`promote_model.py` queries MLflow for the run with the highest `test_roc_auc`, transitions that version to **Production** in the Model Registry, and archives any previous Production version. This mimics a real CI/CD gate: only the best-validated model serves traffic.

### 5. Prediction API

A FastAPI application with two inference endpoints:

- **`POST /predict`** — real-time single-customer prediction, returns probability + risk level + human-readable confidence string
- **`POST /predict/batch`** — batch inference for up to 500 customers in one call, returns total count and high-risk count summary

The API handles both MLflow-served models (via Model Registry) and local `.pkl` files, so it works in all environments. Pydantic v2 validates every incoming request — invalid contract types, out-of-range values, and missing fields are all rejected with clear 422 errors before they ever reach the model.

Risk levels:

| Level | Probability | Recommended Action |
|---|---|---|
| `low` | < 45% | No action needed |
| `medium` | 45% – 70% | Soft outreach (promotional offer) |
| `high` | ≥ 70% | Immediate retention intervention |

### 6. Data Drift Monitoring

Production data distributions shift over time — prices change, customer demographics evolve, service packages get updated. When input distributions drift too far from what the model was trained on, predictions silently degrade.

`monitoring/monitor.py` runs Evidently AI's statistical tests (Kolmogorov-Smirnov for continuous features) comparing the training set (reference) against the latest production batch. It outputs:

- A full interactive HTML report per run (saved to `reports/`)
- A JSON summary with per-column p-values and drift flags
- A console alert with actionable recommendation if ≥ 30% of columns drift
- A distribution comparison table showing mean shift (Δ%) per column

The production sample has a built-in synthetic drift on `MonthlyCharges` (+$6 mean shift) to demonstrate detection working end-to-end.



## Tech Stack

| Layer | Tool | Why this choice |
|---|---|---|
| Language | Python 3.11 | Type hints, match statements, performance improvements |
| Data | pandas + numpy | Industry standard for tabular data manipulation |
| ML | scikit-learn | Battle-tested Pipeline API; excellent for tabular data |
| Experiment tracking | MLflow | Open source, self-hosted, UI + Model Registry in one |
| Model | GradientBoostingClassifier | Strong on tabular data; interpretable feature importances; no GPU needed |
| API | FastAPI | Async, auto-generates OpenAPI docs, Pydantic validation built in |
| Monitoring | Evidently AI | Purpose-built for ML data drift; generates shareable HTML reports |
| Containerization | Docker Compose | Reproducible multi-service setup in one command |
| Testing | pytest | Clean fixture system; works with FastAPI's TestClient |



## Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (downloads data, trains 3 models, runs monitoring)
python run_pipeline.py --skip-promote

# 3. Start the prediction API
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** — the Swagger UI lets you send predictions directly from the browser.



## Step-by-Step Guide

### Step 1 — Download & Prepare Data

```bash
python data/load_data.py
```

Fetches the IBM Telco CSV from GitHub, fixes the `TotalCharges` string issue,
engineers 3 new features, and writes three files to `data/raw/`:

- `train.csv` — 5,634 rows for training
- `test.csv` — 1,409 rows for evaluation
- `production_sample.csv` — simulated production traffic with slight drift on `MonthlyCharges`

### Step 2 — Train Models + Track Experiments

```bash
# Optional: start MLflow UI to watch experiments in real time
mlflow ui --port 5000    # then open http://localhost:5000

python src/train.py
```

Runs 3 experiments (learning rates: 0.03, 0.05, 0.10). Each takes ~60 seconds.
When done, open the MLflow UI to compare runs on the Metrics tab.

Expected performance range:
- ROC-AUC: ~0.84 – 0.86
- F1 Score: ~0.60 – 0.64
- Recall (catching churners): ~0.72 – 0.78

### Step 3 — Promote the Best Model

```bash
python promote_model.py
```

Automatically finds the highest ROC-AUC run and promotes it to Production.

> Skip this step if you are not running an MLflow server. The API automatically falls back to `models/churn_pipeline.pkl`.

### Step 4 — Start the Prediction API

```bash
uvicorn api.main:app --reload --port 8000
```

#### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Model load status + source |
| `GET` | `/model/info` | Risk thresholds + model metadata |
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 500 customers) |

#### Example — High Risk Customer

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 3,
    "MonthlyCharges": 95.75,
    "TotalCharges": 287.25,
    "charges_per_tenure": 71.81,
    "num_services": 1,
    "SeniorCitizen": 0,
    "is_long_term": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes",
    "gender": "Male",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "PaymentMethod": "Electronic check",
    "Contract": "Month-to-month"
  }'
```

```json
{
  "churn_probability": 0.7823,
  "churn_prediction": true,
  "risk_level": "high",
  "confidence": "Very likely to churn (78%)",
  "timestamp": "2026-01-01T12:00:00+00:00"
}
```

### Step 5 — Run Data Drift Monitoring

```bash
python monitoring/monitor.py
```

Runs in ~10 seconds. Sample output:

```
[MONITOR] Per-column drift summary:
  Column                    Status   p-value    Stat score   Test
  -----------------------------------------------------------------
  tenure                    OK    ✓  0.8821     0.05         ks
  MonthlyCharges            DRIFT ⚠  0.0012     0.05         ks
  TotalCharges              OK    ✓  0.5340     0.05         ks
  charges_per_tenure        DRIFT ⚠  0.0089     0.05         ks
  num_services              OK    ✓  0.4120     0.05         ks

[SUMMARY] Drift ratio: 40% (2/5 columns drifted)

  ================================================
  ⚠  DATA DRIFT ALERT
  Drifted columns: MonthlyCharges, charges_per_tenure
  Recommendation:  Consider retraining the model.
  ================================================
```

Reports saved to `reports/drift_report_<timestamp>.html`.



## Run Tests

```bash
pytest tests/ -v
```

The test suite covers:

- Health check and model load verification
- Single prediction returns correct schema and probability in [0, 1]
- High-risk customers score higher than low-risk customers (business logic test)
- Batch prediction returns correct count and high-risk summary
- Invalid contract type → 422
- Out-of-range tenure → 422
- Missing required field → 422
- Empty batch list → 422



## Docker (Full Stack)

Starts the MLflow tracking server, trains the model, launches the API, and runs monitoring — all from a single command.

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| Prediction API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |



## Project Structure

```
mlops-churn/
│
├── data/
│   └── load_data.py          # Download IBM Telco CSV, clean, feature engineer, split
│
├── src/
│   ├── preprocess.py         # sklearn ColumnTransformer covering all feature types
│   ├── train.py              # Training loop + MLflow experiment tracking + local save
│   └── evaluate.py           # Metrics + ROC curve / CM / feature importance plots
│
├── api/
│   └── main.py               # FastAPI: /predict, /predict/batch, /health, /model/info
│
├── monitoring/
│   └── monitor.py            # Evidently drift detection + HTML report + JSON summary + alert
│
├── tests/
│   └── test_api.py           # pytest: schema, edge cases, business logic
│
├── promote_model.py          # Query MLflow → find best run → promote to Production
├── run_pipeline.py           # One-click runner: data → train → promote → monitor
│
├── Dockerfile
├── docker-compose.yml        # MLflow server + trainer + API + monitoring
├── requirements.txt
└── README.md
```


## Dataset

**IBM Telco Customer Churn**
- Source: https://github.com/IBM/telco-customer-churn-on-icp4d
- 7,043 customers · 21 columns · ~26% churn rate
- License: Apache 2.0
- No signup or API key required — downloads automatically on first run



