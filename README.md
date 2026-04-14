# MLOps Churn Prediction Pipeline

End-to-end MLOps project using the **IBM Telco Customer Churn** open dataset.
Covers data ingestion, feature engineering, experiment tracking, model serving,
and production monitoring.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Dataset | IBM Telco Customer Churn (GitHub) |
| Feature Engineering | scikit-learn Pipeline + ColumnTransformer |
| Experiment Tracking | MLflow |
| Model | GradientBoostingClassifier |
| API | FastAPI + Pydantic v2 |
| Monitoring | Evidently AI |
| Containerization | Docker + Docker Compose |
| Testing | pytest |

---

## Quick Start (Local — No Docker)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the entire pipeline in one command
python run_pipeline.py --skip-promote

# 3. Start the prediction API
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Step-by-Step

### Step 1 — Download & Prepare Data

```bash
python data/load_data.py
```

Downloads the IBM Telco CSV from GitHub (~7,000 rows), cleans it,
engineers 3 new features, and splits into `train.csv`, `test.csv`,
and `production_sample.csv`.

**Engineered features:**
- `charges_per_tenure` — Total charges divided by tenure. Captures price sensitivity.
- `num_services` — Count of active add-on services (0–8). Measures stickiness.
- `is_long_term` — Binary flag: 1 if tenure ≥ 24 months.

### Step 2 — Train Models

```bash
# Optionally start MLflow UI first:
mlflow ui --port 5000

python src/train.py
```

Trains 3 GradientBoosting models with different learning rates.
Each run logs:
- All hyperparameters
- 5-fold cross-validation metrics (ROC-AUC, F1, Precision, Recall)
- Test set metrics
- ROC curve + confusion matrix + feature importance plots
- The full sklearn Pipeline as a registered MLflow model
- A local copy at `models/churn_pipeline.pkl`

View experiments at **http://localhost:5000**.

### Step 3 — Promote Best Model

```bash
python promote_model.py
```

Queries MLflow for the run with the highest `test_roc_auc`,
promotes that version to **Production** in the Model Registry,
and archives any previous Production version.

> Skip this step if you are not running an MLflow server.
> The API falls back to `models/churn_pipeline.pkl` automatically.

### Step 4 — Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check + model status |
| GET | `/model/info` | Model metadata + thresholds |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch prediction (up to 500) |

**Example request:**

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

**Example response:**

```json
{
  "churn_probability": 0.7823,
  "churn_prediction": true,
  "risk_level": "high",
  "confidence": "Very likely to churn (78%)",
  "timestamp": "2026-01-01T12:00:00+00:00"
}
```

### Step 5 — Monitor Data Drift

```bash
python monitoring/monitor.py
```

Compares the training distribution (reference) against `production_sample.csv`
using Evidently's statistical tests. Outputs:

- `reports/drift_report_<timestamp>.html` — full interactive Evidently report
- `reports/drift_summary_<timestamp>.json` — machine-readable summary
- Console alert if drift ratio ≥ 30%

---

## Run Tests

```bash
pytest tests/ -v
```

Tests cover health check, single/batch prediction, schema validation,
and business logic (high-risk customers score higher than low-risk).

---

## Docker

```bash
# Build and start all services
docker-compose up --build

# Services:
#   MLflow UI    →  http://localhost:5000
#   Prediction API → http://localhost:8000
#   API docs     →  http://localhost:8000/docs
```

---

## Project Structure

```
mlops-churn/
├── data/
│   └── load_data.py          # download + clean + split
├── src/
│   ├── preprocess.py         # sklearn ColumnTransformer pipeline
│   ├── train.py              # training + MLflow experiment tracking
│   └── evaluate.py           # metrics + diagnostic plots
├── api/
│   └── main.py               # FastAPI serving
├── monitoring/
│   └── monitor.py            # Evidently data drift detection
├── tests/
│   └── test_api.py           # pytest test suite
├── promote_model.py          # promote best MLflow run to Production
├── run_pipeline.py           # one-click end-to-end runner
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Dataset

**IBM Telco Customer Churn**
- Source: https://github.com/IBM/telco-customer-churn-on-icp4d
- 7,043 rows · 21 columns · ~26% churn rate
- License: Apache 2.0

---

## Key Design Decisions

**Why sklearn Pipeline?**
Wrapping preprocessing and the model in a single Pipeline prevents
train/serve skew — the exact same transformations run at both training
and inference time.

**Why `Contract` uses OrdinalEncoder?**
Contract type has a natural order (Month-to-month < One year < Two year)
that correlates with churn risk. Preserving the ordering gives gradient
boosting trees better split candidates than one-hot encoding would.

**Why save a local `.pkl` as well as logging to MLflow?**
The API falls back to the local pickle if no MLflow tracking server is
reachable, making local development easier without needing Docker.
