"""
Step 5 — FastAPI Model Serving
================================
Loads the trained pipeline and exposes REST endpoints for real-time
and batch inference.

Model loading priority:
  1. MLflow Model Registry (if MLFLOW_TRACKING_URI env var is set and server reachable)
  2. Local file at models/churn_pipeline.pkl  ← default for local dev

Usage:
    uvicorn api.main:app --reload --port 8000
    # Docs: http://localhost:8000/docs
"""

import os
import sys
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "")
MODEL_NAME      = os.getenv("MODEL_NAME",  "churn-classifier")
MODEL_STAGE     = os.getenv("MODEL_STAGE", "Production")
LOCAL_MODEL_PATH = os.path.join(ROOT, "models", "churn_pipeline.pkl")

# ── Global model handle ────────────────────────────────────────────────────────
_model      = None
_model_info = {}


def load_model():
    """Try MLflow registry first, fall back to local pickle."""
    global _model, _model_info

    if MLFLOW_URI:
        try:
            import mlflow.pyfunc
            mlflow.set_tracking_uri(MLFLOW_URI)
            uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            _model = mlflow.pyfunc.load_model(uri)
            _model_info = {"source": "mlflow_registry", "uri": uri}
            logger.info(f"Model loaded from MLflow Registry: {uri}")
            return
        except Exception as e:
            logger.warning(f"MLflow load failed ({e}), falling back to local model.")

    if os.path.exists(LOCAL_MODEL_PATH):
        _model = joblib.load(LOCAL_MODEL_PATH)
        _model_info = {"source": "local", "path": LOCAL_MODEL_PATH}
        logger.info(f"Model loaded from local file: {LOCAL_MODEL_PATH}")
        return

    raise RuntimeError(
        "No model found. Run 'python src/train.py' to train and save a model."
    )


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Predicts customer churn probability for telecom customers.\n\n"
        "**Model**: GradientBoostingClassifier trained on IBM Telco dataset.\n\n"
        "**Threshold**: 0.5  (churn_prediction=True if probability ≥ 0.5)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    # Numeric
    tenure:             int   = Field(..., ge=0,  le=100,   description="Months as customer (0–72)")
    MonthlyCharges:     float = Field(..., gt=0,  le=200,   description="Current monthly charge")
    TotalCharges:       float = Field(..., ge=0,            description="Total charges to date")
    charges_per_tenure: float = Field(..., ge=0,            description="TotalCharges / (tenure+1)")
    num_services:       int   = Field(..., ge=0,  le=8,     description="Number of add-on services active")

    # Binary numeric
    SeniorCitizen:      int   = Field(..., ge=0,  le=1)
    is_long_term:       int   = Field(..., ge=0,  le=1,     description="1 if tenure ≥ 24 months")

    # Yes/No categoricals
    Partner:            str   = Field(..., description="Yes or No")
    Dependents:         str   = Field(..., description="Yes or No")
    PhoneService:       str   = Field(..., description="Yes or No")
    PaperlessBilling:   str   = Field(..., description="Yes or No")

    # Multi-class categoricals
    gender:             str   = Field(..., description="Male or Female")
    MultipleLines:      str   = Field(..., description="Yes / No / No phone service")
    InternetService:    str   = Field(..., description="DSL / Fiber optic / No")
    OnlineSecurity:     str   = Field(..., description="Yes / No / No internet service")
    OnlineBackup:       str   = Field(..., description="Yes / No / No internet service")
    DeviceProtection:   str   = Field(..., description="Yes / No / No internet service")
    TechSupport:        str   = Field(..., description="Yes / No / No internet service")
    StreamingTV:        str   = Field(..., description="Yes / No / No internet service")
    StreamingMovies:    str   = Field(..., description="Yes / No / No internet service")
    PaymentMethod:      str   = Field(
        ...,
        description="Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)"
    )

    # Ordered categorical
    Contract:           str   = Field(..., description="Month-to-month / One year / Two year")

    @field_validator("Partner", "Dependents", "PhoneService", "PaperlessBilling")
    @classmethod
    def validate_yes_no(cls, v):
        if v not in {"Yes", "No"}:
            raise ValueError("Must be 'Yes' or 'No'")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        valid = {"Month-to-month", "One year", "Two year"}
        if v not in valid:
            raise ValueError(f"Must be one of {valid}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "Contract": "Month-to-month",
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability:  float
    churn_prediction:   bool
    risk_level:         str   = Field(..., description="low / medium / high")
    confidence:         str   = Field(..., description="Human-readable confidence band")
    timestamp:          str


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions:    list[PredictionResponse]
    total:          int
    high_risk_count: int


# ── Helpers ────────────────────────────────────────────────────────────────────
THRESHOLD = 0.5


def _risk(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "high",   f"Very likely to churn ({prob:.0%})"
    if prob >= 0.45:
        return "medium", f"At risk of churning ({prob:.0%})"
    return "low",        f"Likely to stay ({prob:.0%})"


def _predict_df(df: pd.DataFrame) -> list[float]:
    """Run inference; handles both MLflow pyfunc and sklearn pipeline."""
    if hasattr(_model, "predict_proba"):
        return _model.predict_proba(df)[:, 1].tolist()
    # MLflow pyfunc returns raw numpy / DataFrame
    raw = _model.predict(df)
    if hasattr(raw, "values"):
        raw = raw.values.flatten()
    return [float(v) for v in raw]


# ── Error handler ──────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Meta"])
def root():
    return {
        "service":     "Churn Prediction API",
        "version":     "1.0.0",
        "docs":        "/docs",
        "health":      "/health",
    }


@app.get("/health", tags=["Meta"])
def health():
    return {
        "status":      "ok" if _model is not None else "degraded",
        "model_loaded": _model is not None,
        "model_info":  _model_info,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(customer: CustomerFeatures):
    """Single-customer real-time prediction."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df   = pd.DataFrame([customer.model_dump()])
    prob = _predict_df(df)[0]
    risk, confidence = _risk(prob)

    logger.info(f"Predict: prob={prob:.3f} risk={risk}")

    return PredictionResponse(
        churn_probability = round(prob, 4),
        churn_prediction  = prob >= THRESHOLD,
        risk_level        = risk,
        confidence        = confidence,
        timestamp         = datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(body: BatchRequest):
    """Batch prediction for up to 500 customers."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df    = pd.DataFrame([c.model_dump() for c in body.customers])
    probs = _predict_df(df)
    ts    = datetime.now(timezone.utc).isoformat()

    preds = []
    for prob in probs:
        risk, confidence = _risk(prob)
        preds.append(PredictionResponse(
            churn_probability = round(prob, 4),
            churn_prediction  = prob >= THRESHOLD,
            risk_level        = risk,
            confidence        = confidence,
            timestamp         = ts,
        ))

    high_risk = sum(1 for p in preds if p.risk_level == "high")

    return BatchResponse(
        predictions     = preds,
        total           = len(preds),
        high_risk_count = high_risk,
    )


@app.get("/model/info", tags=["Meta"])
def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "model_info":   _model_info,
        "threshold":    THRESHOLD,
        "risk_levels": {
            "low":    "probability < 0.45",
            "medium": "0.45 ≤ probability < 0.70",
            "high":   "probability ≥ 0.70",
        },
    }
