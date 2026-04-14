"""
API Tests
==========
Pytest test suite for the FastAPI application.
Tests health check, single predict, batch predict, and validation errors.

Usage:
    pytest tests/ -v
    pytest tests/ -v --tb=short
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import app (model loads on startup)
from api.main import app

client = TestClient(app)

# ── Reusable fixture ───────────────────────────────────────────────────────────
HIGH_RISK_CUSTOMER = {
    "tenure": 2,
    "MonthlyCharges": 99.5,
    "TotalCharges": 199.0,
    "charges_per_tenure": 66.3,
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

LOW_RISK_CUSTOMER = {
    **HIGH_RISK_CUSTOMER,
    "tenure": 60,
    "TotalCharges": 5990.0,
    "charges_per_tenure": 98.2,
    "num_services": 6,
    "Contract": "Two year",
    "PaymentMethod": "Credit card (automatic)",
    "is_long_term": 1,
}


# ── Health check ───────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()

    def test_model_info(self):
        r = client.get("/model/info")
        assert r.status_code == 200
        body = r.json()
        assert "threshold" in body
        assert body["threshold"] == 0.5


# ── Single prediction ──────────────────────────────────────────────────────────
class TestPredict:
    def test_predict_returns_200(self):
        r = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert r.status_code == 200

    def test_predict_response_schema(self):
        r = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        body = r.json()
        assert "churn_probability"  in body
        assert "churn_prediction"   in body
        assert "risk_level"         in body
        assert "confidence"         in body
        assert "timestamp"          in body

    def test_predict_probability_range(self):
        r = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        prob = r.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_risk_levels(self):
        for customer in [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]:
            r = client.post("/predict", json=customer)
            risk = r.json()["risk_level"]
            assert risk in {"low", "medium", "high"}

    def test_high_risk_customer(self):
        """Month-to-month, fiber optic, short tenure → should be higher risk than long-term"""
        r_high = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        r_low  = client.post("/predict", json=LOW_RISK_CUSTOMER)
        assert r_high.json()["churn_probability"] > r_low.json()["churn_probability"]

    def test_prediction_boolean(self):
        r = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert isinstance(r.json()["churn_prediction"], bool)


# ── Batch prediction ───────────────────────────────────────────────────────────
class TestBatchPredict:
    def test_batch_predict_returns_200(self):
        r = client.post("/predict/batch", json={"customers": [HIGH_RISK_CUSTOMER]})
        assert r.status_code == 200

    def test_batch_predict_multiple(self):
        payload = {"customers": [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]}
        r = client.post("/predict/batch", json=payload)
        body = r.json()
        assert body["total"] == 2
        assert len(body["predictions"]) == 2

    def test_batch_high_risk_count(self):
        payload = {"customers": [HIGH_RISK_CUSTOMER] * 5}
        r = client.post("/predict/batch", json=payload)
        body = r.json()
        assert 0 <= body["high_risk_count"] <= 5

    def test_batch_empty_list_rejected(self):
        r = client.post("/predict/batch", json={"customers": []})
        assert r.status_code == 422   # Pydantic min_length=1


# ── Input validation ───────────────────────────────────────────────────────────
class TestValidation:
    def test_invalid_contract_rejected(self):
        bad = {**HIGH_RISK_CUSTOMER, "Contract": "Daily"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_yes_no_rejected(self):
        bad = {**HIGH_RISK_CUSTOMER, "Partner": "Maybe"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_negative_tenure_rejected(self):
        bad = {**HIGH_RISK_CUSTOMER, "tenure": -1}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_missing_field_rejected(self):
        incomplete = {k: v for k, v in HIGH_RISK_CUSTOMER.items()
                      if k != "MonthlyCharges"}
        r = client.post("/predict", json=incomplete)
        assert r.status_code == 422

    def test_extra_fields_ignored(self):
        """Extra fields should not cause a 422 (Pydantic ignores by default)."""
        with_extra = {**HIGH_RISK_CUSTOMER, "unknown_field": "whatever"}
        r = client.post("/predict", json=with_extra)
        assert r.status_code == 200
