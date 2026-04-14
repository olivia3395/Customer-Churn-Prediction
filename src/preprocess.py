"""
Step 2 — Preprocessing Pipeline
=================================
Defines the sklearn ColumnTransformer for the Telco dataset.
All transformations are encapsulated in a pipeline to prevent
train/serve skew (same logic at training and inference time).
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# ── Feature Groups ────────────────────────────────────────────────────────────

TARGET = "Churn"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_tenure",   # engineered
    "num_services",         # engineered
]

# Ordered: contract length has natural ordering that trees can exploit
ORDINAL_FEATURES  = ["Contract"]
CONTRACT_ORDER    = [["Month-to-month", "One year", "Two year"]]

# Binary Yes/No columns → encode as 0/1
BINARY_YES_NO = [
    "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "is_long_term",
]

# Multi-class categoricals (3+ distinct values)
CATEGORICAL_FEATURES = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaymentMethod",
]

# Already numeric 0/1
PASSTHROUGH_FEATURES = ["SeniorCitizen"]


# ── Sub-Pipelines ──────────────────────────────────────────────────────────────

def _numeric_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def _ordinal_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=CONTRACT_ORDER,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])


def _binary_pipe():
    """Maps Yes→1, No→0 via OrdinalEncoder."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["No", "Yes"]] * len(BINARY_YES_NO),
            handle_unknown="use_encoded_value",
            unknown_value=0,
        )),
    ])


def _categorical_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",           # avoid dummy variable trap
        )),
    ])


# ── Public API ────────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer covering all feature groups."""
    return ColumnTransformer(
        transformers=[
            ("num",  _numeric_pipe(),     NUMERIC_FEATURES),
            ("ord",  _ordinal_pipe(),     ORDINAL_FEATURES),
            ("bin",  _binary_pipe(),      BINARY_YES_NO),
            ("cat",  _categorical_pipe(), CATEGORICAL_FEATURES),
            ("pass", "passthrough",       PASSTHROUGH_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_data(csv_path: str):
    """Load a CSV and return (X, y) split."""
    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {csv_path}")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Return feature names after fitting, useful for SHAP / feature importance."""
    return list(preprocessor.get_feature_names_out())
