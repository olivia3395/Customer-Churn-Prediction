"""
Step 3 — Model Training + MLflow Experiment Tracking
======================================================
Trains a GradientBoostingClassifier pipeline, logs everything to MLflow,
saves plots, and also persists the model locally so the API can load it
without a running MLflow server.

Usage:
    # Start MLflow UI first (optional but recommended):
    #   mlflow ui --port 5000
    python src/train.py
"""

import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

# ── Path setup (run from project root) ───────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import build_preprocessor, load_data, feature_names_out
from src.evaluate import compute_metrics, print_report, plot_roc_curve, \
                          plot_confusion_matrix, plot_feature_importance

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH  = os.path.join(ROOT, "data", "raw", "train.csv")
TEST_PATH   = os.path.join(ROOT, "data", "raw", "test.csv")
MODEL_PATH  = os.path.join(ROOT, "models", "churn_pipeline.pkl")

# ── MLflow Config ─────────────────────────────────────────────────────────────
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlflow.db")
EXPERIMENT_NAME  = "churn-prediction"
REGISTERED_NAME  = "churn-classifier"


def train(params: dict | None = None, run_name: str | None = None) -> str:
    """
    Train the full pipeline and log to MLflow.
    Returns the MLflow run_id.
    """
    # Default hyperparameters
    default_params = {
        "n_estimators":       300,
        "learning_rate":      0.05,
        "max_depth":          4,
        "min_samples_split":  20,
        "min_samples_leaf":   10,
        "subsample":          0.8,
        "max_features":       "sqrt",
    }
    params = {**default_params, **(params or {})}
    run_name = run_name or f"GB_lr{params['learning_rate']}_d{params['max_depth']}"

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[TRAIN] Loading data...")
    X_train, y_train = load_data(TRAIN_PATH)
    X_test,  y_test  = load_data(TEST_PATH)
    print(f"[TRAIN] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[TRAIN] Churn rate — train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    preprocessor = build_preprocessor()
    classifier   = GradientBoostingClassifier(**params, random_state=42)
    pipeline     = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\n[MLFLOW] Run ID: {run_id}")

        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.set_tags({
            "model_type": "GradientBoosting",
            "framework":  "scikit-learn",
            "dataset":    "IBM-Telco-Churn",
            "version":    "1.0",
        })

        # ── Cross-validation ──────────────────────────────────────────────────
        print("[TRAIN] Running 5-fold cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=["roc_auc", "f1", "precision", "recall"],
            return_train_score=False,
        )
        for metric, scores in cv_results.items():
            if metric.startswith("test_"):
                name = metric.replace("test_", "cv_")
                mlflow.log_metric(f"{name}_mean", round(scores.mean(), 4))
                mlflow.log_metric(f"{name}_std",  round(scores.std(),  4))
                print(f"  {name:<25} {scores.mean():.4f} ± {scores.std():.4f}")

        # ── Train on full training set ────────────────────────────────────────
        print("\n[TRAIN] Fitting final model...")
        pipeline.fit(X_train, y_train)

        y_pred      = pipeline.predict(X_test)
        y_prob      = pipeline.predict_proba(X_test)[:, 1]

        # ── Log test metrics ──────────────────────────────────────────────────
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        print_report(y_test, y_pred, y_prob)

        # ── Log plots ─────────────────────────────────────────────────────────
        roc_path = plot_roc_curve(y_test, y_prob, run_name)
        cm_path  = plot_confusion_matrix(y_test, y_pred, run_name)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)

        # Feature importance plot
        feat_names   = feature_names_out(pipeline.named_steps["preprocessor"])
        importances  = pipeline.named_steps["classifier"].feature_importances_
        fi_path      = plot_feature_importance(feat_names, importances, run_name)
        mlflow.log_artifact(fi_path)

        # ── Log model to MLflow ───────────────────────────────────────────────
        signature = infer_signature(X_train, y_prob)
        mlflow.sklearn.log_model(
            sk_model              = pipeline,
            artifact_path         = "model",
            signature             = signature,
            input_example         = X_train.head(3),
            registered_model_name = REGISTERED_NAME,
        )

        # ── Save model locally for API (no MLflow server required) ───────────
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"\n[SAVED] Local model → {MODEL_PATH}")
        print(f"[MLFLOW] View experiments: mlflow ui --port 5000")
        print(f"[DONE] Run complete: {run_id}")

    return run_id


if __name__ == "__main__":
    # Run a small grid search across learning rates — each is a separate run
    grid = [
        {"learning_rate": 0.03, "n_estimators": 400},
        {"learning_rate": 0.05, "n_estimators": 300},
        {"learning_rate": 0.10, "n_estimators": 200},
    ]

    for p in grid:
        train(params=p)

    print("\n[DONE] All runs complete.")
    print("[NEXT] Promote best model: python promote_model.py")
