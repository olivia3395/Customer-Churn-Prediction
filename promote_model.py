"""
Step 4 — Promote Best Model to Production
==========================================
Queries MLflow for the best run (by test ROC-AUC) in the experiment,
then promotes that model version to the "Production" stage in the
Model Registry.

Usage:
    python promote_model.py
"""

import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT}/mlflow.db")
EXPERIMENT_NAME = "churn-prediction"
REGISTERED_NAME = "churn-classifier"
METRIC          = "test_roc_auc"


def get_best_run() -> tuple[str, float]:
    """Return (run_id, metric_value) for the best run in the experiment."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found. "
            "Run 'python src/train.py' first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{METRIC} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found. Run 'python src/train.py' first.")

    best = runs[0]
    return best.info.run_id, best.data.metrics[METRIC]


def promote_model(run_id: str) -> None:
    """Transition the model version linked to run_id to Production."""
    client = MlflowClient()

    # Find the model version registered from this run
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    target = None
    for v in versions:
        if v.run_id == run_id:
            target = v
            break

    if target is None:
        raise RuntimeError(
            f"No registered model version found for run {run_id}.\n"
            "Ensure MLflow logged the model with registered_model_name set."
        )

    # Archive any existing Production versions
    for v in versions:
        if v.current_stage == "Production" and v.version != target.version:
            client.transition_model_version_stage(
                name=REGISTERED_NAME,
                version=v.version,
                stage="Archived",
            )
            print(f"[ARCHIVE] Version {v.version} → Archived")

    # Promote target to Production
    client.transition_model_version_stage(
        name=REGISTERED_NAME,
        version=target.version,
        stage="Production",
    )
    print(f"[PROMOTE] Version {target.version} (run={run_id[:8]}...) → Production")


if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_URI)

    print(f"[INFO] Looking for best run by '{METRIC}'...")
    run_id, score = get_best_run()
    print(f"[INFO] Best run: {run_id[:8]}...  {METRIC}={score:.4f}")

    promote_model(run_id)
    print(f"\n[DONE] Model promoted to Production.")
    print("[NEXT] Start API: uvicorn api.main:app --reload --port 8000")
