"""
One-Click Pipeline Runner
==========================
Runs every step of the MLOps pipeline in sequence:

  1. Download & prepare data
  2. Train models (3 learning rate configs)
  3. Promote best model to MLflow Production stage
  4. Run data drift monitoring

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-promote   # skip if MLflow server not running
"""

import argparse
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_step(description: str, command: list[str], check: bool = True) -> bool:
    print(f"\n{'='*60}")
    print(f"  ▶  {description}")
    print(f"{'='*60}")
    start = time.time()

    result = subprocess.run(command, cwd=ROOT)

    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n  ✓  Done in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ✗  Failed (exit code {result.returncode})")
        if check:
            sys.exit(result.returncode)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the full MLOps pipeline")
    parser.add_argument(
        "--skip-promote",
        action="store_true",
        help="Skip the MLflow model promotion step",
    )
    parser.add_argument(
        "--skip-monitor",
        action="store_true",
        help="Skip the monitoring step",
    )
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("  MLOps Churn Prediction Pipeline")
    print("  IBM Telco Customer Churn Dataset")
    print("█" * 60)

    total_start = time.time()

    # Step 1: Data
    run_step(
        "Step 1/4 — Download & prepare data",
        [sys.executable, "data/load_data.py"],
    )

    # Step 2: Train
    run_step(
        "Step 2/4 — Train models + track experiments in MLflow",
        [sys.executable, "src/train.py"],
    )

    # Step 3: Promote (optional)
    if not args.skip_promote:
        success = run_step(
            "Step 3/4 — Promote best model to Production",
            [sys.executable, "promote_model.py"],
            check=False,
        )
        if not success:
            print(
                "\n  [WARN] Model promotion failed — this is OK if you are not\n"
                "         running an MLflow tracking server. The API will load\n"
                "         the model from models/churn_pipeline.pkl instead.\n"
                "         To start MLflow: mlflow ui --port 5000\n"
            )
    else:
        print("\n[SKIP] Model promotion skipped.")

    # Step 4: Monitor
    if not args.skip_monitor:
        run_step(
            "Step 4/4 — Run data drift monitoring",
            [sys.executable, "monitoring/monitor.py"],
        )
    else:
        print("\n[SKIP] Monitoring skipped.")

    total_elapsed = time.time() - total_start

    print("\n" + "█" * 60)
    print(f"  Pipeline complete in {total_elapsed:.0f}s")
    print("█" * 60)
    print("""
  Next steps:
  ──────────────────────────────────────────────────────
  Start the API:
    uvicorn api.main:app --reload --port 8000

  Open API docs:
    http://localhost:8000/docs

  View MLflow experiments:
    mlflow ui --port 5000
    http://localhost:5000

  Run tests:
    pytest tests/ -v

  Start all services with Docker:
    docker-compose up --build
  ──────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
