"""
Step 6 — Data Drift Monitoring
================================
Compares the training set (reference) against incoming production data
using Evidently AI. Generates a full HTML report and prints a per-column
drift summary. Triggers a console alert if drift ratio exceeds a threshold.

Usage:
    python monitoring/monitor.py

For scheduled monitoring (e.g. daily):
    Add to cron: 0 8 * * * cd /path/to/mlops-churn && python monitoring/monitor.py
"""

import os
import sys
import json
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)

# ── Config ────────────────────────────────────────────────────────────────────
REFERENCE_PATH  = os.path.join(ROOT, "data", "raw", "train.csv")
CURRENT_PATH    = os.path.join(ROOT, "data", "raw", "production_sample.csv")
REPORTS_DIR     = os.path.join(ROOT, "reports")
DRIFT_THRESHOLD = 0.3   # alert if >30% of columns drift

# Columns to monitor (numeric only for statistical tests)
MONITOR_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_tenure",
    "num_services",
]


# ── Load & validate data ──────────────────────────────────────────────────────
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [REFERENCE_PATH, CURRENT_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Run 'python data/load_data.py' first."
            )

    ref = pd.read_csv(REFERENCE_PATH)[MONITOR_COLS]
    cur = pd.read_csv(CURRENT_PATH)[MONITOR_COLS]

    # Basic sanity checks
    assert len(ref) > 0, "Reference dataset is empty"
    assert len(cur) > 0, "Current dataset is empty"

    print(f"[DATA] Reference rows: {len(ref)} | Current rows: {len(cur)}")
    return ref, cur


# ── Run monitoring ────────────────────────────────────────────────────────────
def run_monitoring() -> dict:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ref, cur = load_datasets()

    # ── 1. Full drift + quality report (HTML) ─────────────────────────────────
    print("\n[MONITOR] Generating drift report...")
    full_report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    full_report.run(reference_data=ref, current_data=cur)

    html_path = os.path.join(REPORTS_DIR, f"drift_report_{timestamp}.html")
    full_report.save_html(html_path)
    print(f"[REPORT] Saved → {html_path}")

    # ── 2. Per-column drift metrics ───────────────────────────────────────────
    col_report = Report(metrics=[
        ColumnDriftMetric(column_name=col) for col in MONITOR_COLS
    ])
    col_report.run(reference_data=ref, current_data=cur)
    results = col_report.as_dict()

    # ── 3. Parse results ──────────────────────────────────────────────────────
    print("\n[MONITOR] Per-column drift summary:")
    print(f"  {'Column':<25} {'Status':<8} {'p-value':<10} {'Stat score':<12} Test")
    print("  " + "-" * 65)

    drifted_cols = []
    col_summary  = []

    for metric_result in results["metrics"]:
        r = metric_result.get("result", {})
        col_name   = r.get("column_name", "unknown")
        drifted    = r.get("drift_detected", False)
        p_val      = r.get("p_value")
        stat_val   = r.get("stattest_threshold")
        test_name  = r.get("stattest_name", "")

        status = "DRIFT ⚠" if drifted else "OK    ✓"
        p_str  = f"{p_val:.4f}" if p_val is not None else "N/A"
        s_str  = f"{stat_val:.4f}" if stat_val is not None else "N/A"

        print(f"  {col_name:<25} {status:<8} {p_str:<10} {s_str:<12} {test_name}")

        if drifted:
            drifted_cols.append(col_name)

        col_summary.append({
            "column":          col_name,
            "drift_detected":  drifted,
            "p_value":         p_val,
            "stattest_name":   test_name,
        })

    drift_ratio = len(drifted_cols) / len(MONITOR_COLS)

    # ── 4. Distribution comparison (mean / std shift) ─────────────────────────
    print("\n[MONITOR] Distribution comparison (Reference vs Current):")
    print(f"  {'Column':<25} {'Ref mean':>10} {'Cur mean':>10} {'Δ%':>8}")
    print("  " + "-" * 55)
    for col in MONITOR_COLS:
        r_mean = ref[col].mean()
        c_mean = cur[col].mean()
        delta  = ((c_mean - r_mean) / (abs(r_mean) + 1e-9)) * 100
        print(f"  {col:<25} {r_mean:>10.2f} {c_mean:>10.2f} {delta:>+8.1f}%")

    # ── 5. Alert ──────────────────────────────────────────────────────────────
    print(f"\n[SUMMARY] Drift ratio: {drift_ratio:.0%} "
          f"({len(drifted_cols)}/{len(MONITOR_COLS)} columns drifted)")

    summary = {
        "timestamp":      timestamp,
        "drift_ratio":    round(drift_ratio, 4),
        "drifted_columns": drifted_cols,
        "columns":        col_summary,
        "report_path":    html_path,
        "alert_triggered": drift_ratio >= DRIFT_THRESHOLD,
    }

    if drift_ratio >= DRIFT_THRESHOLD:
        _trigger_alert(summary)

    # Save JSON summary
    json_path = os.path.join(REPORTS_DIR, f"drift_summary_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON]   Summary   → {json_path}")

    return summary


def _trigger_alert(summary: dict) -> None:
    drifted = summary["drifted_columns"]
    ratio   = summary["drift_ratio"]

    msg = (
        f"\n{'='*60}\n"
        f"  ⚠  DATA DRIFT ALERT\n"
        f"{'='*60}\n"
        f"  Drift ratio:       {ratio:.0%}\n"
        f"  Drifted columns:   {', '.join(drifted)}\n"
        f"  Report:            {summary['report_path']}\n"
        f"  Recommendation:    Consider retraining the model.\n"
        f"{'='*60}"
    )
    print(msg)

    # ── Hook: integrate your alerting here ────────────────────────────────────
    # Slack webhook example:
    # import requests
    # requests.post(os.getenv("SLACK_WEBHOOK_URL"), json={"text": msg})

    # PagerDuty / email hooks can go here too


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = run_monitoring()
    print("\n[DONE] Monitoring complete.")
    if not summary["alert_triggered"]:
        print("[OK]   No significant drift detected.")
