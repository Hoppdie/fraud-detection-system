"""
drift_monitor.py — Population Stability Index (PSI) drift detector
Compares a reference score distribution against a current batch to flag
model drift before it silently degrades precision/recall in production.

Usage (standalone):
    python drift_monitor.py --ref data/reference_scores.csv --cur data/batch_scores.csv

Or import in your inference loop:
    from drift_monitor import compute_psi, DriftAlert
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    buckets: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.10  → no significant shift
    PSI 0.10-0.25 → moderate shift, investigate
    PSI > 0.25  → major shift, retrain likely needed
    """
    breakpoints = np.linspace(0.0, 1.0, buckets + 1)
    ref_counts = np.histogram(reference, bins=breakpoints)[0] + eps
    cur_counts = np.histogram(current, bins=breakpoints)[0] + eps
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_feature_psi(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    features: List[str],
    buckets: int = 10,
) -> dict:
    return {
        feat: compute_psi(ref_df[feat].dropna().values, cur_df[feat].dropna().values, buckets)
        for feat in features
        if feat in ref_df.columns and feat in cur_df.columns
    }


@dataclass
class DriftAlert:
    score_psi: float
    feature_psi: dict = field(default_factory=dict)
    drifted_features: List[str] = field(default_factory=list)
    status: str = "ok"

    def __post_init__(self):
        if self.score_psi > 0.25:
            self.status = "critical"
        elif self.score_psi > 0.10:
            self.status = "warn"
        self.drifted_features = [f for f, v in self.feature_psi.items() if v > 0.10]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "score_psi": round(self.score_psi, 4),
            "feature_psi": {k: round(v, 4) for k, v in self.feature_psi.items()},
            "drifted_features": self.drifted_features,
        }


def run_drift_check(
    ref_scores: np.ndarray,
    cur_scores: np.ndarray,
    ref_df: Optional[pd.DataFrame] = None,
    cur_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> DriftAlert:
    score_psi = compute_psi(ref_scores, cur_scores)
    feat_psi = {}
    if ref_df is not None and cur_df is not None and feature_cols:
        feat_psi = compute_feature_psi(ref_df, cur_df, feature_cols)
    return DriftAlert(score_psi=score_psi, feature_psi=feat_psi)


# Optional FastAPI router — only available if fastapi is installed
try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    router = APIRouter(prefix="/drift", tags=["monitoring"])

    class DriftRequest(BaseModel):
        reference_scores: List[float]
        current_scores: List[float]
        buckets: int = 10

    @router.post("/psi", summary="Compute PSI between reference and current score distributions")
    def psi_endpoint(req: DriftRequest):
        if len(req.reference_scores) < 10 or len(req.current_scores) < 10:
            raise HTTPException(status_code=422, detail="Need at least 10 scores per distribution")
        psi = compute_psi(np.array(req.reference_scores), np.array(req.current_scores), req.buckets)
        status = "critical" if psi > 0.25 else "warn" if psi > 0.10 else "ok"
        return {"psi": round(psi, 4), "status": status, "buckets": req.buckets}

except ImportError:
    pass


def _cli():
    parser = argparse.ArgumentParser(description="Fraud model drift checker (PSI)")
    parser.add_argument("--ref", required=True, help="CSV with 'score' column (reference)")
    parser.add_argument("--cur", required=True, help="CSV with 'score' column (current batch)")
    parser.add_argument("--buckets", type=int, default=10)
    parser.add_argument("--features", nargs="*", default=[])
    args = parser.parse_args()
    ref_df = pd.read_csv(args.ref)
    cur_df = pd.read_csv(args.cur)
    alert = run_drift_check(
        ref_df["score"].values, cur_df["score"].values,
        ref_df if args.features else None,
        cur_df if args.features else None,
        args.features or None,
    )
    print(json.dumps(alert.to_dict(), indent=2))
    sys.exit(0 if alert.status == "ok" else 1)


if __name__ == "__main__":
    _cli()
