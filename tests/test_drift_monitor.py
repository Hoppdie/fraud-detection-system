import numpy as np
import pytest
from drift_monitor import compute_psi, run_drift_check, DriftAlert


def stable():
    return np.random.default_rng(42).beta(2, 5, 500)

def drifted():
    return np.random.default_rng(99).beta(5, 2, 500)


def test_psi_stable_low():
    psi = compute_psi(stable(), stable() + np.random.default_rng(7).normal(0, 0.01, 500))
    assert psi < 0.10

def test_psi_drifted_high():
    assert compute_psi(stable(), drifted()) > 0.25

def test_alert_critical():
    assert run_drift_check(stable(), drifted()).status == "critical"

def test_alert_ok():
    ref = stable()
    assert run_drift_check(ref, ref + np.random.default_rng(0).normal(0, 0.005, 500)).status == "ok"

def test_alert_dict_keys():
    d = DriftAlert(score_psi=0.05).to_dict()
    assert set(d.keys()) == {"status", "score_psi", "feature_psi", "drifted_features"}
