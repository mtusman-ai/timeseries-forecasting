"""Test fixtures.

CI has no downloaded dataset and no trained artefact, so we train a tiny bundle
on the bundled sample fixture and point the service at it. This exercises the
real load -> forecast path while staying hermetic and fast.
"""
from __future__ import annotations

import json

import joblib
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from src.dataset import TARGET, load_series
from src.features import DEFAULT_LAGS, build_direct_dataset

_HORIZONS = (1, 6)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    series = load_series().asfreq("h")[TARGET].astype(float)
    models = {}
    for h in _HORIZONS:
        X, y = build_direct_dataset(series, h)
        models[h] = HistGradientBoostingRegressor(max_iter=50, random_state=0).fit(X, y)

    max_lag = max(DEFAULT_LAGS)
    bundle = {
        "models": models,
        "lags": list(DEFAULT_LAGS),
        "horizons": list(_HORIZONS),
        "context": series.iloc[-(max_lag + 1):].tolist(),
        "last_timestamp": series.index[-1].isoformat(),
        "freq": "h",
        "target": TARGET,
    }
    bundle_path = tmp_path / "forecaster.joblib"
    joblib.dump(bundle, bundle_path)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({
        "dataset": "ETTh1 sample (test)", "target": TARGET,
        "method": "direct multi-horizon",
        "by_horizon": {"1": {"GradientBoosting": {"MAE": 0.0}}},
    }))

    monkeypatch.setenv("FORECASTER_PATH", str(bundle_path))
    monkeypatch.setenv("METRICS_PATH", str(metrics_path))

    from app import model as model_mod

    model_mod.get_bundle.cache_clear()
    model_mod.get_metrics.cache_clear()

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
