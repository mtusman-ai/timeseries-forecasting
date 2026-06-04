"""Artefact loading for the forecasting service, kept out of the routes.

The artefact bundle (one model per horizon, plus a recent context window and the
origin timestamp) is produced by ``src/train.py``. Its path is configurable so
tests can point at a throwaway bundle and the container can mount a volume.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_BUNDLE = _REPO / "model_artefacts" / "forecaster.joblib"
_DEFAULT_METRICS = _REPO / "model_artefacts" / "metrics.json"


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    path = Path(os.environ.get("FORECASTER_PATH", _DEFAULT_BUNDLE))
    if not path.exists():
        raise FileNotFoundError(
            f"No forecaster at {path}. Train one first: python -m src.train"
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_metrics() -> dict | None:
    path = Path(os.environ.get("METRICS_PATH", _DEFAULT_METRICS))
    return json.loads(path.read_text()) if path.exists() else None


def forecast() -> dict:
    """Return the horizon curve from the latest known origin."""
    from src.forecast import direct_forecast

    bundle = get_bundle()
    lags = tuple(bundle["lags"])
    preds = direct_forecast(
        bundle["models"], bundle["context"], bundle["last_timestamp"], lags,
        bundle.get("freq", "h"),
    )
    origin = pd.Timestamp(bundle["last_timestamp"])
    step = pd.tseries.frequencies.to_offset(bundle.get("freq", "h"))
    points = [
        {
            "horizon_hours": h,
            "target_timestamp": (origin + step * h).isoformat(),
            "predicted_value": v,
        }
        for h, v in preds.items()
    ]
    return {"target": bundle["target"], "origin_timestamp": origin.isoformat(),
            "points": points}
