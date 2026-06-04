"""Direct multi-horizon forecasting, shared by the back-test and the service.

A separate model is trained per horizon. Given recent history and the origin
timestamp, ``direct_forecast`` returns one prediction per trained horizon.
"""
from __future__ import annotations

import pandas as pd

from .features import DEFAULT_LAGS, feature_row


def direct_forecast(
    models: dict[int, object],
    history: list[float],
    last_timestamp: pd.Timestamp,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    freq: str = "h",
) -> dict[int, float]:
    """Predict each trained horizon from history (most recent value last)."""
    ts = pd.Timestamp(last_timestamp)
    step = pd.tseries.frequencies.to_offset(freq)
    out: dict[int, float] = {}
    for h in sorted(models):
        target_ts = ts + step * h
        x = feature_row(history, target_ts, lags)
        out[h] = round(float(models[h].predict(x)[0]), 4)
    return out
