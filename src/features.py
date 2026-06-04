"""Feature construction for direct multi-horizon forecasting.

To predict the value ``h`` steps after an origin ``t`` the features are:

* ``cur`` — the value observed at the origin, ``y[t]`` (the strongest predictor);
* ``lag_k`` — earlier values ``y[t-k]`` for the configured lags;
* calendar features of the *target* time ``t + h`` (deterministic, known ahead).

Training one model per horizon ("direct" forecasting) avoids the error
accumulation of rolling a one-step model forward. The exact same column set and
order are produced at train time (``build_direct_dataset``) and at serve time
(``feature_row``), so there is no train/serve skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Lags (hours): recent (1-3h), daily (24/48h), weekly (168h) seasonality.
DEFAULT_LAGS = (1, 2, 3, 24, 48, 168)
_CAL_COLS = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos")


def feature_columns(lags: tuple[int, ...] = DEFAULT_LAGS) -> list[str]:
    return ["cur"] + [f"lag_{k}" for k in lags] + list(_CAL_COLS)


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * index.hour / 24),
            "dow_sin": np.sin(2 * np.pi * index.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * index.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * (index.month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (index.month - 1) / 12),
        },
        index=index,
    )


def build_direct_dataset(
    series: pd.Series, horizon: int, lags: tuple[int, ...] = DEFAULT_LAGS
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for predicting ``horizon`` steps ahead from origin ``t``."""
    cols = {"cur": series}
    for k in lags:
        cols[f"lag_{k}"] = series.shift(k)
    X = pd.DataFrame(cols, index=series.index)

    future_index = series.index + pd.Timedelta(hours=horizon)
    cal = calendar(future_index)
    cal.index = series.index
    X = X.join(cal)[feature_columns(lags)]

    y = series.shift(-horizon)
    valid = X.dropna().index.intersection(y.dropna().index)
    return X.loc[valid], y.loc[valid]


def feature_row(
    history: list[float], target_timestamp: pd.Timestamp,
    lags: tuple[int, ...] = DEFAULT_LAGS
) -> pd.DataFrame:
    """One feature row from recent history (most recent value last). The origin
    is the last value; ``cur`` is ``history[-1]`` and ``lag_k`` is
    ``history[-1-k]``. Needs at least ``max(lags) + 1`` values."""
    row = {"cur": history[-1]}
    for k in lags:
        row[f"lag_{k}"] = history[-1 - k]
    cal = calendar(pd.DatetimeIndex([target_timestamp])).iloc[0].to_dict()
    row.update(cal)
    return pd.DataFrame([row])[feature_columns(lags)]
