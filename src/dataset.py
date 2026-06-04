"""Load the public ETT (Electricity Transformer Temperature) dataset.

ETTh1 is an hourly series from the Informer paper, widely used as a forecasting
benchmark. It is public on GitHub (no auth). The forecasting target is ``OT``
(transformer oil temperature); the other columns are load measurements that can
serve as exogenous features.

The loader falls back to a tiny bundled sample (``tests/fixtures/ett_sample.csv``)
when the full file has not been downloaded, so imports never hard-fail.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
RAW = _REPO / "data" / "raw" / "ETTh1.csv"
SAMPLE = _REPO / "tests" / "fixtures" / "ett_sample.csv"
TARGET = "OT"


def load_series(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Return the dataset indexed by hourly timestamp."""
    path = Path(csv_path) if csv_path else (RAW if RAW.exists() else SAMPLE)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df.asfreq("h")
    df = df.interpolate(limit_direction="both")
    return df
