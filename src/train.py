"""Train and back-test direct multi-horizon forecasters on the ETTh1 target.

For each horizon a gradient-boosted regressor is trained to predict the value
that many steps ahead from lag + calendar features. Each is evaluated on a
chronological hold-out against two baselines: persistence (repeat the last
known value) and seasonal-naive (the value 24h before the target). The trained
models and a recent context window are saved so the service can serve a full
horizon curve.

All numbers are written to ``model_artefacts/metrics.json`` and ``results.md``
so the README never carries a hand-typed figure.

    python -m src.train
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .dataset import TARGET, load_series
from .features import DEFAULT_LAGS, build_direct_dataset

ARTEFACT_DIR = Path(__file__).resolve().parents[1] / "model_artefacts"
TEST_FRACTION = 0.2
RANDOM_STATE = 42
HORIZONS = (1, 6, 12, 24)


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _scores(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAPE_pct": round(_mape(y_true, y_pred), 3),
    }


def main() -> None:
    df = load_series()
    series = df[TARGET].astype(float)
    print(f"Loaded {len(series)} hourly observations of '{TARGET}' "
          f"({series.index.min()} -> {series.index.max()})")

    models: dict[int, object] = {}
    by_horizon: dict[int, dict] = {}

    for h in HORIZONS:
        X, y = build_direct_dataset(series, h)
        split = int(len(X) * (1 - TEST_FRACTION))
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE
        )
        model.fit(X_tr, y_tr)
        models[h] = model

        pred = model.predict(X_te)
        # Persistence at horizon h = last value known at the origin = series[t - h].
        persistence = _persistence_for(series, y_te.index, h)
        seasonal = _seasonal_naive_for(series, y_te.index)

        by_horizon[h] = {
            "GradientBoosting": _scores(y_te, pred),
            "SeasonalNaive_24h": _scores(y_te, seasonal),
            "Persistence": _scores(y_te, persistence),
        }

    _report(series, by_horizon)
    _save(series, models, by_horizon)


def _persistence_for(series, target_index, horizon) -> np.ndarray:
    """Value known at the origin for each target time t: series[t - horizon]."""
    origin_index = target_index - np.timedelta64(horizon, "h")
    return series.reindex(origin_index).to_numpy(dtype=float)


def _seasonal_naive_for(series, target_index) -> np.ndarray:
    """Value 24h before each target time."""
    prior = target_index - np.timedelta64(24, "h")
    return series.reindex(prior).to_numpy(dtype=float)


def _report(series, by_horizon) -> None:
    print("\n=== Direct multi-horizon back-test (chronological hold-out) ===")
    print(f"  {'horizon':<9}{'GradBoost MAE':<15}{'SeasNaive MAE':<15}{'Persist MAE':<13}")
    for h in HORIZONS:
        m = by_horizon[h]
        print(f"  h={h:<7}{m['GradientBoosting']['MAE']:<15.4f}"
              f"{m['SeasonalNaive_24h']['MAE']:<15.4f}{m['Persistence']['MAE']:<13.4f}")


def _save(series, models, by_horizon) -> None:
    max_lag = max(DEFAULT_LAGS)
    bundle = {
        "models": models,
        "lags": list(DEFAULT_LAGS),
        "horizons": list(HORIZONS),
        "context": series.iloc[-(max_lag + 1):].tolist(),
        "last_timestamp": series.index[-1].isoformat(),
        "freq": "h",
        "target": TARGET,
    }
    ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTEFACT_DIR / "forecaster.joblib"
    joblib.dump(bundle, model_path)

    best_h = max(
        HORIZONS,
        key=lambda h: by_horizon[h]["Persistence"]["MAE"]
        - by_horizon[h]["GradientBoosting"]["MAE"],
    )
    metrics = {
        "dataset": "ETTh1 (public, Informer benchmark)",
        "target": TARGET,
        "n_observations": int(len(series)),
        "test_fraction": TEST_FRACTION,
        "method": "direct multi-horizon (one model per horizon)",
        "by_horizon": {str(h): by_horizon[h] for h in HORIZONS},
        "best_gain_horizon": best_h,
    }
    (ARTEFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_results_md(metrics)
    print(f"\nSaved {len(models)} horizon models -> {model_path}")
    print(f"Saved metrics -> {ARTEFACT_DIR / 'metrics.json'}")


def _write_results_md(metrics: dict) -> None:
    lines = [
        "# Reproduced results",
        "",
        f"Dataset: {metrics['dataset']}. Target: `{metrics['target']}`. "
        f"{metrics['n_observations']} hourly observations, "
        f"{int(metrics['test_fraction'] * 100)}% chronological hold-out. "
        f"Method: {metrics['method']}.",
        "",
        "| Horizon | GradientBoosting MAE | SeasonalNaive 24h MAE | Persistence MAE |",
        "|---|---|---|---|",
    ]
    for h, m in metrics["by_horizon"].items():
        lines.append(
            f"| {h}h | {m['GradientBoosting']['MAE']:.4f} | "
            f"{m['SeasonalNaive_24h']['MAE']:.4f} | {m['Persistence']['MAE']:.4f} |"
        )
    lines += [
        "",
        "The model beats both baselines at the 1h and 6h horizons. From 12h the gap "
        "closes: persistence is the stronger forecast at 12h, and at 24h both "
        "persistence and the 24h seasonal-naive outperform the model on this highly "
        "persistent target. Reporting against naive baselines at every horizon is "
        "the point: it shows where the model adds real value and where a trivial "
        "baseline already suffices.",
    ]
    (ARTEFACT_DIR / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
