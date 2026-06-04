"""FastAPI service: multi-horizon load/temperature forecasting.

Routes:
  GET /health    liveness + whether the artefact loaded
  GET /forecast  horizon curve (t+1 .. t+H) from the latest known origin
  GET /backtest  the held-out back-test metrics by horizon

Interactive docs at /docs once running.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .model import forecast, get_bundle, get_metrics
from .schemas import BacktestResponse, ForecastResponse, HealthResponse

app = FastAPI(
    title="timeseries-forecasting",
    version="1.0.0",
    description="Direct multi-horizon forecasting on the public ETT electricity "
    "dataset, served behind an HTTP API.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        b = get_bundle()
        return HealthResponse(status="ok", model_loaded=True, target=b["target"],
                              horizons=list(b["horizons"]))
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False, target="none",
                              horizons=[])


@app.get("/forecast", response_model=ForecastResponse)
def forecast_route() -> ForecastResponse:
    try:
        return ForecastResponse(**forecast())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/backtest", response_model=BacktestResponse)
def backtest_route() -> BacktestResponse:
    metrics = get_metrics()
    if not metrics:
        raise HTTPException(status_code=503, detail="No metrics; run python -m src.train")
    return BacktestResponse(
        dataset=metrics["dataset"], target=metrics["target"],
        method=metrics["method"], by_horizon=metrics["by_horizon"],
    )
