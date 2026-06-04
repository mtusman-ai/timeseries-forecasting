"""Pydantic request/response models for the forecasting service."""
from __future__ import annotations

from pydantic import BaseModel


class HorizonPoint(BaseModel):
    horizon_hours: int
    target_timestamp: str
    predicted_value: float


class ForecastResponse(BaseModel):
    target: str
    origin_timestamp: str
    points: list[HorizonPoint]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    target: str
    horizons: list[int]


class BacktestResponse(BaseModel):
    dataset: str
    target: str
    method: str
    by_horizon: dict
