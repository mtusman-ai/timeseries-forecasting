"""Integration tests for the forecasting API (TestClient, no Docker)."""
from __future__ import annotations


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["target"] == "OT"
    assert 1 in body["horizons"]


def test_forecast_curve(client):
    resp = client.get("/forecast")
    assert resp.status_code == 200
    body = resp.json()
    assert body["target"] == "OT"
    assert len(body["points"]) == 2  # horizons (1, 6) in the test bundle
    for p in body["points"]:
        assert isinstance(p["predicted_value"], float)
        assert p["horizon_hours"] in {1, 6}
        assert p["target_timestamp"] > body["origin_timestamp"]


def test_backtest(client):
    resp = client.get("/backtest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["target"] == "OT"
    assert "by_horizon" in body
