# timeseries-forecasting

Multi-horizon time-series forecasting on a public electricity dataset, taken
from a benchmark through to a served HTTP API. The model predicts several hours
ahead, is back-tested against naive baselines at every horizon, and is served by
a containerised FastAPI app with CI.

A forecast is only useful if it beats the trivial baseline a stakeholder could
compute by hand, so every horizon is scored against persistence and a
seasonal-naive baseline.

> **Honesty note.** The table below is produced by `src/train.py` and written to
> `model_artefacts/`; this README quotes those reproduced numbers.

---

## Data and method

- **Dataset**: ETTh1 (Electricity Transformer Temperature), a public hourly
  series from the Informer paper, widely used as a forecasting benchmark.
  17,420 hourly observations. Downloaded by `data/download.py` (no account
  needed); nothing is committed.
- **Target**: `OT` (transformer oil temperature). The load columns are available
  as exogenous features.
- **Method**: direct multi-horizon forecasting. A separate gradient-boosted
  regressor is trained per horizon to predict that many steps ahead from the
  current value, lagged values (1, 2, 3, 24, 48, 168 hours) and calendar
  features of the target time. Training one model per horizon avoids the error
  accumulation of rolling a one-step model forward.

### Back-test (20% chronological hold-out, MAE)

| Horizon | GradientBoosting | Seasonal-naive (24h) | Persistence |
|---|---|---|---|
| 1h | **0.469** | 1.763 | 0.649 |
| 6h | **1.415** | 2.047 | 1.630 |
| 12h | 2.005 | 2.288 | **1.715** |
| 24h | 2.741 | **2.362** | 2.362 |

The model beats both baselines at the 1h and 6h horizons. From 12h the gap
closes: persistence is the stronger forecast at 12h (MAE 1.715 against the
model's 2.005), and at 24h both persistence and the 24h seasonal-naive
outperform the model on this highly persistent target. Reporting against naive
baselines at every horizon is the point: it shows where the model adds real
value and where a trivial baseline already suffices. That judgement, knowing
when not to deploy a model, matters as much in production as the headline
metric.

Regenerate:

```bash
pip install -r requirements.txt
python data/download.py
python -m src.train     # trains per-horizon models, writes metrics.json + results.md
```

---

## Service

### Quickstart

```bash
git clone https://github.com/mtusman-ai/timeseries-forecasting
cd timeseries-forecasting
docker compose up --build
```

The image downloads the dataset and trains the models during the build, so the
service is ready when the container is up.

```bash
curl http://localhost:8000/forecast
# {"target":"OT","origin_timestamp":"...","points":[
#   {"horizon_hours":1,"target_timestamp":"...","predicted_value":9.74}, ... ]}

curl http://localhost:8000/backtest    # held-out MAE by horizon
```

Interactive API docs at `http://localhost:8000/docs`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness, target, and the trained horizons. |
| GET | `/forecast` | Forecast curve (t+1 .. t+24) from the latest known origin. |
| GET | `/backtest` | Held-out back-test metrics by horizon. |

### Engineering notes

- The trained models plus a recent context window are saved as one artefact, so
  the service produces a forecast without re-reading the dataset.
- Multi-stage Docker build; runtime image carries only installed packages,
  application code, and the artefact. Measured image size: ~715 MB.
- CI (`.github/workflows/ci.yml`): `ruff` lint, an integration test suite that
  trains a tiny bundle so CI needs no dataset, and a `docker build`.

---

## Repository layout

```
timeseries-forecasting/
  src/      dataset loader, feature builder, direct-forecast, training + back-test
  data/     download.py (public ETT); raw data is git-ignored
  app/      FastAPI service (health, forecast, backtest)
  tests/    integration tests + a small sample fixture
  .github/workflows/ci.yml
  requirements.txt  pyproject.toml  LICENSE  README.md
```

## Limitations and next steps

- A single target series with a single model family. Exogenous load covariates,
  probabilistic intervals, and a stronger sequence model are the natural
  extensions; the back-test harness makes adding them measurable.
- Forecasts assume the recent regime holds. A real deployment would monitor for
  drift and re-train on a schedule.

## Licence

MIT for the code (`LICENSE`). The ETT dataset is public under its own terms.

## Contact

M Usman — mt_usman@outlook.com · https://github.com/mtusman-ai
