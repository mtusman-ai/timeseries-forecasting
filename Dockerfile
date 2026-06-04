# Multi-stage build. The builder installs dependencies, downloads the public
# ETT dataset and trains the per-horizon models (so the image serves forecasts
# out of the box); the runtime stage copies only the installed packages, the
# application code and the trained artefact. Build from the repo root:
#   docker build -t timeseries-forecasting .
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

COPY src ./src
COPY data ./data
COPY app ./app
COPY tests ./tests

ENV PYTHONPATH=/build:/install/lib/python3.11/site-packages \
    PATH=/install/bin:$PATH
RUN python data/download.py && python -m src.train


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY --from=builder /install /usr/local
COPY --from=builder /build/src ./src
COPY --from=builder /build/app ./app
COPY --from=builder /build/model_artefacts ./model_artefacts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
