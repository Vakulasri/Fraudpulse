# Container for the scoring API only. Train the model first so models/ contains
# xgb_model.json + model_meta.json, then:
#   docker build -t fraudpulse-api .
#   docker run -p 8000:8000 fraudpulse-api
FROM python:3.12-slim

# libgomp1 = OpenMP runtime that XGBoost needs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY models/ ./models/

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
