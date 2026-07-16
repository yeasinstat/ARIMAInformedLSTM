# ─── ARIMA-Informed LSTM — Dockerfile for Hugging Face Spaces (Docker SDK) ───
FROM python:3.10-slim

# pmdarima and some scientific packages need a C compiler to build from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as a non-root user (UID 1000) by convention
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies first (better layer caching on rebuilds)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY --chown=user . .

# Hugging Face Spaces always expect the app to listen on port 7860
EXPOSE 7860

CMD ["python", "app.py"]