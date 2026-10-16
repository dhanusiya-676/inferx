# Build stage for InferX Serving Platform
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY pyproject.toml setup.py README.md ./
COPY inferx/ ./inferx/
COPY tests/ ./tests/

# Install dependencies and inferx package
RUN pip install --no-cache-dir -e .

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/v1/health || exit 1

# Default launch command: FastAPI Gateway
CMD ["inferx", "serve", "--host", "0.0.0.0", "--port", "8000"]
