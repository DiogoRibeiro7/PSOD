# Production Dockerfile for PSOD
# =================================
# Multi-stage build for minimal final image

# Build stage
FROM python:3.11-slim as builder

LABEL maintainer="Diogo Ribeiro <dfr@esmad.ipp.pt>"
LABEL description="PSOD - Pseudo-Supervised Outlier Detection"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first for better caching
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

# Copy source code
WORKDIR /app
COPY . /app/

# Install PSOD
RUN pip install --no-cache-dir .

# ================================
# Runtime stage
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/diogoribeiro7/PSOD"
LABEL org.opencontainers.image.description="PSOD - Pseudo-Supervised Outlier Detection"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd -m -u 1000 psod && \
    mkdir -p /app /data /models && \
    chown -R psod:psod /app /data /models

USER psod
WORKDIR /app

# Copy application code
COPY --chown=psod:psod . /app/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import psod; print('OK')" || exit 1

# Default command
CMD ["python"]

