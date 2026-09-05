# Multi-stage image for local PSOD development and smoke testing.

FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml README.md LICENSE PROVENANCE.md ./
COPY src ./src

RUN python -m pip install --upgrade "pip>=26.2" "setuptools>=83" wheel && \
    python -m pip install .

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/DiogoRibeiro7/PSOD"
LABEL org.opencontainers.image.description="PSOD - pseudo-supervised outlier detection"
LABEL org.opencontainers.image.licenses="GPL-3.0-only"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN useradd -m -u 1000 psod
USER psod
WORKDIR /home/psod

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import psod; print('OK')" || exit 1

CMD ["python"]
