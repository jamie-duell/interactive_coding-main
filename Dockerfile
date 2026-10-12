# syntax=docker/dockerfile:1
# ---------- 1. build the React frontend ----------
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Prepare CIFAR-10 once at build time; evaluation containers never need the network.
FROM python:3.12-slim AS cifar10-data
RUN pip install --no-cache-dir "numpy>=1.26"
COPY backend/app/scripts/build_cifar10.py /build_cifar10.py
RUN python /build_cifar10.py --out /cifar10_test.npz

# MedMNIST is needed in Colab, but the server only needs this verified test archive.
FROM python:3.12-slim AS pneumoniamnist-data
RUN pip install --no-cache-dir "numpy>=1.26"
COPY backend/app/scripts/build_pneumoniamnist.py /build_pneumoniamnist.py
RUN python /build_pneumoniamnist.py --out /pneumoniamnist_test.npz

# ---------- 2. backend image (serves API + built frontend; also used by the worker) ----------
FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Docker CLI so the worker can start sandbox containers through the mounted socket.
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

# Install dependencies first (cached unless pyproject.toml changes).
COPY backend/pyproject.toml ./
RUN python -c "import tomllib;d=tomllib.load(open('pyproject.toml','rb'));print('\n'.join(d['project']['dependencies']+d['project']['optional-dependencies']['dev']))" > /tmp/req.txt \
 && pip install -r /tmp/req.txt

COPY backend/ /app/
COPY --from=cifar10-data /cifar10_test.npz /opt/arena/cifar10_test.npz
COPY --from=pneumoniamnist-data /pneumoniamnist_test.npz /opt/arena/pneumoniamnist_test.npz
COPY --from=frontend /fe/dist /app/static
COPY examples/ /app/static/examples/
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh && mkdir -p /data

ENV DATA_DIR=/data STATIC_DIR=/app/static CIFAR10_ARCHIVE=/opt/arena/cifar10_test.npz
ENV PNEUMONIAMNIST_ARCHIVE=/opt/arena/pneumoniamnist_test.npz
EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["web"]
