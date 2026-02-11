# ── Stage 1: Build frontend ─────────────────────────────────────────────
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python application ────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     ffmpeg libsm6 libxext6 \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY . /app

# Copy built frontend from stage 1
COPY --from=frontend /frontend/dist /app/frontend/dist

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []
EXPOSE 7860

CMD ["python3", "main.py"]
