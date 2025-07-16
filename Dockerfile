# Используем официальный образ с uv
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
WORKDIR /app

# Устанавливаем только зависимости из lockfile
COPY uv.lock pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Добавляем код и устанавливаем весь проект (зависимости уже закешированы)
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Runtime-образ — тот же базовый, без uv
FROM python:3.10-slim-bookworm

# Устанавливаем системные зависимости для opencv
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Копируем venv из builder
COPY --from=builder /app/.venv /app/.venv
WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV GRADIO_SERVER_NAME="0.0.0.0"
EXPOSE 7860

ENTRYPOINT []
CMD ["uv", "run", "python3", "main.py"]
