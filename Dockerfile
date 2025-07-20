FROM cgr.dev/chainguard/python:latest-dev AS builder

USER root

# https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1 
# https://docs.astral.sh/uv/guides/integration/docker/#caching
ENV UV_LINK_MODE=copy
# https://docs.astral.sh/uv/concepts/cache/#cache-directory
# Found this cache dir by running;
# docker run -it --entrypoint /bin/bash --rm cgr.dev/chainguard/python:latest-dev
# uv cache dir
ENV UV_CACHE_DIR=/root/.cache/uv
ENV GRADIO_SERVER_NAME="0.0.0.0"

WORKDIR /app

# Install dependencies
# Mount the cache and lock files

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     ffmpeg libsm6 libxext6 \
  && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-editable

# Copy source code
COPY . app/

# Install the application
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable


FROM cgr.dev/chainguard/python:latest AS runtime

WORKDIR /app

COPY --from=builder /app /app

EXPOSE 7860

CMD ["uv", "run", "main.py"]