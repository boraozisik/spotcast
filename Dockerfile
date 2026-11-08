# SpotCast API image.
#
#   docker compose up --build        # API + database (see compose.yaml)
#
# Two stages: the first installs dependencies with uv, the second copies only the result.
# Build tools and caches stay in the first stage, so the final image is smaller.

# ---------- Stage 1: install dependencies ----------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first, on their own layer: they change rarely, so Docker can reuse this
# layer when only our code changes. --no-default-groups: only what the API needs
# (no torch, xgboost, mlflow, langgraph, mcp, pillow).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-install-project

# Then our own code
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups

# ---------- Stage 2: the image that runs ----------
FROM python:3.12-slim

# Never run as root: if someone broke into the app, they would not own the container
RUN useradd --create-home --uid 1000 spotcast
WORKDIR /app
COPY --from=builder --chown=spotcast:spotcast /app /app
USER spotcast

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# The model file is not baked in: it is mounted at /app/models (see compose.yaml),
# so a retrained model doesn't need a new image.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "spotcast.api:app", "--host", "0.0.0.0", "--port", "8000"]
