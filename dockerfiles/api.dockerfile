# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY LICENSE LICENSE
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY models/* src/project/models/
COPY models/* models/

COPY default.json default.json
RUN uv run dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc .dvc/
RUN uv run dvc remote modify remote_storage credentialpath default.json
RUN uv run dvc config core.no_scm true
RUN uv run dvc pull

WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

CMD ["uv", "run", "uvicorn", "src.project.api:app", "--host", "0.0.0.0", "--port", "8000"]