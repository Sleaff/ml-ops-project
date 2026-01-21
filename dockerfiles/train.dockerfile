# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc curl gnupg && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

COPY uv.lock uv.lock
COPY LICENSE LICENSE
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

WORKDIR /
ENV UV_LINK_MODE=copy
RUN uv sync
RUN chmod +x scripts/train_cloud.sh

ENTRYPOINT ["scripts/train_cloud.sh"]
