import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "project"
PYTHON_VERSION = "3.12"

# GCP Configuration
GCP_PROJECT = "mlops-483515"
GCP_ZONE = "europe-west1-b"
GCP_VM_NAME = "mlops-training-vm"
GCP_IMAGE_NAME = f"gcr.io/{GCP_PROJECT}/train:latest"
GCS_BUCKET = "gs://sleaff_mlops_data_bucket"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context, lr=2e-5, epochs=10) -> None:
    """Train model."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/train.py --lr {lr}, --e {epochs}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run(
        "uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build",
        echo=True,
        pty=not WINDOWS,
    )


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# GCP commands
@task
def gcloud_create_vm(ctx: Context) -> None:
    """Create GCP VM for training."""
    ctx.run(
        f"gcloud compute instances create {GCP_VM_NAME} "
        f"--zone={GCP_ZONE} "
        f"--machine-type=e2-medium "
        f"--image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 "
        f"--image-project=deeplearning-platform-release "
        f"--boot-disk-size=100GB "
        f"--scopes=cloud-platform "
        f"--project={GCP_PROJECT}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def gcloud_build_push(ctx: Context) -> None:
    """Build and push training Docker image using Cloud Build (no local Docker needed)."""
    ctx.run(
        f"gcloud builds submit --config=cloudbuild.yaml --project={GCP_PROJECT} .",
        echo=True,
        pty=not WINDOWS,
    )


@task
def gcloud_train(ctx: Context, wandb_key: str = "") -> None:
    """Run training on GCP VM using Docker.

    Args:
        wandb_key: W&B API key. If not provided, uses WANDB_API_KEY env var.
    """
    wandb_api_key = wandb_key or os.environ.get("WANDB_API_KEY", "")
    env_flags = f"-e WANDB_API_KEY={wandb_api_key}" if wandb_api_key else ""

    # Use --network=host to access GCP metadata server for GCS auth
    docker_cmd = (
        f"gcloud auth configure-docker gcr.io --quiet && "
        f"docker pull {GCP_IMAGE_NAME} && "
        f"docker run --rm --network=host {env_flags} {GCP_IMAGE_NAME}"
    )
    ctx.run(
        f"gcloud compute ssh {GCP_VM_NAME} --zone={GCP_ZONE} --project={GCP_PROJECT} "
        f'--command="{docker_cmd}"',
        echo=True,
        pty=not WINDOWS,
    )


@task
def gcloud_stop_vm(ctx: Context) -> None:
    """Stop GCP VM to save costs."""
    ctx.run(
        f"gcloud compute instances stop {GCP_VM_NAME} --zone={GCP_ZONE} --project={GCP_PROJECT}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def gcloud_upload_data(ctx: Context) -> None:
    """Upload local data to GCS bucket."""
    ctx.run(f"gsutil -m cp data/raw/* {GCS_BUCKET}/data/raw/", echo=True, pty=not WINDOWS)
    ctx.run(f"gsutil -m cp data/processed/* {GCS_BUCKET}/data/processed/", echo=True, pty=not WINDOWS)


@task
def gcloud_download_models(ctx: Context) -> None:
    """Download trained models from GCS bucket."""
    ctx.run("mkdir -p models", echo=True, pty=not WINDOWS)
    ctx.run(f"gsutil -m cp -r {GCS_BUCKET}/src/project/models/* models/", echo=True, pty=not WINDOWS)
