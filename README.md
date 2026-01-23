# Fake News Classifier

An MLOps project for binary classification of news articles as real or fake, built as part of the DTU course 02476 Machine Learning Operations.

## Team

| Student ID | Name |
|------------|------|
| s195171 | Kenneth Plum Toft |
| s242726 | Filip Arthur Blaafjell |
| s243586 | Joakim Dinh |
| s242723 | Vebjørn Sæten Skre |

## Project Overview

This project classifies news articles as **fake** or **real** using a fine-tuned DistilBERT model. The focus is on learning MLOps tools and practices: version control, containerization, cloud training, CI/CD, experiment tracking, and monitoring.

### Dataset

[Kaggle: Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

- ~44,000 news articles (half fake, half real)
- Text-based classification task
- Stored in GCS bucket, versioned with DVC

### Model

We use **DistilBERT** (`distilbert-base-uncased`) from Hugging Face Transformers:
- Encoder is frozen, only the classifier head is trained
- Binary classification with BCEWithLogitsLoss
- Achieves ~96% validation accuracy after fine-tuning

---

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/Sleaff/ml-ops-project.git
cd ml-ops-project

# Install dependencies (requires uv)
uv sync

# Pull data from GCS
uv run dvc pull
```

### Local Training

```bash
uv run python src/project/train.py
```

Override hyperparameters with Hydra:
```bash
uv run python src/project/train.py model.lr=1e-4 training.batch_size=16
```

### Run Tests

```bash
uv run pytest tests/
uv run coverage run -m pytest tests/ && uv run coverage report
```

---

## Project Structure

```
src/project/
├── data.py              # Data downloading & preprocessing
├── dataset.py           # PyTorch Dataset wrapper
├── model.py             # DistilBERT + classifier model
├── train.py             # Training loop (PyTorch Lightning)
├── api.py               # FastAPI inference API
├── frontend.py          # Gradio web interface
└── drift_detection.py   # Evidently data drift detection

configs/                 # Hydra configuration files
dockerfiles/             # Docker images for training and API
tests/                   # Unit and integration tests
.github/workflows/       # CI/CD pipelines
```

---

## Cloud Training on Google Cloud Platform

Training runs on a GPU VM in GCP. Everything is pre-configured - just run the commands.

### Quick Start (for teammates)

```bash
# 1. Authenticate (one-time)
gcloud auth login
gcloud config set project mlops-483515

# 2. Run training (with GPU + W&B logging)
uv run invoke gcloud-train --gpu

# 3. Stop VM when done (saves money!)
uv run invoke gcloud-stop-vm
```

### What's Already Set Up

| Component | Details |
|-----------|---------|
| **GCP Project** | `mlops-483515` |
| **VM** | `mlops-training-vm` in `us-west1-b` (T4 GPU, SPOT instance) |
| **Docker Image** | `gcr.io/mlops-483515/train:latest` |
| **Data** | `gs://sleaff_mlops_data_bucket/data/` |
| **Models** | `gs://sleaff_mlops_data_bucket/src/project/models/` |
| **W&B Project** | `news-classification` |

### Training Commands

```bash
# GPU training with W&B logging (recommended)
uv run invoke gcloud-train --gpu

# GPU training without W&B
uv run invoke gcloud-train --gpu --no-wandb

# CPU-only (slow, for testing)
uv run invoke gcloud-train
```

### What Happens During Training

1. SSH into VM
2. Pull Docker image from GCR
3. Download data from GCS bucket
4. Run training with PyTorch Lightning
5. Upload model checkpoints to GCS
6. Logs stream to your terminal + W&B

### Other Commands

```bash
# Download trained models locally
uv run invoke gcloud-download-models

# Check VM status
gcloud compute instances list --project=mlops-483515

# SSH into VM manually
gcloud compute ssh mlops-training-vm --zone=us-west1-b --project=mlops-483515

# View GPU usage (while training)
gcloud compute ssh mlops-training-vm --zone=us-west1-b --project=mlops-483515 --command="nvidia-smi"
```

### Cost Management

- **VM is SPOT instance** - cheaper but can be preempted
- **Always stop VM when done**: `uv run invoke gcloud-stop-vm`
- Cost: ~$0.40/hour when running (T4 GPU)

### Rebuilding Docker Image (after code changes)

Only needed if you change training code:
```bash
uv run invoke gcloud-build-push  # ~15-20 min
```

### Troubleshooting

**VM not running?**
```bash
gcloud compute instances start mlops-training-vm --zone=us-west1-b --project=mlops-483515
```

**Docker not found on VM?**
```bash
gcloud compute ssh mlops-training-vm --zone=us-west1-b --project=mlops-483515
sudo apt-get update && sudo apt-get install -y docker.io nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Permission denied?** Make sure you're added as Owner on the GCP project `mlops-483515`.

---

## Running the API & Frontend

### Quick Start

```bash
# 1. Download model from GCS
gcloud storage cp gs://sleaff_mlops_data_bucket/models/best_model_epoch=09_val_loss=0.1466.ckpt models/

# 2. Start API (Terminal 1)
uv run uvicorn src.project.api:app --reload --port 8000

# 3. Start Gradio frontend (Terminal 2)
uv run python src/project/frontend.py
```

Open http://localhost:8000 in your browser.

### Docker (no GCP auth needed)

```bash
docker build -f dockerfiles/api.dockerfile -t news-api .
docker run -p 8000:8000 news-api
```

---

## API Monitoring & Alerts

### Prometheus Metrics

The API exposes metrics at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `api_requests_total` | Counter | Total requests by endpoint |
| `api_request_latency_seconds` | Histogram | Request latency |
| `api_predictions_total` | Counter | Predictions by class |

```bash
curl http://localhost:8000/metrics
```

### GCP Cloud Monitoring

When deployed to Cloud Run, go to **GCP Console** → **Monitoring** → **Metrics Explorer** to view:
- Request count, latency, error rate
- Container CPU/memory usage

### Alerts Setup

1. **GCP Console** → **Monitoring** → **Alerting** → **Create Policy**
2. Add condition (e.g., latency > 5s, error rate > 10%)
3. Add notification channel (email, Slack)
4. Save

---

## Data Drift Detection

Uses Evidently AI to detect data drift between training and production data.

```bash
# Generate drift report
uv run python src/project/drift_detection.py

# Open report
open reports/drift_report.html
```

The HTML report shows:
- Dataset drift summary (drift detected yes/no)
- Per-column drift scores
- Distribution comparisons between reference and current data

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **ML Framework** | PyTorch, PyTorch Lightning, Hugging Face Transformers |
| **Experiment Tracking** | Weights & Biases |
| **Configuration** | Hydra |
| **Data Versioning** | DVC + Google Cloud Storage |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Cloud** | Google Cloud Platform (Compute Engine, Cloud Storage, Container Registry, Cloud Build) |
| **API** | FastAPI, Uvicorn |
| **Frontend** | Gradio |
| **Monitoring** | Prometheus, Evidently AI |
| **Code Quality** | Ruff, pre-commit hooks |
| **Package Management** | uv |
