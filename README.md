# project

Machine Learning Operations Project

## Students

s195171 - Kenneth Plum Toft <br>
s242726 - Philip Arthur Blaafjell <br>
s243586 - Joakim Dinh <br>
s242723 - Vebjørn Sæten Skre <br>

## Goal of the project

The goal for this project is to learn the frameworks around ML-ops. Our group consists of students with extensive experience with machine learning, but not too much with the framework around it. We are all inexperienced with using the tools proposed in the course so the goal for our group would be to implement as much of the tools from the course as possible into the project so we can see which tools we like and would want to work with later. Moreover we also want to get used to working in a collaborative manner where we can document, view and understand the changes the others make.

## The Data

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

We propse a project to classify news. We have identified a dataset which includes two types of classes. One with fake news and one with true news. The dataset is mostly pure text but also includes other features such as the type of news and the date the text was published. The dataset includes 17903 unqiue datapoints.

## The models

For the model we will try to use a BERT model. BERT is a language model from google which only uses the encoder part of a transformer architecture. We expect the model to do well on the task after we fine tune since it had a huge impact on the NLP field when it came out.

## Google drive

Data used and model can be found here:

https://drive.google.com/drive/folders/1hMOLYBrIgZx8Rt_HkMGnNt8k7VrpikL_?usp=sharing

---

## Cloud Training on Google Cloud Platform

This project supports distributed training on Google Cloud Platform (GCP) using Docker containers, Cloud Build, and Cloud Storage. This setup allows you to train models on remote VMs without needing local Docker or GPU resources.

### Architecture Overview

- **Cloud Build**: Builds Docker images in the cloud (no local Docker required)
- **Container Registry (GCR)**: Stores training Docker images
- **Compute Engine VM**: Runs training workloads (`mlops-training-vm`)
- **Cloud Storage (GCS)**: Stores datasets and trained model checkpoints
- **Weights & Biases**: Optional experiment tracking and visualization

### Prerequisites

1. **GCP Project Setup**
   - Project ID: `mlops-483515`
   - Region: `europe-west1-b`
   - Ensure you have appropriate IAM permissions (Compute Admin, Storage Admin, Cloud Build Editor)

2. **Local Requirements**
   - Google Cloud SDK (`gcloud`) installed and authenticated
   - Python environment with `uv` and `invoke`

3. **Authentication**
   ```bash
   gcloud auth login
   gcloud config set project mlops-483515
   ```

### One-Time Setup

#### 1. Create GCP VM
```bash
uv run invoke gcloud-create-vm
```

Creates a VM with:
- Machine type: `e2-medium` (2 vCPU, 4GB RAM)
- Image: Deep Learning VM with PyTorch 2.7 + CUDA 12.8
- Boot disk: 100GB
- Full cloud platform access scope

#### 2. Upload Training Data to Cloud Storage
```bash
uv run invoke gcloud-upload-data
```

Uploads data from `data/raw/` and `data/processed/` to GCS bucket `gs://sleaff_mlops_data_bucket/`.

#### 3. Install Docker on VM (First Time Only)
```bash
gcloud compute ssh mlops-training-vm --zone=europe-west1-b --project=mlops-483515
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER
exit
```

This is automatically handled in the training command, but can be done manually if needed.

### Training Workflow

#### Standard Training Cycle

**1. Build Docker Image**
```bash
uv run invoke gcloud-build-push
```

- Builds training image using Cloud Build (takes ~2-3 minutes)
- Pushes to `gcr.io/mlops-483515/train:latest`
- No local Docker installation required
- Uses `cloudbuild.yaml` configuration

**2. Run Training on VM**
```bash
uv run invoke gcloud-train
```

What happens:
- SSHs into the VM
- Configures Docker authentication for GCR
- Pulls the latest training image (first time only, cached afterwards)
- Runs the training container with:
  - Data downloaded from GCS bucket
  - Training executed with PyTorch Lightning
  - Model checkpoints uploaded back to GCS
- **Training logs stream directly to your terminal**

**3. Download Trained Models**
```bash
uv run invoke gcloud-download-models
```

Downloads all model checkpoints from `gs://sleaff_mlops_data_bucket/src/project/models/` to local `models/` directory.

**4. Stop VM to Save Costs**
```bash
uv run invoke gcloud-stop-vm
```

**Important**: Always stop the VM when not training to avoid unnecessary charges.

### Training with Weights & Biases

To enable W&B experiment tracking:

```bash
# Set your W&B API key
export WANDB_API_KEY=your_wandb_api_key_here

# Run training with W&B logging
uv run invoke gcloud-train --wandb-key=$WANDB_API_KEY
```

View experiments at: `https://wandb.ai/your-username/news-classification`

Get your API key from: https://wandb.ai/authorize

### Monitoring Training

#### Option 1: Terminal Logs (Recommended)
Training logs automatically stream to your terminal when running `uv run invoke gcloud-train`. You'll see:
- Data download progress
- Training progress bars
- Loss and accuracy metrics
- Model checkpoint saves
- Upload status to GCS

#### Option 2: Live Docker Logs
While training is running, view live logs from another terminal:
```bash
gcloud compute ssh mlops-training-vm --zone=europe-west1-b --project=mlops-483515 \
  --command="docker logs -f \$(docker ps -q)"
```

#### Option 3: GCP Cloud Logging
View logs in GCP Console:
- Navigate to: https://console.cloud.google.com/logs/query?project=mlops-483515
- Filter: `resource.type="gce_instance"` and `resource.labels.instance_id="mlops-training-vm"`

#### Option 4: Weights & Biases Dashboard
If W&B is configured, view real-time metrics, system stats, and experiment comparisons at your W&B project page.

### Configuration

All GCP settings are defined in `tasks.py`:

```python
GCP_PROJECT = "mlops-483515"
GCP_ZONE = "europe-west1-b"
GCP_VM_NAME = "mlops-training-vm"
GCP_IMAGE_NAME = f"gcr.io/{GCP_PROJECT}/train:latest"
GCS_BUCKET = "gs://sleaff_mlops_data_bucket"
```

Training hyperparameters are configured in `configs/config.yaml`:
```yaml
epochs: 10
batch_size: 64
lr: 2e-5
model_name: "distilbert-base-uncased"
max_length: 256
```

Override via command line in `scripts/train_cloud.sh` or modify the config file.

### Docker Image Details

The training Docker image (`dockerfiles/train.dockerfile`):
- Base: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
- Includes Google Cloud SDK for GCS access
- Installs all dependencies via `uv sync`
- Entrypoint: `scripts/train_cloud.sh`

Training script (`scripts/train_cloud.sh`) workflow:
1. Downloads data from GCS bucket
2. Runs training with `src/project/train.py`
3. Uploads model checkpoints to GCS bucket

### Cost Management

**Estimated Costs** (europe-west1):
- VM (e2-medium): ~$0.03/hour when running
- Cloud Storage: ~$0.02/GB/month
- Container Registry: ~$0.10/GB/month
- Cloud Build: 120 free build-minutes/day

**Cost Saving Tips**:
1. Always stop the VM when not training: `uv run invoke gcloud-stop-vm`
2. Delete old model checkpoints from GCS periodically
3. Use preemptible VMs for non-critical training (modify `gcloud_create_vm` task)
4. Monitor usage: https://console.cloud.google.com/billing

### Troubleshooting

**VM doesn't have Docker**
```bash
gcloud compute ssh mlops-training-vm --zone=europe-west1-b --project=mlops-483515
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER
exit
```

**Docker authentication fails**
```bash
gcloud compute ssh mlops-training-vm --zone=europe-west1-b --project=mlops-483515 \
  --command="gcloud auth configure-docker gcr.io --quiet"
```

**Training fails due to missing data**
Ensure data is uploaded to GCS:
```bash
uv run invoke gcloud-upload-data
```

**Out of disk space on VM**
Increase boot disk size when creating VM (modify `--boot-disk-size` in `tasks.py`).

**Permission denied errors**
Ensure your GCP account has the required IAM roles:
- Compute Instance Admin
- Storage Object Admin
- Cloud Build Editor

### Advanced Usage

#### Custom Training Arguments
Modify `scripts/train_cloud.sh` to pass Hydra overrides:
```bash
uv run src/project/train.py \
    epochs=20 \
    batch_size=128 \
    lr=1e-5 \
    "$@"
```

#### Using Different VM Types
Edit `tasks.py` to use GPU-enabled instances:
```python
--machine-type=n1-standard-4 \
--accelerator=type=nvidia-tesla-t4,count=1 \
```

#### Multiple Training Runs
The training script supports Hydra multirun for hyperparameter sweeps. Modify the entrypoint to use:
```bash
uv run src/project/train.py --multirun lr=1e-5,2e-5,5e-5 batch_size=32,64
```

### Available Commands Summary

| Command | Description |
|---------|-------------|
| `uv run invoke gcloud-create-vm` | Create training VM (one-time) |
| `uv run invoke gcloud-build-push` | Build Docker image in cloud |
| `uv run invoke gcloud-train` | Run training on VM |
| `uv run invoke gcloud-train --wandb-key=KEY` | Run training with W&B logging |
| `uv run invoke gcloud-stop-vm` | Stop VM to save costs |
| `uv run invoke gcloud-upload-data` | Upload data to GCS |
| `uv run invoke gcloud-download-models` | Download trained models |

### Quick Reference

**Full training workflow:**
```bash
# 1. Make code changes locally
# 2. Build new image
uv run invoke gcloud-build-push

# 3. Run training (logs stream to terminal)
uv run invoke gcloud-train

# 4. Download results
uv run invoke gcloud-download-models

# 5. Stop VM
uv run invoke gcloud-stop-vm
```

**Check VM status:**
```bash
gcloud compute instances list --project=mlops-483515
```

**SSH into VM:**
```bash
gcloud compute ssh mlops-training-vm --zone=europe-west1-b --project=mlops-483515
```

**View GCS bucket contents:**
```bash
gsutil ls -r gs://sleaff_mlops_data_bucket/
```
