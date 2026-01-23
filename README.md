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
