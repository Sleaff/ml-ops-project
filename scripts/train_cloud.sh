#!/bin/bash
set -e

# GCS bucket configuration
GCS_BUCKET="gs://sleaff_mlops_data_bucket"

echo "=== Cloud Training Script ==="

# 1. Pull data from GCS
echo "Pulling data from GCS..."
mkdir -p /data/raw /data/processed
gsutil -m cp "${GCS_BUCKET}/data/raw/*" /data/raw/
gsutil -m cp "${GCS_BUCKET}/data/processed/*" /data/processed/

echo "Data downloaded:"
ls -la /data/raw/
ls -la /data/processed/

# 2. Run training
echo "Starting training..."
uv run src/project/train.py \
    data_path=/data/raw \
    processed_data_path=/data/processed/news.csv \
    save_dir=/models \
    "$@"

# 3. Upload checkpoints to GCS
echo "Uploading models to GCS..."
gsutil -m cp -r /models/* "${GCS_BUCKET}/src/project/models/"

echo "=== Training complete ==="
