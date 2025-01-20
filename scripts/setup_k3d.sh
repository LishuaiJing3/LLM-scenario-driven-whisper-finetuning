#!/bin/bash

# Default values
export CONTAINER_REGISTRY=${CONTAINER_REGISTRY:-"ghcr.io"}
export REGISTRY_NAMESPACE=${REGISTRY_NAMESPACE:-"local"}
export IMAGE_TAG=${IMAGE_TAG:-"latest"}
export GITHUB_OWNER=${GITHUB_OWNER:-"local"}
export GITHUB_REPO=${GITHUB_REPO:-"whisper-finetuning"}

# Create k3d cluster with GPU support
k3d cluster create whisper-cluster \
  --gpus all \
  --agents 2 \
  --k3s-arg '--node-label=gpu=true@agent:0' \
  --k3s-arg '--node-label=gpu=true@agent:1' \
  --volume /tmp/k3dvol:/var/lib/rancher/k3s/storage@all

# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Create namespace
kubectl create namespace whisper

# Generate Kubernetes manifests
python scripts/generate_manifests.py

# Apply manifests
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/data-curation.yaml
kubectl apply -f k8s/training.yaml
kubectl apply -f k8s/serving.yaml

# Setup port forwarding in background
kubectl port-forward -n whisper svc/data-curation 8001:8000 &
kubectl port-forward -n whisper svc/training 8002:8000 &
kubectl port-forward -n whisper svc/serving 8003:8000 &

echo "Cluster setup complete!"
echo "Services are available at:"
echo "- Data Curation: http://localhost:8001"
echo "- Training: http://localhost:8002"
echo "- Serving: http://localhost:8003"
echo
echo "Use 'kubectl get pods -n whisper' to check status." 