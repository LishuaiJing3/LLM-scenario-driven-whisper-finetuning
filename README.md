# Whisper Fine-tuning Project

This project provides a complete pipeline for fine-tuning Whisper models with custom data, including data curation, training, and serving components.

## Architecture

The project consists of three microservices:
- Data Curation Service: Generates and prepares training data
- Training Service: Handles model fine-tuning
- Serving Service: Provides inference endpoints

## Prerequisites

- Python 3.11+
- Poetry for dependency management
- Docker
- k3d
- kubectl
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

## Configuration

The project can be configured using environment variables:

```bash
# Container Registry Configuration
export CONTAINER_REGISTRY=ghcr.io  # or your preferred registry
export REGISTRY_NAMESPACE=your-namespace
export IMAGE_TAG=latest  # or specific version

# GitHub Configuration
export GITHUB_OWNER=your-username
export GITHUB_REPO=your-repo-name
```

For local development, you can leave these unset and they will default to local values.

## Local Development

1. Install dependencies:
```bash
poetry install
```

2. Run services locally:
```bash
# Data Curation Service
make data_curation

# Training Service
make training

# Serving Service
make serving
```

## Kubernetes Deployment

1. Set up k3d cluster with GPU support:
```bash
chmod +x scripts/setup_k3d.sh
./scripts/setup_k3d.sh
```

2. Check deployment status:
```bash
kubectl get pods -n whisper
```

3. Access services:
- Data Curation: http://localhost:8001
- Training: http://localhost:8002
- Serving: http://localhost:8003

## Development Workflow

1. Local Development:
```bash
# Build images locally
docker build -f docker/Dockerfile.data_curation -t data-curation:local .
docker build -f docker/Dockerfile.training -t training:local .
docker build -f docker/Dockerfile.serving -t serving:local .
```

2. Cloud Deployment:
```bash
# Set your cloud configuration
export CONTAINER_REGISTRY=your-registry
export REGISTRY_NAMESPACE=your-namespace
export IMAGE_TAG=your-version

# Generate Kubernetes manifests
python scripts/generate_manifests.py

# Apply to your cluster
kubectl apply -f k8s/
```

## CI/CD Pipeline

The project uses GitHub Actions for CI/CD:
1. On pull requests:
   - Runs tests
   - Builds Docker images
2. On merge to main:
   - Runs tests
   - Builds and pushes Docker images to configured registry
   - Updates Kubernetes deployments

## API Documentation

Each service provides a Swagger UI at `/docs` endpoint:
- Data Curation: http://localhost:8001/docs
- Training: http://localhost:8002/docs
- Serving: http://localhost:8003/docs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License