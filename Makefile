.PHONY: data_curation training serving docker-build docker-run docker-stop docker-clean clean

# Local Environment
# Run Data Curation API
data_curation:
	poetry run uvicorn apps.data_curation.main:app --host 0.0.0.0 --port 8001 --reload

# Run Training API
training:
	poetry run uvicorn apps.training.main:app --host 0.0.0.0 --port 8002 --reload

# Run Serving API
serving:
	poetry run uvicorn apps.serving.main:app --host 0.0.0.0 --port 8003 --reload

# Docker Environment
# Build Docker images for all services
docker-build:
	docker build -f docker/Dockerfile.data_curation -t data-curation-api .
	docker build -f docker/Dockerfile.training -t training-api .
	docker build -f docker/Dockerfile.serving -t serving-api .

# Run Docker containers for all services
docker-run:
	docker run -d -p 8001:8000 --name data-curation-api data-curation-api
	docker run -d -p 8002:8000 --name training-api training-api
	docker run -d -p 8003:8000 --name serving-api serving-api

# Stop Docker containers
docker-stop:
	docker stop data-curation-api || true
	docker stop training-api || true
	docker stop serving-api || true

# Remove Docker containers and images
docker-clean:
	docker rm -f data-curation-api || true
	docker rm -f training-api || true
	docker rm -f serving-api || true
	docker rmi -f data-curation-api training-api serving-api || true

# Remove all Docker containers and images
clean:
	docker rm -f $$(docker ps -aq) || true
	docker rmi -f $$(docker images -q) || true
