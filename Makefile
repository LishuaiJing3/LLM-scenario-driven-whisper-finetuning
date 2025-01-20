.PHONY: data_curation training serving docker-build docker-run docker-stop docker-clean compose-build compose-run compose-stop compose-clean deploy

# Loading environment variables
include .env
export $(shell sed 's/=.*//' .env)

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

# Docker Environment with Docker Compose
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose stop

docker-clean:
	docker-compose down -v

# Kubernetes (k3d) Environment
k3d-setup:
	./scripts/setup_k3d.sh

k3d-deploy:
	kubectl apply -f k8s/storage.yaml
	kubectl apply -f k8s/data-curation.yaml
	kubectl apply -f k8s/training.yaml
	kubectl apply -f k8s/serving.yaml

k3d-clean:
	kubectl delete -f k8s/serving.yaml
	kubectl delete -f k8s/training.yaml
	kubectl delete -f k8s/data-curation.yaml
	kubectl delete -f k8s/storage.yaml
	k3d cluster delete whisper-finetuning-cluster

# Deployment to Kubernetes
deploy:
	./scripts/deploy.sh

# Shortcut Targets
compose-build: docker-build
compose-run: docker-run
compose-stop: docker-stop
compose-clean: docker-clean

# General Cleanup
clean: compose-clean k3d-clean
	docker system prune -a -f --volumes
