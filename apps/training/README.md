# Training Service

This service handles the training of Whisper models.

## Endpoints

- `/api/training/train`: Start training a Whisper model.

## Setup

1. Install dependencies:
    ```sh
    poetry install
    ```

2. Run the service:
    ```sh
    poetry run uvicorn apps.training.api.main:app --host 0.0.0.0 --port 8000
    ```