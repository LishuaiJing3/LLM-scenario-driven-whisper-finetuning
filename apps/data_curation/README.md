# Data Curation Service

This service handles data curation for training Whisper models.

## Endpoints

- `/api/data_curation/generate`: Generate data for training.

## Setup

1. Install dependencies:
    ```sh
    poetry install
    ```

2. Run the service:
    ```sh
    poetry run uvicorn apps.data_curation.api.main:app --host 0.0.0.0 --port 8000
    ```

# notes
We do not use parallel for geenrating audios yet. it causes issues when running parallel jobs and need locks for synchronization 