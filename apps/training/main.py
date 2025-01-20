from fastapi import FastAPI, HTTPException
from shared.utils.logger import setup_logging
from shared.config.app_config import config
from pydantic import BaseModel
from apps.training.utils.training_utils import start_training

# Initialize the app
app = FastAPI(title=config.TRAINING_SERVICE_TITLE, version=config.API_VERSION)

# Setup logging
setup_logging()

class TrainingRequest(BaseModel):
    dataset_path: str
    model_name: str
    output_dir: str
    language_filter: str

# Root endpoint for health check
@app.get("/")
async def health_check():
    return {"status": "Training Service is running"}

@app.post("/train")
async def train_model(request: TrainingRequest):
    try:
        start_training(
            dataset_path=request.dataset_path,
            model_name=request.model_name,
            output_dir=request.output_dir,
            language_filter=request.language_filter
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))