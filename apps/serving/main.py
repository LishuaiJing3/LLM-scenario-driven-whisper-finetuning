from fastapi import FastAPI
from apps.serving.api.endpoints import router
from apps.serving.utils.logger import setup_logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from apps.serving.utils.inference_utils import transcribe_audio
import os

# Initialize the app
app = FastAPI(title="Serving Service", version="1.0")

class InferenceRequest(BaseModel):
    audio_path: str
    model_dir: str
    language_code: str
# Setup logging
setup_logging()

# Include the API router
app.include_router(router, prefix="/api/serving", tags=["Serving"])

# Root endpoint for health check
@app.get("/")
async def health_check():
    return {"status": "Serving Service is running"}

@app.post("/transcribe")
async def transcribe(request: InferenceRequest):
    try:
        transcription = transcribe_audio(
            audio_path=request.audio_path,
            model_dir=request.model_dir,
            language_code=request.language_code
        )
        return {"status": "success", "transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))