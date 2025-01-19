from fastapi import FastAPI, HTTPException
from shared.utils.logger import setup_logging
from pydantic import BaseModel
from apps.serving.utils.inference_utils import transcribe_audio
from shared.config.app_config import config

# Initialize the app
app = FastAPI(title=config.SERVING_SERVICE_TITLE, version=config.API_VERSION)

class InferenceRequest(BaseModel):
    audio_path: str
    model_dir: str
    language_code: str
# Setup logging
setup_logging()

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