from fastapi import FastAPI, HTTPException
from shared.utils.logger import setup_logging
from pydantic import BaseModel
from apps.data_curation.utils.llm_client import LLMClient
from apps.data_curation.utils.tts_client import TTSClient
from apps.data_curation.utils.data_preparation import prepare_whisper_data
import os

# Initialize the app
app = FastAPI(title="Data Curation Service", version="1.0")


class GenerationRequest(BaseModel):
    language: str
    scenario: str
    character: str
    request: str
    nSample: int  # in seconds
    tone: str


# Setup logging
setup_logging()


# Root endpoint for health check
@app.get("/")
async def health_check():
    return {"status": "Data Curation Service is running"}

@app.post("/generate")
async def generate_data(request: GenerationRequest):
    try:
        # Step 1: Generate the next version for the dataset
        
        version = "v1"
        
        # add init db to create if it does not exist
        db_path = "data/assets/scenarios.db"

        # Step 2: Generate conversation scripts
        llm = LLMClient(model="gemini-2.0-flash-exp", prompt_version="v1", db_path="data/assets/scenarios.db")
        prompts_path = f"apps/data_curation/prompts/{version}/"
        utterance_ids = llm.generate_conversations(
            request.dict(),
            prompt_path=prompts_path,
            output_dir=f"data/datasets/"
        )
        
        # Step 3: Generate audio files
        tts = TTSClient(tts_model="tts_models/multilingual/multi-dataset/xtts_v2", db_path="data/assets/scenarios.db")
        tts.generate_audio(
            utterance_ids,
            "data/assets/test_audio.wav"
        )

        # Step 4: Prepare Whisper data
        whisper_data_path = prepare_whisper_data(db_path, output_dir="data/training_data/")

        return {"status": "success", "data_version": version, "whisper_data": whisper_data_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

