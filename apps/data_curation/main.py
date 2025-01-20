from fastapi import FastAPI, HTTPException
from shared.utils.logger import setup_logging
from shared.config.app_config import config
from pydantic import BaseModel
from apps.data_curation.utils.llm_client import LLMClient
from apps.data_curation.utils.tts_client import TTSClient
from apps.data_curation.utils.data_preparation import prepare_whisper_data
from apps.data_curation.utils.db import create_sqlite_database
import os

# Initialize the app
app = FastAPI(title=config.DATA_CURATION_SERVICE_TITLE, version=config.API_VERSION)

# Initialize database on startup
create_sqlite_database(config.DB_PATH)  # This will create the DB if it doesn't exist

class GenerationRequest(BaseModel):
    language: str
    scenario: str
    character: str
    request: str
    nSample: int  
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
        
        # Step 2: Generate conversation scripts
        llm = LLMClient(model=config.LLM_MODEL, prompt_version=version, db_path=config.DB_PATH)
        prompts_path = config.PROMPTS_DIR / version
        utterance_ids = llm.generate_conversations(
            request.dict(),
            prompt_path=prompts_path,
            output_dir=config.DATASETS_DIR
        )
        
        # Step 3: Generate audio files
        tts = TTSClient(tts_model=config.TTS_MODEL, db_path=config.DB_PATH)
        tts.generate_audio(
            utterance_ids,
            config.ASSETS_DIR / "test_audio.wav"
        )

        # Step 4: Prepare Whisper data
        whisper_data_path = prepare_whisper_data(config.DB_PATH, output_dir=config.TRAINING_DATA_DIR)

        return {"status": "success", "data_version": version, "whisper_data": str(whisper_data_path)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

