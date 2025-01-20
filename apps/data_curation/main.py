from fastapi import FastAPI, HTTPException
from shared.utils.logger import setup_logging
from shared.config.app_config import config
from pydantic import BaseModel
from apps.data_curation.utils.llm_client import LLMClient
from apps.data_curation.utils.tts_client_fb import TTSClientFB
from apps.data_curation.utils.data_preparation import prepare_whisper_data
from apps.data_curation.utils.db import create_sqlite_database

# Setup logging
setup_logging()

# Initialize the app
app = FastAPI(
    title=config.DATA_CURATION_SERVICE_TITLE,
    version=config.API_VERSION
)

# Initialize database on startup
create_sqlite_database(config.DB_PATH)

# Define request model
class GenerationRequest(BaseModel):
    language: str
    scenario: str
    character: str
    request: str
    nSample: int
    tone: str

# Create TTS client instance using Facebook's MMS-TTS
tts_client_fb = TTSClientFB(
    model_name="facebook/mms-tts",
    db_path=config.DB_PATH
)

# Root endpoint for health check
@app.get("/")
async def health_check():
    return {"status": "Data Curation Service is running"}

@app.post("/generate")
async def generate_data(request: GenerationRequest):
    try:
        # Step 1: Generate the next version for the dataset
        version = "v1"
        
        # Step 2: Generate conversation scripts and get utterance IDs
        llm = LLMClient(
            model=config.LLM_MODEL,
            prompt_version=version,
            db_path=config.DB_PATH
        )
        prompts_path = config.PROMPTS_DIR / version
        utterance_ids = llm.generate_conversations(
            request.dict(),
            prompt_path=prompts_path,
            output_dir=config.DATASETS_DIR
        )
        
        # Step 3: Generate audio files using database-driven approach
        audio_files = tts_client_fb.generate_audio(utterance_ids)
        if not audio_files:
            raise Exception("Failed to generate any audio files")

        # Step 4: Prepare Whisper data
        whisper_data_path = prepare_whisper_data(
            config.DB_PATH,
            output_dir=config.TRAINING_DATA_DIR
        )

        return {
            "status": "success",
            "data_version": version,
            "audio_files": audio_files,
            "whisper_data": str(whisper_data_path)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))