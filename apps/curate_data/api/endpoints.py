from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from apps.data_curation.utils.llm_client import LLMClient
from apps.data_curation.utils.tts_client import TTSClient
from apps.data_curation.utils.data_preparation import prepare_whisper_data
from apps.data_curation.utils.versioning import get_next_version
import os

router = APIRouter()

class GenerationRequest(BaseModel):
    context: str
    num_conversations: int
    max_audio_length: int  # in seconds

@router.post("/generate")
async def generate_data(request: GenerationRequest):
    try:
        # Step 1: Generate the next version for the dataset
        version = get_next_version("data/datasets/")

        # Step 2: Generate conversation scripts
        llm = LLMClient()
        scripts = llm.generate_conversations(
            context=request.context,
            num_conversations=request.num_conversations
        )
        
        # Save scripts
        os.makedirs(f"data/datasets/{version}/", exist_ok=True)
        scripts_path = f"data/datasets/{version}/scripts.json"
        with open(scripts_path, "w") as f:
            f.write(scripts)

        # Step 3: Generate audio files
        tts = TTSClient()
        audio_paths = tts.generate_audio(
            conversations=scripts_path,
            max_length=request.max_audio_length,
            output_dir=f"data/datasets/{version}/audios/"
        )

        # Step 4: Prepare Whisper data
        whisper_data_path = prepare_whisper_data(audio_paths, scripts_path, version)

        return {"status": "success", "data_version": version, "whisper_data": whisper_data_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
