import httpx
from typing import Optional, Dict, Any
import os


class WhisperAPIClient:
    def __init__(self):
        self.data_curation_url = os.getenv(
            "DATA_CURATION_URL", 
            "http://data-curation:8000"
        )
        self.training_url = os.getenv("TRAINING_URL", "http://training:8000")
        self.serving_url = os.getenv("SERVING_URL", "http://serving:8000")
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minutes timeout
    
    async def generate_data(
        self,
        language: str,
        scenario: str,
        character: str,
        request: str,
        n_sample: int,
        tone: str
    ) -> Dict[str, Any]:
        """Generate training data using the data curation service."""
        response = await self.client.post(
            f"{self.data_curation_url}/generate",
            json={
                "language": language,
                "scenario": scenario,
                "character": character,
                "request": request,
                "nSample": n_sample,
                "tone": tone
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def train_model(
        self,
        dataset_path: str,
        model_name: str,
        output_dir: str,
        language_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start model training using the training service."""
        response = await self.client.post(
            f"{self.training_url}/train",
            json={
                "dataset_path": dataset_path,
                "model_name": model_name,
                "output_dir": output_dir,
                "language_filter": language_filter
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def transcribe_audio(
        self,
        audio_path: str,
        model_dir: str,
        language_code: str
    ) -> Dict[str, Any]:
        """Transcribe audio using the serving service."""
        response = await self.client.post(
            f"{self.serving_url}/transcribe",
            json={
                "audio_path": audio_path,
                "model_dir": model_dir,
                "language_code": language_code
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose() 