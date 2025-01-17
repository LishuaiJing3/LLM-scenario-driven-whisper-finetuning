import torch
from TTS.api import TTS
from concurrent.futures import ThreadPoolExecutor
import os
import sqlite3

from language_mapping import is_supported_language


class TTSClient:
    def __init__(self, tts_model="tts_models/multilingual/multi-dataset/xtts_v2", db_path="../../data/assets/scenarios.db"):
        """
        Initialize the TTS client.
        :param tts_model: The TTS model name to use.
        :param db_path: Path to the SQLite database.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name=tts_model, progress_bar=False).to(self.device)
        self.db_path = db_path
        print(f"Using TTS model: {tts_model}")

    @staticmethod
    def list_available_models():
        """
        List all available TTS models.
        :return: List of model names.
        """
        models = TTS().list_models()
        print("Available TTS Models:")
        for model in models:
            print(f"- {model}")
        return models

    def _generate_single_audio(self, utterance, language_code, output_path, speaker_wav="../../data/assets/test_audio.wav"):
        """
        Generate a single audio file for a given utterance and language.
        :param utterance: The text of the utterance.
        :param language_code: The language code for the utterance.
        :param output_path: File path to save the audio.
        :param speaker_wav: Optional reference speaker audio for speaker adaptation.
        """
        # Validate language code
        if not is_supported_language(language_code):
            raise ValueError(f"Unsupported language code: {language_code}")

        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate the audio file
        self.tts.tts_to_file(text=utterance, language=language_code, speaker_wav=speaker_wav, file_path=output_path)
        print(f"Generated audio for: {utterance} -> {output_path}")
        return output_path

    def generate_audio_for_hashes(self, hash_ids, speaker_wav="../../data/assets/test_audio.wav"):
        audio_files = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for hash_id in hash_ids:
                cursor.execute("SELECT utterance, language_code, audio_path FROM scenarios WHERE scenario_id = ?", (hash_id,))
                row = cursor.fetchone()
                if not row:
                    print(f"No record found for hash ID: {hash_id}")
                    continue
                utterance, language_code, audio_path = row
                audio_files.append(self._generate_single_audio(utterance, language_code, 
                                                            audio_path, speaker_wav))
        return audio_files


if __name__ == "__main__":
    tts_client = TTSClient()

    # List available models (optional)
    # tts_client.list_available_models()

    # Define hash IDs to generate audio for
    hash_ids = ["228e1a089f4b063f","76c6d59d82159fa7"]

    # Generate audio for the given hash IDs
    audio_files = tts_client.generate_audio_for_hashes(hash_ids, speaker_wav="../../data/assets/test_audio.wav")

    print(f"Generated audio files: {audio_files}")
