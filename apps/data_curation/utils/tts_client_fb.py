from transformers import VitsModel, AutoTokenizer
import torch
import os
import sqlite3
import torchaudio
from apps.data_curation.utils.language_mapping import is_supported_language


# MMS-TTS language code mapping
LANGUAGE_CODE_MAPPING = {
    'yo': 'yor',  # Yoruba
    'en': 'eng',  # English
    'fr': 'fra',  # French
    'es': 'spa',  # Spanish
    'de': 'deu',  # German
    'it': 'ita',  # Italian
    'pt': 'por',  # Portuguese
    'pl': 'pol',  # Polish
    'tr': 'tur',  # Turkish
    'ru': 'rus',  # Russian
    'nl': 'nld',  # Dutch
    'cs': 'ces',  # Czech
    'ar': 'ara',  # Arabic
    'zh': 'zho',  # Chinese
    'ja': 'jpn',  # Japanese
    'ko': 'kor',  # Korean
}


class TTSClientFB:
    def __init__(
        self,
        model_name="facebook/mms-tts-eng",
        device=None,
        db_path="assets/scenarios.db"
    ):
        """
        Initialize the MMS-TTS client.

        Args:
            model_name (str): Base Hugging Face model repository name.
            device (str, optional): Device to run the model on.
            db_path (str): Path to the SQLite database.
        """
        # Remove language suffix from model name
        self.base_model_name = model_name.rsplit('-', 1)[0]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.db_path = db_path
        print(f"Base model initialized on {self.device}")

    def _load_model_for_language(self, language_code):
        """Load model and tokenizer for a specific language if not already loaded."""
        if language_code not in self.models:
            # Convert ISO language code to MMS-TTS format
            mms_lang = LANGUAGE_CODE_MAPPING.get(language_code.lower())
            if not mms_lang:
                supported = list(LANGUAGE_CODE_MAPPING.keys())
                raise ValueError(
                    f"Language {language_code} not supported. "
                    f"Supported languages: {supported}"
                )

            model_name = f"{self.base_model_name}-tts-{mms_lang}"
            print(f"Loading model for language {language_code}: {model_name}")
            
            self.models[language_code] = (
                VitsModel.from_pretrained(model_name).to(self.device)
            )
            self.tokenizers[language_code] = (
                AutoTokenizer.from_pretrained(model_name)
            )

    def _generate_single_audio(self, utterance, language_code, output_path):
        """
        Generate a single audio file for a given utterance and language.

        Args:
            utterance (str): The text to convert to speech.
            language_code (str): The language code for the utterance.
            output_path (str): Path to save the generated WAV file.
        """
        try:
            # Validate and load language-specific model
            if not is_supported_language(language_code):
                raise ValueError(f"Unsupported language code: {language_code}")

            self._load_model_for_language(language_code)
            
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Get language-specific model and tokenizer
            model = self.models[language_code]
            tokenizer = self.tokenizers[language_code]

            # Tokenize input text
            print(f"Generating audio for text in {language_code}: {utterance}")
            inputs = tokenizer(utterance, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate waveform
            with torch.no_grad():
                waveform = model(**inputs).waveform

            # Save the waveform as a WAV file
            torchaudio.save(
                output_path,
                waveform.cpu(),
                sample_rate=model.config.sampling_rate,
                encoding="PCM_S",
                bits_per_sample=16
            )
            print(f"Generated audio saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return False

    def generate_audio(self, hash_ids):
        """
        Generate audio files for multiple utterances from the database.

        Args:
            hash_ids (list): List of scenario IDs to generate audio for.
        """
        audio_files = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for hash_id in hash_ids:
                query = """
                    SELECT utterance, language_code, audio_path 
                    FROM scenarios WHERE scenario_id = ?
                """
                cursor.execute(query, (hash_id,))
                row = cursor.fetchone()
                if not row:
                    print(f"No record found for hash ID: {hash_id}")
                    continue
                
                utterance, language_code, audio_path = row
                success = self._generate_single_audio(
                    utterance, language_code, audio_path
                )
                if success:
                    audio_files.append(audio_path)

        return audio_files