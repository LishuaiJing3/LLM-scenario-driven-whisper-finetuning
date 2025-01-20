from pathlib import Path

class AppConfig:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    # Database configuration
    DB_PATH = DATA_DIR / "assets" / "scenarios.db"
    
    # Data directories
    ASSETS_DIR = DATA_DIR / "assets"
    DATASETS_DIR = DATA_DIR / "datasets"
    TRAINING_DATA_DIR = DATA_DIR / "training_data"
    
    # Prompts configuration
    PROMPTS_DIR = BASE_DIR / "apps" / "data_curation" / "prompts"
    
    # Model configurations
    LLM_MODEL = "gemini-2.0-flash-exp"
    TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    # API configurations
    TRAINING_SERVICE_TITLE = "Training Service"
    SERVING_SERVICE_TITLE = "Serving Service"
    DATA_CURATION_SERVICE_TITLE = "Data Curation Service"
    API_VERSION = "1.0"

config = AppConfig()
