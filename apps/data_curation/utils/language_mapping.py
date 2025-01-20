# language_mapping.py

# Define the supported languages and their mappings
LANGUAGE_MAPPING = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "pl": "polish",
    "tr": "turkish",
    "ru": "russian",
    "nl": "dutch",
    "cs": "czech",
    "ar": "arabic",
    "zh-cn": "chinese",
    "hu": "hungarian",
    "ko": "korean",
    "ja": "japanese",
    "hi": "hindi"
}

# Reverse the mapping for lookup by full name
FULL_NAME_TO_CODE = {v: k for k, v in LANGUAGE_MAPPING.items()}

def get_language_code(language_name):
    """
    Get the language code for a given language name.
    :param language_name: Full language name (case insensitive) or language code.
    :return: Language code (e.g., 'en') or None if not found.
    """
    language_name = language_name.lower().strip()
    
    # If it's already a language code, return it if valid
    if language_name in LANGUAGE_MAPPING:
        return language_name
    
    # Check the full name to code mapping
    return FULL_NAME_TO_CODE.get(language_name)

def is_supported_language(language_code):
    """
    Check if a given language code is supported.
    :param language_code: Language code to check.
    :return: True if supported, False otherwise.
    """
    return language_code in LANGUAGE_MAPPING


