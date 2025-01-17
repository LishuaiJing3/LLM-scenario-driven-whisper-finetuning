import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

def load_model_and_processor(model_dir="whisper_finetuned"):
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    return model, processor

def preprocess_audio(audio_path, processor, sampling_rate=16000):
    print(f"Preprocessing audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    input_features = processor.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
    return input_features

def transcribe_audio(audio_path, model_dir, language_code="en"):
    model, processor = load_model_and_processor(model_dir)
    input_features = preprocess_audio(audio_path, processor)
    print(f"Generating transcription for {audio_path}...")
    with torch.no_grad():
        generated_ids = model.generate(input_features, forced_decoder_ids=processor.get_decoder_prompt_ids(language=language_code, task="transcribe"))
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription