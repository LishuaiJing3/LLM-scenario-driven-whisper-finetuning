import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

def load_model_and_processor(model_dir="whisper_finetuned"):
    """
    Load the fine-tuned Whisper model and processor.
    """
    print(f"Loading model and processor from {model_dir}...")
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.eval()  # Set the model to evaluation mode
    return model, processor

def preprocess_audio(audio_path, processor, sampling_rate=16000):
    """
    Load and preprocess audio for Whisper inference.
    """
    print(f"Preprocessing audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sampling_rate)

    # Ask the processor to return the attention mask
    features = processor.feature_extractor(
        audio, 
        sampling_rate=sampling_rate, 
        return_tensors="pt", 
        return_attention_mask=True
    )
    return features

def transcribe_audio(audio_path, model_dir, language_code="en"):
    """
    Perform inference on a single audio file and generate transcription.
    """
    model, processor = load_model_and_processor(model_dir)
    # Preprocess the audio to get both input_features and attention_mask
    features = preprocess_audio(audio_path, processor)
    input_features = features["input_features"]
    attention_mask = features["attention_mask"]

    # Generate transcription
    print(f"Generating transcription for {audio_path}...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,  # Pass the attention mask here
            forced_decoder_ids=processor.get_decoder_prompt_ids(
                language=language_code,
                task="transcribe"
            )
        )
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

if __name__ == "__main__":
    model_dir = "data/whisper_finetuned"
    audio_path = "data/inference_audio/output1.wav"

    try:
        transcription = transcribe_audio(audio_path, model_dir, language_code="en")
        print(f"Transcription:\n{transcription}")
    except Exception as e:
        print(f"Error during inference: {e}")
