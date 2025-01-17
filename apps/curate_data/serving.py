import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

def load_model_and_processor(model_dir="whisper_finetuned"):
    """
    Load the fine-tuned Whisper model and processor.
    :param model_dir: Path to the directory where the fine-tuned model is saved.
    :return: Loaded model and processor.
    """
    print(f"Loading model and processor from {model_dir}...")
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.eval()  # Set the model to evaluation mode
    return model, processor


def preprocess_audio(audio_path, processor, sampling_rate=16000):
    """
    Load and preprocess audio for Whisper inference.
    :param audio_path: Path to the audio file.
    :param processor: Whisper processor for feature extraction.
    :param sampling_rate: Target sampling rate for the audio.
    :return: Preprocessed input features.
    """
    print(f"Preprocessing audio: {audio_path}")
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=sampling_rate)

    # Extract features
    input_features = processor.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
    return input_features


def transcribe_audio(audio_path, model, processor, language_code="en"):
    """
    Perform inference on a single audio file and generate transcription.
    :param audio_path: Path to the audio file.
    :param model: Fine-tuned Whisper model.
    :param processor: Whisper processor.
    :param language_code: Language code for transcription (e.g., 'en').
    :return: Transcription as a string.
    """
    # Preprocess the audio
    input_features = preprocess_audio(audio_path, processor)

    # Generate transcription
    print(f"Generating transcription for {audio_path}...")
    with torch.no_grad():
        generated_ids = model.generate(input_features, forced_decoder_ids=processor.get_decoder_prompt_ids(language=language_code, task="transcribe"))
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription


if __name__ == "__main__":
    # Path to the fine-tuned model directory
    model_dir = "whisper_finetuned"

    # Path to an audio file for inference
    audio_path = "../../data/inference_audio/sample.wav"

    # Load the model and processor
    model, processor = load_model_and_processor(model_dir)

    # Transcribe the audio
    try:
        transcription = transcribe_audio(audio_path, model, processor, language_code="en")
        print(f"Transcription:\n{transcription}")
    except Exception as e:
        print(f"Error during inference: {e}")
