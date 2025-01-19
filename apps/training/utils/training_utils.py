import json
import torch
from datasets import Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from torch.nn.utils.rnn import pad_sequence

# Optional bitsandbytes integration
# pip install bitsandbytes
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


# -----------------------------
# 1. Detect device
# -----------------------------
def get_device():
    """
    Return the best available device: CUDA if available, else MPS if available, else CPU.
    """
    if torch.cuda.is_available():
        print("Using CUDA device.")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple Silicon MPS device.")
        return "mps"
    else:
        print("Using CPU device.")
        return "cpu"

DEVICE = get_device()


# -----------------------------
# 2. Whisper Data Collator
# -----------------------------
class WhisperDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [torch.tensor(f["input_features"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        # Pad the input_features
        input_features = pad_sequence(input_features, batch_first=True)

        # Pad the labels with the tokenizer pad token
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        )

        # Replace pad token with -100 for cross-entropy ignoring
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"input_features": input_features, "labels": labels}


# -----------------------------
# 3. Preprocessing function
# -----------------------------
def preprocess_batch(batch):
    input_features_list = []
    labels_list = []

    for audio, text, language_code in zip(
        batch["audio_filepath"], batch["text"], batch["language_code"]
    ):
        # Extract audio features
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=sampling_rate
        ).input_features[0]

        # Prepend language token
        language_token = f"<|{language_code}|>"
        labels = processor.tokenizer(f"{language_token} {text}", return_tensors="pt").input_ids[0]

        input_features_list.append(input_features)
        labels_list.append(labels)

    return {"input_features": input_features_list, "labels": labels_list}


# -----------------------------
# 4. Training function
# -----------------------------
def start_training(
    dataset_path,
    model_name,
    output_dir,
    language_filter=None,
    use_8bit=False,
):
    """
    Fine-tune a Whisper model with options for device detection (cuda/mps/cpu) and optional 8-bit.
    """

    # Load JSON data
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Filter by language code if provided
    if language_filter:
        data = [item for item in data if item["language_code"] == language_filter]
        if not data:
            raise ValueError(f"No data found for language code: {language_filter}")
        print(f"Filtered dataset to {len(data)} examples for language code: {language_filter}")

    # Convert to HF Dataset
    dataset = Dataset.from_list(data)

    # Cast audio column to Audio feature
    global sampling_rate
    sampling_rate = 16000
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=sampling_rate))

    # Decide if we want quantization
    # For Apple Silicon (MPS), bitsandbytes 8-bit is not supported, so skip it.
    quantization_config = None
    if use_8bit and BITSANDBYTES_AVAILABLE and DEVICE == "cuda":
        # Use 8-bit quantization
        print("Using bitsandbytes 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    else:
        if use_8bit and DEVICE != "cuda":
            print("Warning: 8-bit quantization only works on NVIDIA GPUs, skipping...")

    # Load processor
    global processor
    processor = WhisperProcessor.from_pretrained(model_name)

    # Load model
    if quantization_config is not None:
        # This automatically places the model on the GPU with 8-bit weights
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",  # let HF handle device mapping
        )
    else:
        # Normal load in full precision
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        # Move model to device manually if not using device_map='auto'
        model.to(DEVICE)

    # Preprocess dataset
    dataset = dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Data collator
    data_collator = WhisperDataCollator(processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,  # simpler with a tiny dataset
        learning_rate=1e-5,
        num_train_epochs=3,
        logging_dir=f"{output_dir}/logs",
        logging_steps=1,
        save_steps=50,
        eval_steps=50,
        save_total_limit=2,
        # If you are on MPS, you can try bf16=True if supported:
        bf16=(DEVICE == "cuda"),  # or use bf16 on MPS if PyTorch supports it
        fp16=(DEVICE == "cuda"),  # typical for cuda
        # For MPS, do not set fp16=True. MPS uses a separate half mechanism.
        # If you want compile speedups on newer PyTorch, you can also try:
        # torch_compile=True
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )

    # Train
    trainer.train()

    # Save model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuning complete. Model and processor saved to: {output_dir}")


# -----------------------------
# 5. If running as script
# -----------------------------
if __name__ == "__main__":
    model_name = "openai/whisper-small"

    # Pre-download the model to avoid slow downloads in training loop
    print("Downloading model and processor if not already cached...")
    WhisperProcessor.from_pretrained(model_name)
    WhisperForConditionalGeneration.from_pretrained(model_name)
    print("Model and processor downloaded successfully.")

    # Example usage:
    start_training(
        dataset_path="data/training_data/training_data.json",
        model_name=model_name,
        output_dir="data/whisper_finetuned",
        language_filter="en",  # or None
        use_8bit=True,         # only works if you have a CUDA/NVIDIA GPU
    )
    print("Training complete.")



"""
{
  "dataset_path": "data/training_data/training_data.json",
  "model_name": "openai/whisper-small",
  "output_dir": "data/whisper_finetuned",
  "language_filter": "en"
}
"""