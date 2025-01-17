import json
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Audio
import torch

def start_training(dataset_path, model_name, output_dir, language_filter):
    # Load the prepared dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Filter the dataset by language if specified
    if language_filter:
        data = [item for item in data if item["language_code"] == language_filter]
        if not data:
            raise ValueError(f"No data found for language code: {language_filter}")
        print(f"Filtered dataset to {len(data)} examples for language code: {language_filter}")

    # Create a Hugging Face Dataset
    dataset = Dataset.from_list(data)

    # Cast the audio column to the `Audio` feature
    sampling_rate = 16000
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=sampling_rate))

    # Load the processor and model
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Preprocess the dataset
    def preprocess_batch(batch):
        # Extract audio features
        audio = batch["audio_filepath"]
        input_features = processor.feature_extractor(audio["array"], sampling_rate=sampling_rate).input_features[0]

        # Tokenize transcription with language prefix
        language_token = f"<|{batch['language_code']}|>"
        labels = processor.tokenizer(f"{language_token} {batch['text']}").input_ids

        batch["input_features"] = input_features
        batch["labels"] = labels
        return batch

    dataset = dataset.map(preprocess_batch, remove_columns=dataset.column_names)

    # Create a custom data collator
    data_collator = DataCollatorWithPadding(processor)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        num_train_epochs=3,
        predict_with_generate=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="no",
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        fp16=False,
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,  # Note: Keep tokenizer here for label decoding
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuning complete. Model and processor saved to: {output_dir}")