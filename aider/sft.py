import logging
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

def prepare_dataset(dataset, tokenizer, max_length):
    logger.info(f"Preparing dataset with max_length={max_length}")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    logger.info(f"Dataset preparation complete. Size: {len(tokenized_dataset)}")
    return tokenized_dataset

def load_chat_dataset(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the content into separate conversations
    conversations = content.split('\n\n')
    
    # Process each conversation
    processed_data = []
    for conversation in conversations:
        lines = conversation.strip().split('\n')
        text = ' '.join(lines)
        processed_data.append({"text": text})
    
    return processed_data

def train_sft(model_name, dataset_path, output_dir, num_train_epochs=3, per_device_train_batch_size=8, max_length=512):
    logger.info(f"Starting SFT training with model: {model_name}, dataset: {dataset_path}")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    logger.info("Tokenizer and model loaded successfully")

    # Load and prepare dataset
    logger.info(f"Loading dataset: {dataset_path}")
    dataset = load_chat_dataset(dataset_path)
    dataset = prepare_dataset(dataset, tokenizer, max_length)
    logger.info("Dataset loaded and prepared successfully")

    # Define LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    # Define training arguments
    logger.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=1000,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        fp16=True,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
    )

    # Initialize Trainer
    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")

    # Save the final model
    logger.info(f"Saving the final model to {output_dir}")
    trainer.save_model()
    logger.info("Model saved successfully")

if __name__ == "__main__":
    model_name = "mistralai/Codestral-22B"  # or any other pre-trained model
    dataset_path = "/home/user/seb/embdata/chat_dataset.md"
    output_dir = "./sft_output"
    
    logger.info("Starting main script")
    train_sft(model_name, dataset_path, output_dir)
    logger.info("Main script completed")
