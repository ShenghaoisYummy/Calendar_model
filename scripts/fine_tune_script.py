import sys
import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
import wandb
from huggingface_hub import HfApi

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fine_tune import setup_tokenizer, prepare_dataset, setup_model_for_training
# Set environment variables from constants if available
try:
    from constants import WANDB_API_KEY, HF_TOKEN
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ["HF_TOKEN"] = HF_TOKEN
except (ImportError, AttributeError):
    print("Warning: constants.py not found or missing keys. Using environment variables.")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="Base model to fine-tune")
    parser.add_argument("--train_file", type=str, default="Data/processed/calendar_dataset_fine_tuning_40k_cleaned.csv_train.jsonl", 
                        help="Path to training data")
    parser.add_argument("--val_file", type=str, default="Data/processed/calendar_dataset_fine_tuning_40k_cleaned.csv_val.jsonl", 
                        help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, 
                        help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, 
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, 
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout rate")
    parser.add_argument("--push_to_hub", action="store_true", default=True,
                        help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, 
                        help="Model ID for Hugging Face Hub")
    parser.add_argument("--wandb_project", type=str, default="calendar-assistant-v6", 
                        help="Weights & Biases project name")
    return parser.parse_args()

def train(args):
    """Main training function."""
    # Setup tokenizer with ChatML tokens
    tokenizer = setup_tokenizer(args.model_name)
    
    # Prepare datasets with loss masking
    print(f"Loading and processing training data from {args.train_file}...")
    train_dataset = prepare_dataset(tokenizer, args.train_file, args.max_seq_length)
    
    print(f"Loading and processing validation data from {args.val_file}...")
    eval_dataset = prepare_dataset(tokenizer, args.val_file, args.max_seq_length)
    
    # Setup model with LoRA
    model = setup_model_for_training(args.model_name, tokenizer, args)
    
    # Initialize Weights & Biases
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(project=args.wandb_project)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="wandb" if "WANDB_API_KEY" in os.environ else None,
        load_best_model_at_end=True,
        fp16=True,
    )

    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False,
            pad_to_multiple_of=8  # Ensure padding to multiple of 8 for efficiency
        ),
        # callbacks=[early_stopping_callback]
    )
    
    # Validate datasets before training
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")
    
    # Check a sample to verify shape
    sample = train_dataset[0]
    print(f"Sample input_ids length: {len(sample['input_ids'])}")
    print(f"Sample attention_mask length: {len(sample['attention_mask'])}")
    print(f"Sample labels length: {len(sample['labels'])}")
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    
    full_repo_name = "ShenghaoYummy/calendar-assistant_v6"

    # Push to Hub if requested
    if args.push_to_hub and "HF_TOKEN" in os.environ:
        print(f"Pushing model to Hugging Face Hub as {args.hub_model_id}")
        api = HfApi(token=os.environ["HF_TOKEN"])
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=full_repo_name,
                private=False,
                exist_ok=True
            )
            print(f"Repository {full_repo_name} created or already exists")
        except Exception as e:
            print(f"Error creating repository: {e}")
        
        # Upload model files
        model.push_to_hub(full_repo_name)
        tokenizer.push_to_hub(full_repo_name)
        print("Model successfully pushed to Hub")

if __name__ == "__main__":
    args = parse_args()
    train(args) 