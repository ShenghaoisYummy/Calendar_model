import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import get_peft_model
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fine_tune import (
    setup_model_and_tokenizer,
    setup_lora_config,
    setup_wandb,
    save_checkpoint
)
from constants import (
    DOWNLOAD_MODEL_PATH, 
    FINE_TUNE_DATA_PATH, 
    DEFAULT_TRAINING_ARGS,
    ENV_WANDB_API_KEY, 
    ENV_HF_API_TOKEN,
    WANDB_PROJECT_NAME,
    DEFAULT_HF_ARGS
)
from src.data_prep import prepare_dataset
from upload_to_huggingface import push_to_hub

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a model for calendar event extraction")
    parser.add_argument("--download_model_name", type=str, default=DOWNLOAD_MODEL_PATH, help="Base model to fine-tune")
    parser.add_argument("--dataset_path", type=str, default=FINE_TUNE_DATA_PATH, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_TRAINING_ARGS["output_dir"], help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_TRAINING_ARGS["num_train_epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_TRAINING_ARGS["per_device_train_batch_size"], help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_TRAINING_ARGS["learning_rate"], help="Learning rate")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub after training")
    parser.add_argument("--hf_model_name", type=str, default=None, help="Name for the model on Hugging Face Hub")
    parser.add_argument("--hf_organization", type=str, default=None, help="Hugging Face organization")
    parser.add_argument("--private", action="store_true", default=DEFAULT_HF_ARGS["private"], help="Whether the HF repository should be private")
    args = parser.parse_args()

    # Training configuration
    config = {
        "download_model_name": args.download_model_name,
        "dataset_name": args.dataset_path,
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": DEFAULT_TRAINING_ARGS["gradient_accumulation_steps"],
        "learning_rate": args.learning_rate,
        "warmup_steps": DEFAULT_TRAINING_ARGS["warmup_steps"],
        "logging_steps": DEFAULT_TRAINING_ARGS["logging_steps"],
        "save_steps": DEFAULT_TRAINING_ARGS["save_steps"],
        "early_stopping_patience": DEFAULT_TRAINING_ARGS["early_stopping_patience"],
        "early_stopping_threshold": DEFAULT_TRAINING_ARGS["early_stopping_threshold"],
    }
    
    # Setup wandb using environment variable
    wandb_api_key = os.getenv(ENV_WANDB_API_KEY)
    if not wandb_api_key:
        print(f"Warning: {ENV_WANDB_API_KEY} environment variable not set. W&B logging may not work.")
    setup_wandb(WANDB_PROJECT_NAME, config, wandb_api_key)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config["download_model_name"])
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Prepare datasets
    train_dataset = prepare_dataset(tokenizer, config["dataset_name"], split="train", val_size=0.1)
    val_dataset = prepare_dataset(tokenizer, config["dataset_name"], split="test", val_size=0.1)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        eval_strategy="steps",
        eval_steps=config["save_steps"], 
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=True,
        report_to="wandb",
    )
    
    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_threshold=config["early_stopping_threshold"]
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[early_stopping_callback],  # Add early stopping callback
    )
    
    # Train
    trainer.train()
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(config["output_dir"], "final_checkpoint")
    save_checkpoint(
        model,
        tokenizer,
        config["output_dir"],
        "final_checkpoint"
    )
    
    print(f"Model saved to {final_checkpoint_path}")
    
    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        # Get HF token from environment variable
        hf_token = os.getenv(ENV_HF_API_TOKEN)
        if not hf_token:
            print(f"Warning: {ENV_HF_API_TOKEN} environment variable not set. Will prompt for token.")
        
        if not args.hf_model_name:
            # Create a default name if not provided
            model_base_name = os.path.basename(args.download_model_name.split('/')[-1])
            args.hf_model_name = f"{model_base_name}-calendar-finetuned"
            
        # Push model to Hub
        print(f"Pushing model to Hugging Face Hub as {args.hf_model_name}...")
        repo_url = push_to_hub(
            model_path=final_checkpoint_path,
            repo_name=args.hf_model_name,
            token=hf_token,
            organization=args.hf_organization,
            private=args.private,
            commit_message=DEFAULT_HF_ARGS["commit_message"]
        )
        print(f"Model successfully uploaded to: {repo_url}")
    else:
        print("Skipping upload to Hugging Face Hub. Use --push_to_hub to enable.")

if __name__ == "__main__":
    main() 