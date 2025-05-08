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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fine_tune import (
    setup_model_and_tokenizer,
    setup_lora_config,
    setup_wandb,
    save_checkpoint
)

from src.data_prep import prepare_dataset

def main():
    # Training configuration
    config = {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dataset_name": "Data/raw/schedule_response_en_50k.csv",  
        "output_dir": "outputs",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "logging_steps": 10,
        "save_steps": 100,
        "early_stopping_patience": 10,  # Number of evaluations with no improvement
        "early_stopping_threshold": 0.01,  # Minimum improvement needed to reset patience
    }
    
    # Setup wandb
    api_key = os.getenv("WANDB_API_KEY")
    setup_wandb("tinyllama-finetuning", config, api_key)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config["model_name"])
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Prepare datasets
    train_dataset = prepare_dataset(tokenizer, config["dataset_name"], split="train", val_size=0.2)
    val_dataset = prepare_dataset(tokenizer, config["dataset_name"], split="test", val_size=0.2)
    
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
    save_checkpoint(
        model,
        tokenizer,
        config["output_dir"],
        "final_checkpoint"
    )

if __name__ == "__main__":
    main() 