import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import wandb
from typing import Dict, Any, Optional
import os
os.environ["WANDB_API_KEY"] = "your_wandb_api_key"


def setup_model_and_tokenizer(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple:
    """
    Setup model and tokenizer for training.
    
    Args:
        model_name: Name of the model to load
        device: Device to load the model on
        
    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def setup_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list = ["q_proj", "v_proj"]
) -> LoraConfig:
    """
    Setup LoRA configuration for fine-tuning.
    
    Args:
        r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout
        target_modules: Target modules for LoRA
        
    Returns:
        LoraConfig: LoRA configuration
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

def setup_wandb(
    project_name: str,
    config: dict = None,
    api_key: str = None
) -> None:
    """
    Setup Weights & Biases logging.
    Args:
        project_name: Name of the W&B project
        config: Configuration dictionary to log
        api_key: Your wandb API key (optional)
    """
    if api_key:
        wandb.login(key=api_key)
    wandb.init(
        project=project_name,
        config=config or {}
    )

def save_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    checkpoint_name: str
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Directory to save the checkpoint
        checkpoint_name: Name of the checkpoint
    """
    model.save_pretrained(f"{output_dir}/{checkpoint_name}")
    tokenizer.save_pretrained(f"{output_dir}/{checkpoint_name}") 