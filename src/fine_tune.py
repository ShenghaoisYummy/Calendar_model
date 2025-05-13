import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import wandb
from typing import Dict, Any, Optional
import os



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
    
    # Add ChatML special tokens if not present
    chatml_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    special_tokens_dict = {"additional_special_tokens": chatml_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    # Resize token embeddings if new tokens were added
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    
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


def setup_tokenizer(model_name):
    """Set up the tokenizer with ChatML special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add ChatML special tokens if not present
    chatml_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": chatml_tokens})
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def prepare_dataset(tokenizer, file_path, max_length=1024):
    """
    Prepare dataset with loss masking - only calculate loss on assistant responses.
    """
    dataset = load_dataset("json", data_files=file_path, split="train")
    
    def tokenize_with_loss_mask(examples):
        texts = examples["text"]
        tokenized_inputs = []
        
        for text in texts:
            # Find the index where the assistant starts responding
            assistant_idx = text.find("<|assistant|>")
            if assistant_idx == -1:
                # Skip examples without assistant tag
                continue
                
            # Tokenize the full text
            encodings = tokenizer(
                text, 
                truncation=True, 
                max_length=max_length,
                padding="max_length",  # Add padding to ensure consistent lengths
                return_tensors=None    # Return as Python lists, not tensors
            )
            
            # Tokenize just the prompt part to determine its length
            prompt_tokens = tokenizer(text[:assistant_idx], add_special_tokens=False).input_ids
            n_prompt = len(prompt_tokens)
            
            # Create labels: -100 for prompt tokens (no loss), actual token IDs for response tokens
            labels = [-100] * min(n_prompt, max_length)  # Ensure n_prompt doesn't exceed max_length
            
            # Add the remaining labels, ensuring we don't exceed max_length
            remaining_length = max_length - len(labels)
            if remaining_length > 0 and n_prompt < len(encodings["input_ids"]):
                labels.extend(encodings["input_ids"][n_prompt:min(len(encodings["input_ids"]), n_prompt + remaining_length)])
            
            # Pad labels to max_length with -100
            if len(labels) < max_length:
                labels.extend([-100] * (max_length - len(labels)))
            elif len(labels) > max_length:
                labels = labels[:max_length]
                
            tokenized_inputs.append({
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels
            })
        
        # Convert list of dicts to dict of lists
        if not tokenized_inputs:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        result = {k: [d[k] for d in tokenized_inputs] for k in tokenized_inputs[0].keys()}
        return result
    
    tokenized_dataset = dataset.map(
        tokenize_with_loss_mask,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def setup_model_for_training(model_name, tokenizer, args):
    """Set up the model with LoRA configuration."""
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Update model config with tokenizer changes
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    target_modules = ["q_proj", "v_proj"]  # Default for most models
    if "llama" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

