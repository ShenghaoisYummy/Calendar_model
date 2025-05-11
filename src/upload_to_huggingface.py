#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Upload Fine-tuned Model to Hugging Face Hub
===========================================

This script uploads the fine-tuned model to the Hugging Face Model Hub.
"""

import os
import argparse
import sys
from huggingface_hub import HfApi, create_repo, upload_folder
from getpass import getpass
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import ENV_HF_API_TOKEN, DEFAULT_HF_ARGS, UPLOAD_MODEL_PATH

def push_to_hub(
    model_path: str,
    repo_name: str,
    token: str = None,
    private: bool = DEFAULT_HF_ARGS["private"],
    commit_message: str = DEFAULT_HF_ARGS["commit_message"],
    organization: str = None,
    add_readme: bool = True
) -> str:
    """
    Push a model to the Hugging Face Model Hub.
    
    Args:
        model_path: Path to the model directory
        repo_name: Name of the repository to create or update
        token: HF API token (will prompt if not provided)
        private: Whether the repository should be private
        commit_message: Commit message for the upload
        organization: Organization to upload to (if any)
        add_readme: Whether to add a README.md file with model information
        
    Returns:
        str: URL of the model on the Hugging Face Hub
    """
    # Get token if not provided
    if not token:
        token = os.environ.get(ENV_HF_API_TOKEN)
        if not token:
            token = getpass(f"Enter your Hugging Face API token (or set {ENV_HF_API_TOKEN} environment variable): ")
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create full repo name
    full_repo_name = f"{organization}/{repo_name}" if organization else repo_name
    
    # Check if repo exists
    try:
        api.repo_info(full_repo_name, repo_type="model")
        print(f"Repository {full_repo_name} already exists. Updating...")
    except Exception:
        print(f"Creating new repository: {full_repo_name}")
        create_repo(full_repo_name, repo_type="model", private=private, token=token)
    
    # Add README.md if requested
    if add_readme and not os.path.exists(os.path.join(model_path, "README.md")):
        with open(os.path.join(model_path, "README.md"), "w") as f:
            f.write(f"# {repo_name}\n\n")
            f.write("This is a fine-tuned model for calendar event extraction and scheduling.\n\n")
            f.write("## Model Description\n\n")
            f.write("This model was fine-tuned to understand calendar requests and extract structured data including:\n")
            f.write("- Event title\n")
            f.write("- Intent (add, cancel, query, etc.)\n")
            f.write("- Date and time information\n")
            f.write("- Location\n")
            f.write("- Description\n\n")
            f.write("## Usage\n\n")
            f.write("```python\n")
            f.write("from transformers import AutoTokenizer, AutoModelForCausalLM\n\n")
            f.write("# Load model and tokenizer\n")
            f.write(f"model_name = \"{full_repo_name}\"\n")
            f.write("tokenizer = AutoTokenizer.from_pretrained(model_name)\n")
            f.write("model = AutoModelForCausalLM.from_pretrained(model_name)\n\n")
            f.write("# Example usage\n")
            f.write("prompt = \"Schedule a meeting with John on Friday at 2pm\"\n")
            f.write("inputs = tokenizer(prompt, return_tensors=\"pt\")\n")
            f.write("outputs = model.generate(**inputs)\n")
            f.write("response = tokenizer.decode(outputs[0], skip_special_tokens=True)\n")
            f.write("print(response)\n")
            f.write("```\n")
    
    # Upload the model to the Hub
    upload_folder(
        folder_path=model_path,
        repo_id=full_repo_name,
        repo_type="model",
        commit_message=commit_message,
        token=token
    )
    
    # Return the URL
    repo_url = f"https://huggingface.co/{full_repo_name}"
    print(f"Model successfully uploaded to {repo_url}")
    return repo_url