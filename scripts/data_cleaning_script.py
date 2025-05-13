#!/usr/bin/env python
import pandas as pd
import re
from datetime import datetime
import emoji
import os
from pathlib import Path
from data_cleaning import clean_dataset

def main():
    # Set paths
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    cleaned_dir = data_dir / "cleaned"
    
    # Create cleaned directory if it doesn't exist
    if not cleaned_dir.exists():
        cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean fine-tuning dataset
    eval_input_path = processed_dir / "schedule_response_en_50k.csv"
    eval_output_path = cleaned_dir / "schedule_response_en_50k_cleaned.csv"
    clean_dataset(eval_input_path, eval_output_path)

    # Clean evaluation dataset
    eval_input_path = processed_dir / "evaluation_schedule_response_en_10k.csv"
    eval_output_path = cleaned_dir / "evaluation_schedule_response_en_10k_cleaned.csv"
    clean_dataset(eval_input_path, eval_output_path)
    
    # Process fine-tuning dataset if it exists
    fine_tune_input_path = processed_dir / "fine_tune_schedule_response_en_40k.csv"
    if fine_tune_input_path.exists():
        fine_tune_output_path = cleaned_dir / "fine_tune_schedule_response_en_40k_cleaned.csv"
        clean_dataset(fine_tune_input_path, fine_tune_output_path)
    else:
        print(f"Fine-tuning dataset not found at {fine_tune_input_path}")

if __name__ == "__main__":
    main()
