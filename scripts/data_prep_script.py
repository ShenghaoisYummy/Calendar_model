#!/usr/bin/env python
import pandas as pd
import re
from datetime import datetime
import emoji
import os
from pathlib import Path

def clean_emoji_text(text):
    """Remove emojis and clean text"""
    if pd.isna(text):
        return text
    
    # Remove emojis
    cleaned_text = emoji.replace_emoji(text, '')
    
    # Clean up any weird symbols or multiple spaces
    cleaned_text = re.sub(r'[^\w\s\.,;:!?\'"-]', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def convert_datetime_format(date, time):
    """Convert date and time to ISO 8601 format"""
    if pd.isna(date) or pd.isna(time):
        return None
    
    try:
        # Parse date and time
        date_str = str(date)
        time_str = str(time)
        
        # Ensure time has proper format (HH:MM)
        if not re.match(r'^\d{1,2}:\d{2}$', time_str):
            return None
            
        # Add seconds if not present
        if len(time_str.split(':')) == 2:
            time_str += ':00'
            
        # Combine date and time and convert to ISO format
        datetime_obj = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return datetime_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        print(f"Error converting date {date} and time {time}: {e}")
        return None

def clean_dataset(input_path, output_path):
    """
    Clean the dataset according to requirements:
    - Clean emoji texts in the text column
    - Convert date, start time, end time to ISO 8601 format
    - Clean NaN columns
    - Drop targetId column
    """
    print(f"Processing file: {input_path}")
    
    # Read the dataset
    df = pd.read_csv(input_path)
    
    # Make a copy of original data
    df_original = df.copy()
    
    # 1. Clean emoji texts in the text column
    print("Cleaning emoji texts...")
    df['text'] = df['text'].apply(clean_emoji_text)
    
    # 2. Convert date, startTime, endTime to ISO format
    print("Converting datetime formats...")
    
    # Create temporary storage for ISO dates and times
    date_iso = {}
    start_time_iso = {}
    end_time_iso = {}
    
    # Process each row to generate ISO formatted values
    for idx, row in df.iterrows():
        # Store original date for reference
        original_date = row['date'] if not pd.isna(row['date']) else None
        
        # Convert startTime to ISO
        if not pd.isna(row['startTime']) and original_date:
            iso_datetime = convert_datetime_format(original_date, row['startTime'])
            if iso_datetime:
                date_iso[idx] = iso_datetime.split('T')[0]  # Keep just the date part
                start_time_iso[idx] = iso_datetime  # Full ISO datetime
        
        # Convert endTime to ISO
        if not pd.isna(row['endTime']) and original_date:
            iso_datetime = convert_datetime_format(original_date, row['endTime'])
            if iso_datetime:
                if idx not in date_iso:  # If date wasn't set from startTime
                    date_iso[idx] = iso_datetime.split('T')[0]
                end_time_iso[idx] = iso_datetime  # Full ISO datetime
    
    # Apply the ISO formatted values to the dataframe
    for idx in date_iso:
        df.at[idx, 'date'] = date_iso[idx]
    
    for idx in start_time_iso:
        df.at[idx, 'startTime'] = start_time_iso[idx]
    
    for idx in end_time_iso:
        df.at[idx, 'endTime'] = end_time_iso[idx]
    
    # 3. Clean NaN values
    print("Cleaning NaN values...")
    
    # Replace NaN with empty strings for text columns
    text_columns = ['text', 'title', 'description', 'location', 'response']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Make sure isAllDay is 0 or 1
    if 'isAllDay' in df.columns:
        df['isAllDay'] = df['isAllDay'].fillna(0).astype(int)
    
    # 4. Drop targetId column
    if 'targetId' in df.columns:
        print("Dropping targetId column...")
        df = df.drop(columns=['targetId'])
    
    # Save the processed dataset
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    
    # Print some statistics
    print("\nProcessing Statistics:")
    print(f"Total rows: {len(df)}")
    print(f"Rows with cleaned text: {(df_original['text'] != df['text']).sum()}")
    print(f"Successfully converted dates: {len(date_iso)}")
    print(f"Successfully converted start times: {len(start_time_iso)}")
    print(f"Successfully converted end times: {len(end_time_iso)}")

def main():
    # Set paths
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    cleaned_dir = data_dir / "cleaned"
    
    # Create cleaned directory if it doesn't exist
    if not cleaned_dir.exists():
        cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    # Process test evaluation dataset
    eval_input_path = processed_dir / "evaluation_schedule_response_en_20.csv"
    eval_output_path = cleaned_dir / "evaluation_schedule_response_en_20_cleaned.csv"
    clean_dataset(eval_input_path, eval_output_path)

    # Process  evaluation dataset
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
