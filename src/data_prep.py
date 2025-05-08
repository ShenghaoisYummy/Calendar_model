from datasets import load_dataset

def prepare_dataset(tokenizer, file_path: str, split: str = "train"):
    """
    Prepare dataset for training from a local file.
    
    Args:
        tokenizer: Tokenizer to use
        file_path: Path to the local dataset file (supports .csv, .json, .txt, etc.)
        split: Dataset split to use
        
    Returns:
        Dataset: Prepared dataset
    """
    # Load dataset from local file
    dataset = load_dataset('csv', data_files=file_path, split=split)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset