from datasets import load_dataset

def prepare_dataset(tokenizer, file_path: str, split: str = "train", val_size: float = 0.2):
    """
    Prepare dataset for training from a local file, and split into train/validation.
    
    Args:
        tokenizer: Tokenizer to use
        file_path: Path to the local dataset file (supports .csv, .json, .txt, etc.)
        split: 'train' or 'validation'
        val_size: Fraction of data to use as validation set
    Returns:
        Dataset: Prepared dataset
    """
    # Load full dataset
    full_dataset = load_dataset('csv', data_files=file_path)["train"]
    # Split into train/validation
    split_dataset = full_dataset.train_test_split(test_size=val_size, seed=42)
    if split == "train":
        dataset = split_dataset["train"]
    else:
        dataset = split_dataset["test"]

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