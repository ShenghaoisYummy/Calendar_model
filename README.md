# TinyLlama Fine-tuning Project

This project provides scripts and utilities for fine-tuning the TinyLlama-1.1B-Chat model using LoRA (Low-Rank Adaptation).

## Project Structure

```
.
├── README.md
├── requirements.txt
├── scripts/
│   └── train.py
└── src/
    └── utils.py
```

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up Weights & Biases (optional but recommended):

```bash
wandb login
```

## Usage

1. Modify the training configuration in `scripts/train.py`:

   - Update `dataset_name` to your dataset
   - Adjust training parameters as needed

2. Run the training script:

```bash
python scripts/train.py
```

## Features

- LoRA fine-tuning for efficient training
- Weights & Biases integration for experiment tracking
- Automatic mixed precision training
- Gradient accumulation for larger effective batch sizes
- Checkpoint saving and loading

## Configuration

The main training configuration can be found in `scripts/train.py`. Key parameters include:

- `model_name`: The base model to fine-tune
- `dataset_name`: Your training dataset
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `learning_rate`: Learning rate for training
- `warmup_steps`: Number of warmup steps

## Output

The fine-tuned model checkpoints will be saved in the `outputs` directory. Each checkpoint includes:

- Model weights
- Tokenizer files
- Training configuration
