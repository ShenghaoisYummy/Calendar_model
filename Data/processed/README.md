# Calendar Assistant Fine-tuning Data Preparation

This directory contains the processed training data for fine-tuning the Calendar Assistant model.

## Files

- `train.jsonl`: Training data in JSONL format with ChatML formatting
- `val.jsonl`: Validation data in JSONL format with ChatML formatting

## Preprocessing Workflow

The data in these files has been processed using the following steps:

1. **Canonicalization**: All fields have been canonicalized to ensure consistent formats

   - Event types mapped to standard values (add, edit, delete, query, chitchat)
   - Dates converted to ISO-8601 format (YYYY-MM-DD)
   - Times converted to ISO-8601 format with timezone (HH:MM:SS+TZ:TZ)
   - Boolean values normalized to 0 or 1

2. **Validation**: Each record has been validated with Pydantic to ensure schema compliance

3. **JSON Formatting**: Each assistant response is formatted as a single-line JSON object with a consistent key order

4. **ChatML Wrapping**: Each example is wrapped in ChatML format with system, user, and assistant tags

5. **Loss Masking**: During tokenization, loss is only calculated on the assistant's response tokens

## Fine-tuning Usage

To fine-tune a model using this data:

```bash
python scripts/fine_tune_chatml.py \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train_file prep/train.jsonl \
  --val_file prep/val.jsonl \
  --output_dir ./output \
  --num_train_epochs 3 \
  --push_to_hub \
  --hub_model_id your-username/calendar-assistant
```

## Benefits of This Approach

1. **Consistent Output Format**: The model learns to produce single-line JSON objects with consistent key ordering
2. **Clean Data**: All values are canonicalized before training, ensuring the model learns proper formats
3. **Efficient Learning**: Loss masking ensures the model is only trained to generate the JSON response
4. **Schema Compliance**: Pydantic validation ensures all training examples follow the required schema
