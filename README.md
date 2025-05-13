# Calendar Event Extraction and Scheduling Model

This project focuses on developing an AI model that can extract calendar event information from natural language text and handle various scheduling-related tasks. The model can understand different intents such as adding events, updating events, canceling events, and answering queries about existing events.

## Project Overview

The Calendar Model project provides an end-to-end solution for:

1. **Understanding user requests**: Processing natural language inputs about calendar events
2. **Intent recognition**: Identifying whether the user wants to add, update, query, or cancel events
3. **Entity extraction**: Pulling out important details like event title, date, time, location, etc.
4. **Response generation**: Creating appropriate responses to the user's requests

## File Structure

```
Calendar_model/
├── data/                    # Data directory
│   ├── raw/                 # Original unprocessed data
│   ├── processed/           # Processed data in JSONL format with ChatML wrapping
│   └── cleaned/             # Cleaned data ready for training
├── src/                     # Source code
│   ├── fine_tune.py         # Model fine-tuning utilities with ChatML support
│   └── evaluation.py        # Evaluation metrics and functions
├── scripts/                 # Executable scripts
│   ├── data_prep_script.py  # Script to prepare data with ChatML formatting
│   ├── fine_tune_script.py  # Script to run model fine-tuning with LoRA
│   └── evaluation_fine_tuned_script.py # Script to evaluate model performance
├── constants.py             # Global constants and configuration
└── requirements.txt         # Project dependencies
```

## How the Model Works

### 1. Data Preprocessing

The data preprocessing pipeline has been enhanced to support ChatML formatting, which improves the model's ability to understand context and generate appropriate responses:

- **Cleaning emoji texts**: Removing emojis and other special characters
- **Converting dates and times**: Standardizing to ISO 8601 format
  (YYYY-MM-DDThh:mm:ssZ)
- **Handling missing values**: Ensuring consistency in how missing data is
  represented
- **Formatting**: Making sure all data follows a consistent structure

- **Canonicalization**: All fields are standardized for consistency

  - **Dates**: Converted to ISO 8601 format (YYYY-MM-DD)
  - **Times**: Standardized to ISO format with timezone (HH:MM:SS+TZ:TZ)
  - **Intents**: Mapped to standard values (add, update, delete, query, chitchat)
  - **Boolean values**: Normalized to 0 or 1

- **Validation with Pydantic**: Each record is validated using a Pydantic model to ensure schema compliance

  - All fields must match their expected patterns (e.g., dates must be YYYY-MM-DD)
  - Records failing validation are dropped to maintain data quality

- **ChatML Formatting**: Each example is wrapped in ChatML format with system, user, and assistant tags

  - `<|system|>`: Contains the system prompt that guides the model's behavior
  - `<|user|>`: Contains the user's query
  - `<|assistant|>`: Contains the expected JSON response
  - `<|end|>`: Marks the end of the assistant's response

- **JSON Response Structure**: Each assistant response is formatted as a single-line JSON object with a consistent key order:
  ```json
  {
    "title": "Event Title",
    "intent": "add",
    "description": "Event description",
    "date": "2025-07-19",
    "startTime": "13:00:00+00:00",
    "endTime": "14:00:00+00:00",
    "location": "Location",
    "isAllDay": 0,
    "response": "Confirmation message"
  }
  ```

To run data preprocessing:

```bash
python scripts/data_prep_script.py
```

This will process the data and save the ChatML-formatted JSONL files to the `data/processed/` directory.

### 2. Fine-tuning with LoRA and ChatML

We use a technique called "Low-Rank Adaptation (LoRA)" to fine-tune a pre-trained language model. This allows us to efficiently adapt the model to our calendar task without having to retrain the entire model.

The fine-tuning process has been enhanced to leverage ChatML formatting and Parameter-Efficient Fine-Tuning (PEFT) with LoRA:

- Starts with a pre-trained model (TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Adds special parameter-efficient adapters that update a small subset of model
  parameters
- Trains on our calendar data to teach the model to understand and generate
  calendar information

- **ChatML Special Tokens**: The tokenizer is extended with special tokens for ChatML

  - `<|system|>`, `<|user|>`, `<|assistant|>`, and `<|end|>` tokens help the model understand dialogue structure

- **Loss Masking**: Training loss is calculated only on the assistant's response tokens

  - This focuses the learning on generating correct responses rather than memorizing prompts
  - Implemented using a custom dataset preparation function that masks prompt tokens with `-100`

- **Low-Rank Adaptation (LoRA)**: Efficient fine-tuning that updates only a small subset of parameters
  - Targets specific attention modules (q_proj, k_proj, v_proj, o_proj for LLaMA-based models)
  - Significantly reduces memory requirements and training time
  - Allows fine-tuning larger models on consumer hardware

To fine-tune the model:

```bash
python scripts/fine_tune_script.py --train_file data/processed/fine_tune_40k_cleaned.csv_train.jsonl --val_file data/processed/fine_tune_40k_cleaned.csv_val.jsonl --num_train_epochs 3
```

The fine-tuned model will be saved to the specified output directory and can optionally be pushed to Hugging Face Hub.

### 3. Evaluation

We evaluate the model's performance using a comprehensive set of metrics:

#### Evaluation Metrics

1. **Intent Recognition**:

   - **Accuracy**: Whether the model correctly identifies the user's intent (ADD, UPDATE, CANCEL, QUERY, CHITCHAT)
   - **F1 Score**: A balance of precision and recall for intent classification
   - We also generate a confusion matrix to see where the model might be confusing different intents

2. **Field-Specific Metrics**:

   - **Text fields** (title, description, response):

     - **Exact match**: 1.0 if the prediction exactly matches the reference, 0.0 otherwise
     - **ROUGE-L**: Measures the longest common subsequence between prediction and reference
     - **BLEU**: Measures n-gram overlap between prediction and reference
     - **Combined score**: A weighted average of the above metrics (typically 0.7 _ ROUGE-L + 0.3 _ BLEU)

   - **Date fields**:

     - **Format validity**: Whether the predicted date is in a valid format (1.0 if valid, 0.0 if not)
     - **Value match**: Whether the predicted date matches the reference date (1.0 if match, 0.0 if not)
     - **Overall**: 1.0 only if both format is valid AND values match

   - **Time fields** (startTime, endTime):

     - **Format validity**: Whether the predicted time is in a valid ISO format (1.0 if valid, 0.0 if not)
     - **Value match**: Whether the entire datetime matches the reference (1.0 if match, 0.0 if not)
     - **Time match**: Whether just the time part (ignoring date) matches (1.0 if match, 0.0 if not)
     - **Overall**: 1.0 only if both format is valid AND complete datetime values match

   - **Location field**:
     - **Exact match**: 1.0 if the prediction exactly matches the reference
     - **Partial match**: Score between 0.0-1.0 based on whether one location contains the other
     - **Word overlap**: Proportion of words that overlap between reference and prediction
     - **Overall**: The highest score among these three metrics

3. **Overall Scores**:
   - **Completeness**: Measures how many fields were predicted compared to reference
   - **Overall score**: A weighted average of all field scores, with weights adjusted based on the intent type

#### How Scores Are Calculated

Below is a detailed explanation of how evaluation scores are calculated for each field and how the overall score is determined:

**1. Text Fields (title, description, response)**

```
title:
  - exact_match: 0.000  (Binary: 1.0 if text matches exactly, 0.0 otherwise)
  - rouge_l: 0.200      (ROUGE-L score ranging from 0.0 to 1.0)
  - bleu: 0.024         (BLEU score ranging from 0.0 to 1.0)
  - combined_score: 0.147  (= 0.7 * rouge_l + 0.3 * bleu = 0.7 * 0.2 + 0.3 * 0.024)
```

The combined score is a weighted average of ROUGE-L (70%) and BLEU (30%), providing a balance between sequence matching and n-gram precision.

**2. Location Field**

```
location:
  - exact_match: 0.000    (Binary: 1.0 if locations match exactly, 0.0 otherwise)
  - partial_match: 0.000  (Score if one location contains the other, e.g., "Office" vs "Main Office")
  - word_overlap: 0.000   (Proportion of words that are shared between locations)
  - overall: 0.000        (The maximum value among the three metrics above)
```

The overall location score takes the highest value among the three metrics, allowing for partial credit even when locations aren't exact matches.

**3. Intent Field**

```
intent:
  - match: 0.000  (Binary: 1.0 if intents match exactly, 0.0 otherwise)
```

Intent matching is strict - the predicted intent must match the reference intent exactly.

**4. Date Field**

```
date:
  - format_valid: 1.000  (Binary: 1.0 if date is in valid format like YYYY-MM-DD, 0.0 otherwise)
  - value_match: 1.000   (Binary: 1.0 if date value matches reference, 0.0 otherwise)
  - overall: 1.000       (Binary: 1.0 only if both format_valid AND value_match are 1.0)
```

For dates, the model needs to get both the format right and the actual date value correct.

**5. Time Fields (startTime, endTime)**

```
startTime:
  - format_valid: 1.000   (Binary: 1.0 if time is in valid ISO format, 0.0 otherwise)
  - value_match: 0.000    (Binary: 1.0 if complete datetime matches, 0.0 otherwise)
  - time_match: 1.000     (Binary: 1.0 if just the time part matches, 0.0 otherwise)
  - overall: 0.000        (Binary: 1.0 only if both format_valid AND value_match are 1.0)
```

Time fields need to have valid format and match the reference time exactly to get full credit.

**6. Overall Score Calculation**

The overall score is a weighted average of all field scores based on importance weights assigned to each field. The weights vary depending on the intent type:

```
Overall score = Sum(field_weight × field_score) / Sum(field_weights)
```

For example:

- For ADD/UPDATE intents: date (0.5), time (0.5), title (0.2) have higher weights
- For CANCEL intents: title (0.5), date (0.4) are most important
- For QUERY intents: response (0.5) has the highest weight

An important issue to note is the JSON extraction pattern used in the evaluation process. The current implementation has two different regex patterns for extracting JSON from model outputs:

1. Multiplying each field's score by its weight
2. Summing these weighted scores
3. Dividing by the sum of all applicable weights

Fields with errors or missing values don't contribute to the score. The overall score provides a holistic measure of model performance across all evaluated fields, with emphasis on the fields most critical to each intent type.

#### Why These Metrics?

- **Intent recognition** is critical - a model that can't determine what the user wants will fail regardless of other capabilities
- **Different field types need different metrics**:

  - Text fields benefit from semantic similarity measures like ROUGE and BLEU, not just exact matching
  - Date/time fields need format validation and value matching
  - Locations might be expressed in various ways, so we use partial matching

- **Weighted scoring** recognizes that some fields are more important than others, and importance varies by intent type:
  - For ADD/UPDATE intents: date, time, and title are most critical
  - For CANCEL intents: title and date are most important
  - For QUERY intents: response quality is emphasized

To evaluate the model:

```bash
python scripts/evaluation_script.py --test_data data/cleaned/evaluation_schedule_response_en_20_cleaned.csv
```

This will generate detailed evaluation results with metrics for each sample and aggregated statistics.

### 7. Model Sharing

Once you're satisfied with your model's performance, you can share it on Hugging Face Hub:

```bash
python scripts/fine_tune_script.py --push_to_hub --hub_model_id your-username/calendar-assistant
```

This will upload your model to Hugging Face Hub, making it accessible to others. You'll need to set the `HF_TOKEN` environment variable with your Hugging Face token.

## Usage Tips

1. **Data quality matters**: The better your training data, the better your model will perform
2. **System prompts**: Using a good system prompt can significantly improve model performance
3. **Evaluation**: Always evaluate on a separate test set to get an accurate measure of performance
4. **Fine-tuning parameters**: Experiment with learning rate, batch size, and number of epochs
5. **ChatML formatting**: Proper formatting of system, user, and assistant messages is crucial for good results
6. **Loss masking**: Ensure that loss is only calculated on the assistant's response to focus learning

## Troubleshooting

If your model performs poorly (e.g., low overall score), check:

1. **Data preprocessing**: Ensure your data is properly cleaned, formatted with ChatML, and validated
2. **Model loading**: Make sure the tokenizer and model settings match between training and evaluation
3. **Training parameters**: Try different learning rates, batch sizes, or more training epochs
4. **System prompt**: A well-crafted system prompt can significantly improve performance
5. **JSON extraction**: Check that the model outputs are being correctly parsed during evaluation
6. **Response field**: Ensure the model is generating the response field, which is important for evaluation

### Common Issues and Solutions

1. **Missing response field in evaluation**:

   - Problem: The regex pattern used to extract JSON doesn't capture the response field
   - Solution: Update the regex pattern in `src/evaluation.py` to include the response field

2. **Empty response field in model outputs**:

   - Problem: The model is trained on data where the response field is empty
   - Solution: Ensure your training data includes meaningful responses, not just empty strings

3. **Inconsistent JSON structure**:
   - Problem: The model generates JSON with fields in different orders
   - Solution: Use the make_json function to ensure consistent field ordering in training data

## Security Notes

- Do not hardcode API tokens in your scripts
- Use environment variables for sensitive information like API keys

## Next Steps for Improvement

- Try different base models
- Collect more diverse training data
- Experiment with different tokenizer settings
- Implement more advanced data augmentation techniques
