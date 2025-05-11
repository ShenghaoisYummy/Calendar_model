# Default system prompt for the model
DEFAULT_SYSTEM_PROMPT = """You are a scheduling assistant.
Read the user's request and output **only** a JSON object
with the following exact keys, in this order (no extra text before/after):
'title', 'type', 'description', 'date', 'startTime', 'endTime',
'location', 'isAllDay', 'response'
• 'date', 'startTime', 'endTime' must be RFC3339/ISO-8601 strings.
• 'date' must be a valid date.
• 'date' must be only the date, not the time.
• If 'endTime' is not provided, predict based on other information.
• Based on the user's request, strictly choose intent from add, cancel, update, query, chitchat. 
    - For creating a new event use "add".
    - For cancelling an event use "cancel".
    - For editing an event use "update".
    - For querying an event use "query".
    - For chatting use "chitchat".
• 'isAllDay' = 1 if the request is an all-day event, else 0.
• 'response' is how you would politely confirm the action to the user.
Output NOTHING except that JSON object."""

EVALUATION_DATA_PATH = "Data/cleaned/evaluation_schedule_response_en_20_cleaned.csv"
FINE_TUNE_DATA_PATH = "Data/cleaned/evaluation_schedule_response_en_20_cleaned.csv"

MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Environment variable names for API keys
ENV_WANDB_API_KEY = "WANDB_API_KEY"
ENV_HF_API_TOKEN = "HF_TOKEN"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"

# Default training parameters
DEFAULT_TRAINING_ARGS = {
    "output_dir": "outputs",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 200,
    "early_stopping_patience": 10,
    "early_stopping_threshold": 0.01,
}

# Hugging Face args
DEFAULT_HF_ARGS = {
    "commit_message": "Upload fine-tuned calendar model",
    "private": False,
}

# WandB project name
WANDB_PROJECT_NAME = "calendar-model-finetuning"
WANDB_EVAL_PROJECT_NAME = "calendar-model-evaluation"
