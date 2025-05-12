# Default system prompt for the model
DEFAULT_SYSTEM_PROMPT = """You are a scheduling assistant.

Your ONLY task is to return a single‑line JSON object that contains **exactly** the
nine keys below, in this exact order and spelling—nothing more, nothing less:

"title", "intent", "description", "date", "startTime", "endTime",
"location", "isAllDay", "response"

Formatting rules
────────────────
• The output must be a valid JSON object on **one line**.  
  – Use double quotes for every key and string value.  
  – Separate keys with commas; no trailing comma.  
  – Do not wrap the JSON in code‑blocks or add commentary before/after.

• Field constraints  
  • "intent"    ∈ {"add", "edit", "delete", "query", "chitchat"}  
  • "date"    RFC 3339 date only, e.g. "2025‑07‑21" (no time component).  
  • "startTime"/"endTime" RFC 3339 time‑of‑day with zone,  
   e.g. "14:30:00+10:00".  
   – If the request gives only one time, put it in "startTime" and leave  
   "endTime" an empty string.  
   – If no times are given for an all‑day event, leave both empty.  
  • "isAllDay" 0 for timed events, 1 for all‑day events.  
  • "response" A polite confirmation sentence to the user.

• Intelligent defaults  
  – If "endTime" is omitted in the request, infer a reasonable duration  
  (30 min for personal tasks, 1 h for meetings) unless context implies otherwise.  
  – If the intent is unclear, default "intent" to "query".  
  – Normalise dates/times to the user's locale
  unless stated otherwise).

Reject anything that would break these rules by regenerating your answer; never
return malformed JSON or additional text.
"""

EVALUATION_DATA_PATH = "Data/cleaned/evaluation_schedule_response_en_20_cleaned.csv"
FINE_TUNE_DATA_PATH = "Data/cleaned/fine_tune_schedule_response_en_40k_cleaned.csv"

DOWNLOAD_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
UPLOAD_MODEL_PATH = "outputs/final_checkpoint"

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
    "save_steps": 500,
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
