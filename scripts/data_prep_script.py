import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import json
import random
import pathlib
from dateutil import parser, tz
from pydantic import BaseModel, Field, ValidationError
from constants import DEFAULT_SYSTEM_PROMPT

# Paths
RAW_NAME = "fine_tune_40k_cleaned.csv"
RAW = "data/cleaned/" + RAW_NAME
OUT_DIR = pathlib.Path("data/processed")
OUT_DIR.mkdir(parents=True,exist_ok=True)

# Load raw CSV and select the source of truth columns
print(f"Loading data from {RAW}...")
df = pd.read_csv(RAW, dtype=str).fillna("")
need = ["text", "title", "intent", "description", "date", "startTime", "endTime", "location", "isAllDay"]
df = df[need]  # drop everything else

# Canonicalization functions
INTENT_MAP = {
    "add": "add", "create": "add",
    "update": "update", "edit": "update",
    "delete": "delete", "remove": "delete",
    "query": "query", "search": "query",
    "chitchat": "chitchat"  # Added chitchat intent if needed
}

def canon_intent(x):
    """Canonicalize event intent"""
    if not x:
        return "add"  # default to add
    return INTENT_MAP.get(x.strip().lower(), "add")

def canon_date(d):
    """Canonicalize date to ISO format: '2025-5-3' → '2025-05-03'"""
    if not d:
        return ""
    try:
        return parser.parse(d).date().isoformat()
    except (ValueError, TypeError):
        return ""

def canon_time(t: str) -> str:
    """Canonicalize time to ISO format with timezone:
       '5:30 PM' → '17:30:00+00:00'."""
    if not t or not t.strip():
        return ""
    try:
        dt = parser.parse(t)
        # Include timezone information in the output
        return dt.time().isoformat(timespec="seconds") + "+00:00"
    except (ValueError, TypeError):
        return ""

def canon_bool(x):
    """Canonicalize boolean to 0 or 1"""
    if not x:
        return 0
    return 1 if str(x).strip().lower() in {"1", "true", "yes"} else 0

# Apply canonicalization
print("Canonicalizing fields...")
df["intent"] = df["intent"].apply(canon_intent)
df["date"] = df["date"].apply(canon_date)
df["startTime"] = df["startTime"].apply(lambda x: canon_time(x) if x else "")
df["endTime"] = df["endTime"].apply(lambda x: canon_time(x) if x else "")
df["isAllDay"] = df["isAllDay"].apply(canon_bool)

# If isAllDay is 1, blank out startTime and endTime for consistency
df.loc[df["isAllDay"] == 1, ["startTime", "endTime"]] = ""

# Validate with Pydantic
class EventJSON(BaseModel):
    text: str = Field(default="")  # Added text field to prevent it from being dropped
    title: str = Field(default="")
    intent: str = Field(
        default="",
        pattern=r"^(add|update|delete|query|chitchat)$"
    )
    description: str = Field(default="")
    date: str = Field(
        default="",
        pattern=r"^\d{4}-\d{2}-\d{2}$|^$"
    )
    startTime: str = Field(
        default="",
        pattern=r"^\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$|^$"
    )
    endTime: str = Field(
        default="",
        pattern=r"^\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$|^$"
    )
    location: str = Field(default="")
    isAllDay: int = Field(default=0, ge=0, le=1)

print("Validating rows with Pydantic...")
good_rows = []
for i, row in df.iterrows():
    try:
        good_rows.append(EventJSON(**row.to_dict()))
    except ValidationError as e:
        print(f"⛔ drop row {i}: {e}")

df = pd.DataFrame([r.model_dump() for r in good_rows])
print(f"Kept {len(df)} valid rows out of original dataset")
print("Columns after validation:", df.columns.tolist())

# Convert to single-line JSON string
print("Converting to single-line JSON format...")
def make_json(row):
    # Exclude the text field from the JSON output
    ordered = {k: row[k] for k in EventJSON.model_json_schema()["properties"].keys() if k != 'text'}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))

df["response"] = df.apply(make_json, axis=1)

# Wrap in ChatML roles
def wrap(row):
    return (
        "<|system|>\n" +
        DEFAULT_SYSTEM_PROMPT.strip()+ "\n" +
        f"<|user|>\n{row['text'].strip()}\n" +
        "<|assistant|>\n" +
        f"{row['response']}\n" +
        "<|end|>"
    )

df["chatml"] = df.apply(wrap, axis=1)

# Train/validation split → JSONL files
print("Splitting into train/validation sets...")
records = df["chatml"].tolist()
random.seed(42)
random.shuffle(records)
split = int(0.9 * len(records))

for name, data in [(RAW_NAME + "_train.jsonl", records[:split]),
                   (RAW_NAME + "_val.jsonl", records[split:])]:
    output_path = OUT_DIR / name
    with open(output_path, "w", encoding="utf-8") as f:
        for text in data:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data)} examples to {output_path}")

print("Preprocessing complete!") 

# Write all examples to a single JSONL file
# print("Writing all examples to a single JSONL…")
# output_path = OUT_DIR / "all.jsonl"
# with open(output_path, "w", encoding="utf-8") as f:
#     for text in df["chatml"]:
#         # wrap each ChatML string in a top-level {"text": …} object
#         f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
# print(f"Wrote {len(df)} examples to {output_path}")