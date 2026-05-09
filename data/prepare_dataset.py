"""
prepare_dataset.py
------------------
Converts raw API documentation into instruction-following training examples
in the format expected by SFTTrainer for Llama 3.1 fine-tuning.

Input format  (raw API doc entry):
  {"endpoint": "GET /users/{id}", "description": "...", "example": "..."}

Output format (Llama 3.1 chat template):
  {"text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>..."}
"""

import json
import os
import random
from pathlib import Path

SYSTEM_PROMPT = """You are an expert API assistant. Given a description of what a developer
wants to do, generate the correct API call with proper syntax, parameters, and a brief explanation."""

def format_training_example(entry: dict) -> dict:
    """
    Wraps a raw API doc entry into Llama 3.1 chat template format.

    Args:
        entry: dict with keys: endpoint, description, parameters, example_request, example_response

    Returns:
        dict with a single 'text' key in Llama 3.1 format
    """
    user_message = f"How do I {entry['description']}?"

    assistant_message = (
        f"To {entry['description']}, use the following API call:\n\n"
        f"**Endpoint:** `{entry['endpoint']}`\n\n"
        f"**Example Request:**\n```\n{entry['example_request']}\n```\n\n"
        f"**Example Response:**\n```json\n{entry['example_response']}\n```\n\n"
        f"**Parameters:**\n{entry.get('parameters', 'No additional parameters required.')}"
    )

    text = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_message}<|eot_id|>"
    )

    return {"text": text}


def prepare_dataset(input_path: str, output_dir: str, train_split: float = 0.9):
    """
    Reads raw API docs, formats them, and splits into train/test sets.

    Args:
        input_path:   Path to raw .jsonl file
        output_dir:   Where to save train.jsonl and test.jsonl
        train_split:  Fraction of data to use for training
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(input_path, "r") as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    formatted = [format_training_example(entry) for entry in raw_data]
    random.shuffle(formatted)

    split_idx = int(len(formatted) * train_split)
    train_data = formatted[:split_idx]
    test_data  = formatted[split_idx:]

    for filename, dataset in [("train.jsonl", train_data), ("test.jsonl", test_data)]:
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(dataset)} examples to {path}")


if __name__ == "__main__":
    prepare_dataset(
        input_path="data/sample_data.jsonl",
        output_dir="data/processed"
    )
