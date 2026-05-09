"""
model_loader.py
---------------
Loads the fine-tuned Llama 3.1 model (base + LoRA adapter) once at startup
and keeps it in memory for fast inference across API requests.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

ADAPTER_PATH  = os.getenv("ADAPTER_PATH", "models/llama31-api-copilot-adapter")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")

# Module-level singletons — loaded once, reused across all requests
_model     = None
_tokenizer = None


def get_model_and_tokenizer():
    """
    Returns the loaded model and tokenizer.
    Loads from disk on first call, returns cached instance on subsequent calls.
    """
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        print(f"Loading model from: {ADAPTER_PATH}")

        _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        _tokenizer.pad_token = _tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=os.getenv("HF_TOKEN")
        )
        _model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        _model.eval()
        print("Model loaded and ready.")

    return _model, _tokenizer
