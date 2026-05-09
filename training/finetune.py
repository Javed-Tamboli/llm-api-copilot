
"""
finetune.py
-----------
Fine-tunes Llama 3.1-8B-Instruct on API documentation data using:
  - QLoRA (4-bit quantization via bitsandbytes)
  - PEFT (LoRA adapters)
  - TRL SFTTrainer (supervised fine-tuning)
  - MLflow experiment tracking

Usage:
    python training/finetune.py
"""

import os
import yaml
import mlflow
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
from dotenv import load_dotenv

load_dotenv()

# ── Load config ──────────────────────────────────────────────────────────────
with open("training/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_ID      = cfg["model_id"]
OUTPUT_DIR    = cfg["output_dir"]
TRAIN_DATA    = cfg["train_data"]
MAX_SEQ_LEN   = cfg["max_seq_length"]
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


def load_qlora_model(model_id: str):
    """
    Loads Llama 3.1 in 4-bit precision (QLoRA) to fit on a single GPU.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — best for LLM weights
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,       # nested quantization saves ~0.4 bits/param
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )
    model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model):
    """
    Attaches LoRA adapters to the attention layers.
    Only adapters are trained — base model weights stay frozen.
    """
    lora_config = LoraConfig(
        r=cfg["lora_r"],                     # rank: controls adapter capacity
        lora_alpha=cfg["lora_alpha"],         # scaling factor
        target_modules=["q_proj", "v_proj"],  # which layers to adapt
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()       # shows what % of params are trained
    return model


def main():
    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files={"train": TRAIN_DATA}, split="train")
    print(f"Training on {len(dataset)} examples")

    # ── Model + Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.getenv("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = load_qlora_model(MODEL_ID)
    model = apply_lora(model)

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        learning_rate=cfg["learning_rate"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="mlflow",                  # auto-logs metrics to MLflow
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args,
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("llama31-api-copilot-finetune")

    with mlflow.start_run():
        mlflow.log_params({
            "model_id": MODEL_ID,
            "lora_r": cfg["lora_r"],
            "lora_alpha": cfg["lora_alpha"],
            "epochs": cfg["epochs"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "max_seq_length": MAX_SEQ_LEN,
            "train_examples": len(dataset),
        })

        print("\nStarting fine-tuning...")
        trainer.train()

        # Save LoRA adapter weights (small — only ~50MB vs 16GB full model)
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        mlflow.log_artifacts(OUTPUT_DIR, artifact_path="model")

        print(f"\nFine-tuning complete. Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
