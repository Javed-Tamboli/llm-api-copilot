# LLM-Powered API Copilot with Fine-Tuning

A developer tool that generates accurate API code from plain English questions.
Built by fine-tuning **Llama 3.1-8B** on domain-specific API documentation using
**QLoRA + PEFT**, evaluated against a GPT-4o zero-shot baseline using an
**LLM-as-a-Judge** harness, and deployed as a **streaming REST API** on AWS SageMaker.

> Fine-tuned model improved code generation accuracy by **34%** over zero-shot GPT-4o
> on held-out test cases.

---

## How It Works

```
Developer Question (plain English)
          │
          ▼
  ┌───────────────┐
  │  FastAPI App  │  ← streaming REST endpoint
  └───────┬───────┘
          │
          ▼
  ┌───────────────────────────┐
  │  Llama 3.1-8B + LoRA      │  ← fine-tuned on API docs
  │  (QLoRA, 4-bit precision) │
  └───────────────────────────┘
          │
          ▼
  Streamed API code + explanation
```

---

## Project Structure

```
llm-api-copilot/
│
├── data/
│   ├── prepare_dataset.py     # converts raw API docs → training format
│   └── sample_data.jsonl      # sample training examples
│
├── training/
│   ├── finetune.py            # QLoRA fine-tuning with PEFT + TRL
│   └── config.yaml            # hyperparameters (lora_r, lr, epochs, etc.)
│
├── evaluation/
│   ├── llm_judge.py           # LLM-as-a-Judge scorer (correctness, completeness, quality)
│   └── run_eval.py            # runs eval, compares vs GPT-4o, logs to MLflow
│
├── serving/
│   ├── app.py                 # FastAPI streaming REST API
│   └── model_loader.py        # loads fine-tuned model once at startup
│
├── mlflow_tracking/
│   └── logger.py              # centralised MLflow experiment logging
│
├── Dockerfile                 # containerised deployment
├── requirements.txt
└── .env.example
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Base Model | Llama 3.1-8B-Instruct (Meta) |
| Fine-Tuning | QLoRA (4-bit), PEFT LoRA adapters, TRL SFTTrainer |
| Evaluation | LLM-as-a-Judge (GPT-4o), MLflow experiment tracking |
| Serving | FastAPI, streaming responses, AWS SageMaker |
| DevOps | Docker, CI/CD |

---

## Key Components

### 1. Fine-Tuning (QLoRA + PEFT)
Llama 3.1-8B is loaded in **4-bit precision** using bitsandbytes to fit on a single GPU.
**LoRA adapters** are attached only to the attention layers — the base model stays frozen.
Only ~0.5% of parameters are trained, reducing memory use by ~75% vs full fine-tuning.

### 2. LLM-as-a-Judge Evaluation
Each model output is scored on three dimensions by a GPT-4o judge:
- **Correctness** — Is the HTTP method and endpoint path right?
- **Completeness** — Are all required parameters included?
- **Code Quality** — Is the example syntactically valid?

Scores are 1–5 per dimension, averaged and normalised to 0–1.

### 3. Streaming REST API
The fine-tuned model is served via **FastAPI** with token-by-token streaming
using `TextIteratorStreamer` — so developers see output as it generates,
not after a full wait.

---

## Results

| Metric | Value |
|---|---|
| Fine-tuned model avg score | 0.91 / 1.0 |
| GPT-4o zero-shot baseline | 0.68 / 1.0 |
| Improvement | **+34%** |
| API response latency | streaming, first token < 1s |
| Test set size | 20% held-out split |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/javedtamboli/llm-api-copilot
cd llm-api-copilot
pip install -r requirements.txt
```

### 2. Add your API keys

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY and HF_TOKEN in .env
```

### 3. Prepare data

```bash
python data/prepare_dataset.py
```

### 4. Fine-tune

```bash
python training/finetune.py
```

### 5. Evaluate

```bash
python evaluation/run_eval.py
```

### 6. Run the API

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t api-copilot .
docker run -p 8000:8000 --env-file .env api-copilot
```

### 7. Call the API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I fetch a user by their ID?"}'
```

---

## MLflow Tracking

Training and evaluation runs are tracked with MLflow.
To view the experiment dashboard locally:

```bash
mlflow ui
# Open http://localhost:5000
```

Logged per run: hyperparameters, train loss, eval scores, model artifacts.

---

## Notes

- Model weights are excluded from this repo (too large for GitHub).
  After fine-tuning, push your adapter to HuggingFace Hub:
  `model.push_to_hub("your-username/llama31-api-copilot-adapter")`
- A HuggingFace account + access token is required to download Llama 3.1.
  Request access at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
