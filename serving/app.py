
"""
app.py
------
FastAPI server exposing the fine-tuned Llama 3.1 API Copilot as a
streaming REST endpoint. Responses stream token-by-token so the
client sees output immediately, without waiting for full generation.

Endpoints:
  POST /generate         - streaming code generation
  POST /generate/full    - non-streaming, returns full response
  GET  /health           - health check
"""

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer
from threading import Thread
from serving.model_loader import get_model_and_tokenizer

app = FastAPI(
    title="LLM API Copilot",
    description="Fine-tuned Llama 3.1 for API code generation",
    version="1.0.0"
)

# ── Request / Response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    question: str
    max_new_tokens: int = 300
    temperature: float = 0.1

class GenerateResponse(BaseModel):
    answer: str
    model: str = "llama-3.1-8b-api-copilot"


# ── Helper ────────────────────────────────────────────────────────────────────

def build_prompt(question: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{question}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "llama-3.1-8b-api-copilot"}


@app.post("/generate")
def generate_streaming(request: GenerateRequest):
    """
    Streams the model response token-by-token using Server-Sent Events.
    The client receives text as it's generated — no waiting for full output.
    """
    model, tokenizer = get_model_and_tokenizer()

    prompt  = build_prompt(request.question)
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Run generation in a background thread so we can stream from the main thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def token_stream():
        for token in streamer:
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post("/generate/full", response_model=GenerateResponse)
def generate_full(request: GenerateRequest):
    """
    Non-streaming endpoint — returns the complete answer at once.
    Useful for programmatic clients that don't support streaming.
    """
    model, tokenizer = get_model_and_tokenizer()

    prompt = build_prompt(request.question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer  = decoded[len(prompt):].strip()

    return GenerateResponse(answer=answer)
