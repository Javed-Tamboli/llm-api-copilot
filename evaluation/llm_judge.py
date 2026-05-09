
"""
llm_judge.py
------------
Implements the LLM-as-a-Judge evaluation pattern.

A GPT-4o judge scores each model-generated API response on:
  - Correctness  : Is the endpoint and method right?
  - Completeness : Are all required parameters included?
  - Code Quality : Is the example syntactically valid?

Each dimension is scored 1–5. Final score = average across all 3.
"""

from openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_SYSTEM_PROMPT = """You are an expert API documentation evaluator.
You will be given:
  - A developer question about how to use an API
  - A reference answer (ground truth)
  - A model-generated answer to evaluate

Score the model answer on three dimensions, each from 1 to 5:

1. Correctness   - Is the HTTP method and endpoint path accurate?
2. Completeness  - Are all required parameters and headers included?
3. Code Quality  - Is the example request syntactically valid and runnable?

Return ONLY a JSON object in this exact format:
{
  "correctness": <1-5>,
  "completeness": <1-5>,
  "code_quality": <1-5>,
  "reasoning": "<one sentence explaining the scores>"
}"""


class JudgeScore(BaseModel):
    correctness: int
    completeness: int
    code_quality: int
    reasoning: str

    @property
    def average(self) -> float:
        return round((self.correctness + self.completeness + self.code_quality) / 3, 2)

    @property
    def normalized(self) -> float:
        """Returns score as 0–1 for easier comparison."""
        return round(self.average / 5, 2)


def judge(question: str, reference: str, prediction: str) -> JudgeScore:
    """
    Scores a single model prediction against the reference answer.

    Args:
        question:   The original developer question
        reference:  The ground truth answer
        prediction: The model's generated answer

    Returns:
        JudgeScore with per-dimension scores and reasoning
    """
    user_prompt = f"""Developer Question:
{question}

Reference Answer:
{reference}

Model Answer to Evaluate:
{prediction}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    import json
    data = json.loads(raw)
    return JudgeScore(**data)
