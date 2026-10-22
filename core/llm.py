"""Provider-agnostic LLM helper.

Uses the OpenAI Python SDK, which also works with any OpenAI-compatible endpoint
(Gemini, Groq, OpenRouter, a local Ollama server, ...). Configure with env vars:
  LLM_API_KEY, LLM_BASE_URL (optional for OpenAI), LLM_MODEL
"""
import json
import os
import re

from openai import OpenAI


def get_client() -> OpenAI:
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or None)


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    model = os.getenv("LLM_MODEL")
    if not model:
        raise RuntimeError("LLM_MODEL is not set. See .env.example.")
    resp = get_client().chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return resp.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences and chatter."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
