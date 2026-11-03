"""Talking to a local LLM through Ollama's HTTP API (http://localhost:11434).

Ollama runs the model on this Mac. We send it a list of messages and get text back, or,
with a JSON schema, an answer that is guaranteed to have the shape we asked for.
"""

import os
from typing import Any

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

DEFAULT_OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen3:8b"
VISION_MODEL = "qwen2.5vl:7b"  # can look at images; beat gemma3:4b on our sea photos
EMBED_MODEL = "bge-m3"  # multilingual, works well for Turkish
EMBED_DIMENSIONS = 1024  # every bge-m3 vector has this many numbers
LLM_TIMEOUT_SECONDS = 120  # a local 8B model can take a while on long prompts


class LLMUnavailableError(RuntimeError):
    """Ollama is not running or the model is missing."""


class LLMOutputError(ValueError):
    """The model answered, but not in the requested shape (cut off, broken JSON, ...)."""


def ollama_url() -> str:
    load_dotenv()
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)


def chat[Output: BaseModel](
    messages: list[dict[str, Any]],  # a message may also carry "images" (base64)
    output_model: type[Output],
    model: str = CHAT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Output:
    """Send messages to the LLM and get back a validated Pydantic object.

    The JSON schema of `output_model` is sent as `format`, so Ollama forces the model to
    answer in exactly that shape. Pydantic then checks it again on our side.
    """
    payload = {
        "model": model,
        "messages": messages,
        "format": output_model.model_json_schema(),
        "stream": False,  # one complete answer instead of word-by-word chunks
        "think": False,  # qwen3 can "think out loud" first; we only want the answer
        "options": {
            "temperature": temperature,  # low = more predictable wording
            # A model can get stuck repeating itself; stop it instead of waiting forever
            "num_predict": max_tokens,
        },
    }
    try:
        response = requests.post(
            f"{ollama_url()}/api/chat", json=payload, timeout=LLM_TIMEOUT_SECONDS
        )
    except requests.ConnectionError as error:
        raise LLMUnavailableError(
            "Ollama is not running. Open the Ollama app or run `ollama serve`."
        ) from error

    if response.status_code == 404:
        raise LLMUnavailableError(
            f"Model {model} not found. Run `ollama pull {model}`."
        )
    response.raise_for_status()

    content = response.json()["message"]["content"]
    try:
        return output_model.model_validate_json(content)
    except ValidationError as error:
        raise LLMOutputError(
            f"{model} gave an invalid answer ({len(content)} characters): "
            f"{content[:120]}..."
        ) from error


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Turn texts into vectors. Texts with similar meaning get vectors that point the same way."""
    try:
        response = requests.post(
            f"{ollama_url()}/api/embed",
            json={"model": model, "input": texts},
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError as error:
        raise LLMUnavailableError(
            "Ollama is not running. Open the Ollama app or run `ollama serve`."
        ) from error

    if response.status_code == 404:
        raise LLMUnavailableError(
            f"Model {model} not found. Run `ollama pull {model}`."
        )
    response.raise_for_status()
    return response.json()["embeddings"]
