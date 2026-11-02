"""Talking to a local LLM through Ollama's HTTP API (http://localhost:11434).

Ollama runs the model on this Mac. We send it a list of messages and get text back, or,
with a JSON schema, an answer that is guaranteed to have the shape we asked for.
"""

import os

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

DEFAULT_OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "qwen3:8b"
LLM_TIMEOUT_SECONDS = 120  # a local 8B model can take a while on long prompts


class LLMUnavailableError(RuntimeError):
    """Ollama is not running or the model is missing."""


def ollama_url() -> str:
    load_dotenv()
    return os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)


def chat[Output: BaseModel](
    messages: list[dict[str, str]],
    output_model: type[Output],
    model: str = CHAT_MODEL,
    temperature: float = 0.2,
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
        "options": {"temperature": temperature},  # low = more predictable wording
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

    return output_model.model_validate_json(response.json()["message"]["content"])
