
import os

import requests


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
)

MODEL = os.getenv("NOVA_MODEL", "llama3.2:1b")


def get_available_models():
    """Return the model names currently available in Ollama."""
    response = requests.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=5,
    )
    response.raise_for_status()

    return [
        item["name"]
        for item in response.json().get("models", [])
    ]


def chat_with_model(messages, tools=None):
    """Send conversation context to Ollama and return its reply."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }

    if tools is not None:
        payload["tools"] = tools

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()

    data = response.json()
    return data["message"]
