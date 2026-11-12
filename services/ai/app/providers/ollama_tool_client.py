import logging
import time
import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

OLLAMA_URL = settings.ollama_host.rstrip("/")
MODEL = settings.ollama_model


def get_available_models():
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

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": 256,
        },
    }

    if tools is not None:
        payload["tools"] = tools

    start = time.perf_counter()

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()

    elapsed = time.perf_counter() - start

    logger.info(
        "OLLAMA PERFORMANCE | model=%s | elapsed=%.2fs | "
        "load=%.2fs | prompt_tokens=%s | output_tokens=%s | "
        "generation=%.2fs",
        MODEL,
        elapsed,
        data.get("load_duration", 0) / 1e9,
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
        data.get("eval_duration", 0) / 1e9,
    )

    return data["message"]