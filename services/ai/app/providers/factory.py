from app.core.config import settings

from app.providers.base import LLMProvider
from app.providers.mock_provider import MockProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.ollama_provider import OllamaProvider


def get_llm_provider() -> LLMProvider:

    provider_name = settings.llm_provider.lower()

    if provider_name == "mock":
        return MockProvider()

    if provider_name == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model
        )

    if provider_name == "ollama":
        return OllamaProvider(
            model=settings.ollama_model,
            host=settings.ollama_host
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider_name}"
    )