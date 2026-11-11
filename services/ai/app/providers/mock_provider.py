from app.providers.base import LLMProvider


class MockProvider(LLMProvider):

    def generate(self, prompt: str) -> str:
        return (
            "The RAG pipeline is working. Retrieved university "
            "context was successfully passed to the LLM provider. "
            "Configure a hosted model to generate a real answer."
        )