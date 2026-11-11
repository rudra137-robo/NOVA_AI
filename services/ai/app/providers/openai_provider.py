from openai import OpenAI

from app.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OpenAI API key is missing.")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are SASTRA AI, a university student "
                        "assistant. Answer accurately and clearly. "
                        "Use only the supplied university context "
                        "for university-specific facts. If the "
                        "context does not contain the answer, say "
                        "that the available documents do not provide "
                        "enough information. Never invent university "
                        "policies, dates, deadlines, or procedures. "
                        "Cite supporting passages using [1], [2], etc."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

        return answer.strip() if answer else (
            "I couldn't generate an answer. Please try again."
        )