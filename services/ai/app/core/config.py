from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

AI_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "NOVA AI"

    llm_provider: str = "ollama"

    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    ollama_model: str = "llama3.2:1b"
    ollama_host: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_file=AI_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()