from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    llm_provider: str = "mock"

    openai_api_key: str = ""

    openai_model: str = "gpt-4.1-mini"
    ollama_model: str = "llama3.2:1b"

    ollama_host: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()