from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BOTHUB_API_BASE: str = "https://bothub.chat/api/v2/openai/v1"
    BOTHUB_API_KEY: str
    OPENAI_API_KEY: str
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.7
    STREAMING: bool = True

    class Config:
        env_file = ".env"

settings = Settings()
OPENAI_API_KEY = settings.OPENAI_API_KEY 