from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API endpoints
    BOTHUB_API_BASE: str = "https://bothub.chat/api/v2/openai/v1"
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    
    # API keys
    BOTHUB_API_KEY: str
    OPENAI_API_KEY: str
    
    # Model settings
    MODEL_NAME: str = "gpt-4o"
    TEMPERATURE: float = 0.7
    STREAMING: bool = True
    
    # Proxy settings
    SOCKS5_PROXY: str = "socks5://user:pass@host:port"
    
    # API selection ('openai' or 'bothub')
    LLM_PROVIDER: str = "bothub"

    class Config:
        env_file = ".env"

settings = Settings()
OPENAI_API_KEY = settings.OPENAI_API_KEY 