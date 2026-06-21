from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Dysarthria Voice Backend"
    APP_ENV: str = "development"
    CORS_ALLOW_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    STT_BASE_MODEL: str = "facebook/wav2vec2-base-960h"
    STT_ADAPTER_PATH: str = "models/wav2vec2-torgo-standard-lora"
    STT_SAMPLE_RATE: int = 16000
    DEVICE: str = "auto"

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_AURA_MODEL: str = "aura-2-thalia-en"
    DEEPGRAM_ENCODING: str = "mp3"
    DEEPGRAM_CONTAINER: str = ""
    DEEPGRAM_SAMPLE_RATE: int = 22050

    # Fetch.ai / Agentverse
    FETCHAI_AGENTVERSE_API_KEY: str = ""
    AGENTVERSE_API_KEY: str = ""
    AGENTVERSE_SEARCH_URL: str = "https://agentverse.ai/v1/search/agents"
    AGENTVERSE_TIMEOUT_SECONDS: float = 10.0
    AGENTVERSE_PROTOCOL_DIGEST: str = ""
    AGENTVERSE_CHAT_WAIT_SECONDS: int = 60
    AGENTVERSE_RELAY_AGENT_ADDRESS: str = ""
    AGENTVERSE_START_SESSION: bool = False

    MAX_TTS_CHARS: int = 2000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> List[str]:
        return [
            origin.strip()
            for origin in self.CORS_ALLOW_ORIGINS.split(",")
            if origin.strip()
        ]

    @property
    def agentverse_api_key(self) -> str:
        return self.FETCHAI_AGENTVERSE_API_KEY or self.AGENTVERSE_API_KEY


@lru_cache
def get_settings() -> Settings:
    return Settings()