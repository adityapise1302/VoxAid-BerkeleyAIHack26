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

    # ASI:One
    ASI_ONE_API_KEY: str = ""
    ASI_ONE_BASE_URL: str = "https://api.asi1.ai/v1"
    ASI_ONE_MODEL: str = "asi1"
    ASI_ONE_TIMEOUT_SECONDS: float = 60.0

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
    def asi_one_ready(self) -> bool:
        return bool(self.ASI_ONE_API_KEY)


@lru_cache
def get_settings() -> Settings:
    return Settings()