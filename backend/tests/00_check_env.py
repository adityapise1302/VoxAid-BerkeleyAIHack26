from pathlib import Path

from app.config import get_settings


def mask_key(value: str | None) -> str:
    if not value:
        return "MISSING"
    if len(value) <= 8:
        return "SET"
    return value[:4] + "..." + value[-4:]


def main():
    settings = get_settings()

    print("========== ENV CHECK ==========")
    print(f"APP_NAME: {settings.APP_NAME}")
    print(f"APP_ENV: {settings.APP_ENV}")
    print(f"STT_BASE_MODEL: {settings.STT_BASE_MODEL}")
    print(f"STT_ADAPTER_PATH: {settings.STT_ADAPTER_PATH}")
    print(f"STT_SAMPLE_RATE: {settings.STT_SAMPLE_RATE}")
    print(f"DEVICE: {settings.DEVICE}")
    print(f"ANTHROPIC_MODEL: {settings.ANTHROPIC_MODEL}")
    print(f"DEEPGRAM_AURA_MODEL: {settings.DEEPGRAM_AURA_MODEL}")
    print(f"DEEPGRAM_ENCODING: {settings.DEEPGRAM_ENCODING}")
    print(f"DEEPGRAM_CONTAINER: {settings.DEEPGRAM_CONTAINER}")
    print(f"DEEPGRAM_SAMPLE_RATE: {settings.DEEPGRAM_SAMPLE_RATE}")

    print("\n========== KEY CHECK ==========")
    print(f"ANTHROPIC_API_KEY: {mask_key(settings.ANTHROPIC_API_KEY)}")
    print(f"DEEPGRAM_API_KEY: {mask_key(settings.DEEPGRAM_API_KEY)}")

    print("\n========== MODEL PATH CHECK ==========")
    adapter_path = Path(settings.STT_ADAPTER_PATH)
    print(f"Adapter path exists: {adapter_path.exists()}")
    print(f"Resolved adapter path: {adapter_path.resolve()}")

    expected_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "vocab.json",
        "tokenizer_config.json",
        "processor_config.json",
    ]

    for filename in expected_files:
        path = adapter_path / filename
        print(f"{filename}: {'FOUND' if path.exists() else 'MISSING'}")

    print("\n========== IMPORT CHECK ==========")
    import torch
    import transformers
    import peft
    import anthropic
    import deepgram
    import fastapi

    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"peft: {peft.__version__}")
    print(f"anthropic: imported")
    print(f"deepgram: imported")
    print(f"fastapi: imported")

    print("\nEnv check complete.")


if __name__ == "__main__":
    main()