import argparse
from pathlib import Path

from app.config import get_settings
from app.services.deepgram_tts import DeepgramAuraTTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        default="I want to go to the store to buy medicine.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--out",
        default="test_deepgram_output.mp3",
        help="Output audio file path",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Optional Deepgram Aura voice model, e.g. aura-2-thalia-en",
    )
    args = parser.parse_args()

    settings = get_settings()

    print("========== DEEPGRAM TTS TEST ==========")
    print(f"Default voice model: {settings.DEEPGRAM_AURA_MODEL}")
    print(f"Requested voice model: {args.voice or settings.DEEPGRAM_AURA_MODEL}")
    print(f"Encoding: {settings.DEEPGRAM_ENCODING}")
    print(f"Container: {settings.DEEPGRAM_CONTAINER}")
    print(f"Sample rate: {settings.DEEPGRAM_SAMPLE_RATE}")
    print(f"Text: {args.text}")

    tts = DeepgramAuraTTS(
        api_key=settings.DEEPGRAM_API_KEY,
        default_model=settings.DEEPGRAM_AURA_MODEL,
        encoding=settings.DEEPGRAM_ENCODING,
        container=settings.DEEPGRAM_CONTAINER,
        sample_rate=settings.DEEPGRAM_SAMPLE_RATE,
    )

    audio_bytes = tts.synthesize(
        text=args.text,
        voice_model=args.voice,
    )

    out_path = Path(args.out)
    out_path.write_bytes(audio_bytes)

    print("\n========== RESULT ==========")
    print(f"Saved audio to: {out_path.resolve()}")
    print(f"Audio bytes: {len(audio_bytes)}")
    print(f"MIME type: {tts.mime_type}")

    assert len(audio_bytes) > 0, "Deepgram returned empty audio"

    print("\nDeepgram TTS test passed.")


if __name__ == "__main__":
    main()