import argparse
from pathlib import Path

from app.config import get_settings
from app.services.audio_preprocess import audio_bytes_to_waveform
from app.services.stt_model import LocalWav2Vec2LoRASTT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True,
                        help="Path to a real speech WAV file")
    args = parser.parse_args()

    settings = get_settings()
    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("========== STT MODEL TEST ==========")
    print("Loading STT model...")
    print(f"Base model: {settings.STT_BASE_MODEL}")
    print(f"Adapter path: {settings.STT_ADAPTER_PATH}")
    print(f"Device setting: {settings.DEVICE}")

    stt = LocalWav2Vec2LoRASTT(
        base_model_name=settings.STT_BASE_MODEL,
        adapter_path=settings.STT_ADAPTER_PATH,
        sample_rate=settings.STT_SAMPLE_RATE,
        device=settings.DEVICE,
    )

    print(f"Actual device: {stt.device}")
    print("Model loaded successfully.")

    audio_bytes = audio_path.read_bytes()
    waveform = audio_bytes_to_waveform(
        audio_bytes=audio_bytes,
        target_sample_rate=settings.STT_SAMPLE_RATE,
        filename=audio_path.name,
    )

    print("Running transcription...")
    transcript = stt.transcribe(waveform)

    print("\n========== RESULT ==========")
    print(f"Raw transcript: {transcript}")

    assert isinstance(transcript, str), "Transcript should be a string"

    print("\nSTT model test complete.")


if __name__ == "__main__":
    main()
