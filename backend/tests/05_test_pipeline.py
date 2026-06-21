import argparse
import base64
from pathlib import Path

from app.config import get_settings
from app.services.pipeline import VoicePipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a real speech WAV file")
    parser.add_argument(
        "--out",
        default="test_pipeline_output.mp3",
        help="Output TTS audio path",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Optional Deepgram Aura voice model",
    )
    args = parser.parse_args()

    settings = get_settings()
    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("========== FULL PIPELINE TEST ==========")
    print("Loading full pipeline...")
    pipeline = VoicePipeline(settings)
    print("Pipeline loaded.")

    audio_bytes = audio_path.read_bytes()

    print("Processing audio...")
    result = pipeline.process(
        audio_bytes=audio_bytes,
        voice_model=args.voice,
        filename=audio_path.name,
    )

    audio_output_bytes = base64.b64decode(result["audio_base64"])

    out_path = Path(args.out)
    out_path.write_bytes(audio_output_bytes)

    print("\n========== RESULT ==========")
    print(f"Success: {result['success']}")
    print(f"Raw transcript: {result['raw_transcript']}")
    print(f"Corrected text: {result['corrected_text']}")
    print(f"Voice model: {result['voice_model']}")
    print(f"Audio MIME type: {result['audio_mime_type']}")
    print(f"Processing time ms: {result['processing_time_ms']}")
    print(f"Saved TTS audio to: {out_path.resolve()}")
    print(f"Output audio bytes: {len(audio_output_bytes)}")

    assert result["success"] is True
    assert isinstance(result["raw_transcript"], str)
    assert isinstance(result["corrected_text"], str)
    assert len(audio_output_bytes) > 0

    print("\nFull pipeline test passed.")


if __name__ == "__main__":
    main()