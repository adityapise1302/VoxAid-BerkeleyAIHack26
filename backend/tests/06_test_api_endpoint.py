import argparse
import base64
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a real speech WAV file")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="FastAPI backend base URL",
    )
    parser.add_argument(
        "--out",
        default="test_api_output.mp3",
        help="Output audio file path",
    )
    parser.add_argument(
        "--voice",
        default="aura-2-thalia-en",
        help="Deepgram Aura voice model",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("========== API HEALTH CHECK ==========")
    health_url = f"{args.url}/health"
    health_response = requests.get(health_url, timeout=30)
    print(f"Health status code: {health_response.status_code}")
    print(health_response.text)
    health_response.raise_for_status()

    print("\n========== API VOICE TRANSFORM TEST ==========")
    endpoint = f"{args.url}/api/v1/voice/transform"

    with audio_path.open("rb") as f:
        files = {
            "audio_file": ("recording.wav", f, "audio/mpeg"),
        }
        data = {
            "voice_model": args.voice,
        }

        response = requests.post(
            endpoint,
            files=files,
            data=data,
            timeout=180,
        )

    print(f"Status code: {response.status_code}")
    print(f"Response text preview: {response.text[:500]}")

    response.raise_for_status()

    result = response.json()

    if not result.get("success"):
        raise RuntimeError(f"Backend returned failure: {result}")

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

    print("\nAPI endpoint test passed.")


if __name__ == "__main__":
    main()