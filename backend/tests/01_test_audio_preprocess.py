import argparse
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.services.audio_preprocess import audio_bytes_to_waveform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True,
                        help="Path to a test WAV/audio file")
    args = parser.parse_args()

    settings = get_settings()
    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_bytes = audio_path.read_bytes()

    waveform = audio_bytes_to_waveform(
        audio_bytes=audio_bytes,
        target_sample_rate=settings.STT_SAMPLE_RATE,
        filename=audio_path.name,
    )

    print("========== AUDIO PREPROCESS TEST ==========")
    print(f"Input file: {audio_path}")
    print(f"Input bytes: {len(audio_bytes)}")
    print(f"Output waveform type: {type(waveform)}")
    print(f"Output dtype: {waveform.dtype}")
    print(f"Output shape: {waveform.shape}")
    print(f"Sample rate expected: {settings.STT_SAMPLE_RATE}")
    print(
        f"Duration seconds approx: {len(waveform) / settings.STT_SAMPLE_RATE:.2f}")
    print(f"Min: {np.min(waveform)}")
    print(f"Max: {np.max(waveform)}")
    print(f"Has NaN: {np.isnan(waveform).any()}")
    print(f"Has Inf: {np.isinf(waveform).any()}")

    assert waveform.ndim == 1, "Waveform should be mono"
    assert waveform.dtype == np.float32, "Waveform should be float32"
    assert not np.isnan(waveform).any(), "Waveform has NaN"
    assert not np.isinf(waveform).any(), "Waveform has Inf"

    print("\nAudio preprocessing test passed.")


if __name__ == "__main__":
    main()
