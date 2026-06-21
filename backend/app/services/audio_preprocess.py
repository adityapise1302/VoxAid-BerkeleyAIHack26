from io import BytesIO
from pathlib import Path
import tempfile

import librosa
import numpy as np
import soundfile as sf


class AudioDecodeError(ValueError):
    pass


def audio_bytes_to_waveform(
    audio_bytes: bytes,
    target_sample_rate: int = 16000,
    filename: str | None = None,
) -> np.ndarray:
    """
    Converts uploaded browser-recorded audio bytes into a mono 16kHz float32 waveform.

    Supports:
    - WAV
    - MP3
    - other formats that librosa/ffmpeg can decode

    For MP3 files, if soundfile cannot decode the bytes directly,
    we save to a temporary file and let librosa decode it.
    """

    if not audio_bytes:
        raise AudioDecodeError("Empty audio file received.")

    waveform = None
    sample_rate = None

    # First try soundfile. This works well for WAV and sometimes MP3,
    # depending on installed libsndfile support.
    try:
        waveform, sample_rate = sf.read(
            BytesIO(audio_bytes),
            dtype="float32",
            always_2d=False,
        )
    except Exception:
        waveform = None
        sample_rate = None

    # Fallback for MP3 and other compressed formats.
    if waveform is None or sample_rate is None:
        suffix = ".mp3"

        if filename:
            suffix = Path(filename).suffix or ".mp3"

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio.flush()

                waveform, sample_rate = librosa.load(
                    temp_audio.name,
                    sr=None,
                    mono=False,
                )
        except Exception as exc:
            raise AudioDecodeError(
                "Could not decode audio. If this is an MP3 file, make sure ffmpeg is installed."
            ) from exc

    if waveform is None or len(waveform) == 0:
        raise AudioDecodeError("Decoded audio is empty.")

    waveform = np.asarray(waveform, dtype=np.float32)

    # librosa may return shape as channels x samples.
    # soundfile usually returns samples x channels.
    if waveform.ndim > 1:
        if waveform.shape[0] < waveform.shape[1]:
            waveform = np.mean(waveform, axis=0)
        else:
            waveform = np.mean(waveform, axis=1)

    waveform = np.nan_to_num(waveform).astype(np.float32)

    # Resample to Wav2Vec2 expected sample rate.
    if sample_rate != target_sample_rate:
        waveform = librosa.resample(
            y=waveform,
            orig_sr=sample_rate,
            target_sr=target_sample_rate,
        ).astype(np.float32)

    max_abs = float(np.max(np.abs(waveform)))
    if max_abs > 1.0:
        waveform = waveform / max_abs

    return waveform.astype(np.float32)