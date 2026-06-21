import base64
import time

from app.config import Settings
from app.services.audio_preprocess import audio_bytes_to_waveform
from app.services.claude_corrector import ClaudeCorrector
from app.services.deepgram_tts import DeepgramAuraTTS
from app.services.stt_model import LocalWav2Vec2LoRASTT


class VoicePipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.stt = LocalWav2Vec2LoRASTT(
            base_model_name=settings.STT_BASE_MODEL,
            adapter_path=settings.STT_ADAPTER_PATH,
            sample_rate=settings.STT_SAMPLE_RATE,
            device=settings.DEVICE,
        )

        self.corrector = ClaudeCorrector(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.ANTHROPIC_MODEL,
        )

        self.tts = DeepgramAuraTTS(
            api_key=settings.DEEPGRAM_API_KEY,
            default_model=settings.DEEPGRAM_AURA_MODEL,
            encoding=settings.DEEPGRAM_ENCODING,
            container=settings.DEEPGRAM_CONTAINER,
            sample_rate=settings.DEEPGRAM_SAMPLE_RATE,
        )

    def process(self, audio_bytes: bytes, voice_model: str, filename: str | None = None) -> dict:
        start = time.perf_counter()

        waveform = audio_bytes_to_waveform(
            audio_bytes=audio_bytes,
            target_sample_rate=self.settings.STT_SAMPLE_RATE,
            filename=filename,
        )

        raw_transcript = self.stt.transcribe(waveform)

        corrected_text = self.corrector.correct(raw_transcript)

        # Keep TTS request safely bounded.
        tts_text = corrected_text[: self.settings.MAX_TTS_CHARS]

        output_audio_bytes = self.tts.synthesize(
            text=tts_text,
            voice_model=voice_model,
        )

        processing_time_ms = int((time.perf_counter() - start) * 1000)

        return {
            "success": True,
            "raw_transcript": raw_transcript,
            "corrected_text": corrected_text,
            "audio_base64": base64.b64encode(output_audio_bytes).decode("utf-8"),
            "audio_mime_type": self.tts.mime_type,
            "voice_model": voice_model or self.settings.DEEPGRAM_AURA_MODEL,
            "processing_time_ms": processing_time_ms,
        }