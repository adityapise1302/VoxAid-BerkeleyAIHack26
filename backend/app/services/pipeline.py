import base64
import time

from app.config import Settings
from app.services.agentverse_router import AgentverseRouter
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

        self.agentverse_router = AgentverseRouter(settings)

    def process(
        self,
        audio_bytes: bytes,
        voice_model: str | None = None,
        filename: str | None = None,
        enable_agentverse: bool = True,
    ) -> dict:
        start = time.perf_counter()

        waveform = audio_bytes_to_waveform(
            audio_bytes=audio_bytes,
            target_sample_rate=self.settings.STT_SAMPLE_RATE,
            filename=filename,
        )

        raw_transcript = self.stt.transcribe(waveform)

        corrected_text = self.corrector.correct(raw_transcript)

        agentverse_result = None

        if enable_agentverse:
            agentverse_result = self.agentverse_router.route(
                corrected_text=corrected_text,
            )

        if agentverse_result and agentverse_result.get("spoken_summary"):
            tts_text = agentverse_result["spoken_summary"]
        else:
            tts_text = corrected_text

        tts_text = tts_text[: self.settings.MAX_TTS_CHARS]

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
            "agentverse": agentverse_result,
        }

    def route_agentverse_text(self, text: str) -> dict:
        corrected_text = (text or "").strip()
        return self.agentverse_router.route(corrected_text=corrected_text)