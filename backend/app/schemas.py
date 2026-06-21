from typing import Optional

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str


class VoiceTransformResponse(BaseModel):
    success: bool
    raw_transcript: str
    corrected_text: str
    audio_base64: str
    audio_mime_type: str
    voice_model: str
    processing_time_ms: int


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    stt_model_loaded: bool
    claude_ready: bool
    deepgram_ready: bool