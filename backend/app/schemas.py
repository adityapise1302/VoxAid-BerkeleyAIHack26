from typing import Any, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class AgentverseAgent(BaseModel):
    name: str = "Unnamed Agent"
    address: str = ""
    description: str = ""
    url: str = ""
    score: Optional[Any] = None
    status: Optional[Any] = None
    developer: Optional[Any] = None
    protocols: list[Any] = Field(default_factory=list)


class AgentverseRouteResult(BaseModel):
    enabled: bool = False

    # speech_only or agentverse_task
    mode: str = "speech_only"

    # none, completed, no_agent_found, search_failed
    action_status: str = "none"

    reason: str = ""
    task_for_agent: str = ""
    search_query: str = ""

    selected_agent: Optional[AgentverseAgent] = None

    # What the agent says back to the user
    agent_text: str = ""

    # Same as agent_text, used by TTS
    spoken_summary: str = ""

    search_error: Optional[str] = None


class VoiceTransformResponse(BaseModel):
    success: bool
    raw_transcript: str
    corrected_text: str
    audio_base64: str
    audio_mime_type: str
    voice_model: str
    processing_time_ms: int
    agentverse: Optional[AgentverseRouteResult] = None


class AgentverseRouteRequest(BaseModel):
    text: str


class AgentverseRouteResponse(BaseModel):
    success: bool
    corrected_text: str
    agentverse: AgentverseRouteResult


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    stt_model_loaded: bool
    claude_ready: bool
    deepgram_ready: bool
    agentverse_ready: bool = False