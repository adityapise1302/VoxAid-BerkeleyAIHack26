from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.schemas import ErrorDetail, ErrorResponse, HealthResponse, VoiceTransformResponse
from app.services.pipeline import VoicePipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Loads the local Wav2Vec2 + LoRA model once at startup.
    app.state.pipeline = VoicePipeline(settings)

    yield


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    pipeline_exists = hasattr(request.app.state, "pipeline")

    return HealthResponse(
        status="ok" if pipeline_exists else "starting",
        stt_model_loaded=pipeline_exists,
        claude_ready=bool(settings.ANTHROPIC_API_KEY),
        deepgram_ready=bool(settings.DEEPGRAM_API_KEY),
    )


@app.post("/api/v1/voice/transform", response_model=VoiceTransformResponse)
async def transform_voice(
    request: Request,
    audio_file: Annotated[UploadFile, File(description="Recorded WAV audio")],
    voice_model: Annotated[
        str | None,
        Form(description="Optional Deepgram Aura voice model"),
    ] = None,
):
    try:
        content_type = audio_file.content_type or ""

        if not content_type.startswith("audio/"):
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code="INVALID_FILE_TYPE",
                        message=f"Expected an audio file, got: {content_type}",
                    )
                ).model_dump(),
            )

        audio_bytes = await audio_file.read()

        if not audio_bytes:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error=ErrorDetail(
                        code="EMPTY_AUDIO",
                        message="The uploaded audio file was empty.",
                    )
                ).model_dump(),
            )

        pipeline: VoicePipeline = request.app.state.pipeline

        result = await run_in_threadpool(
            pipeline.process,
            audio_bytes,
            voice_model,
            audio_file.filename,
        )

        return VoiceTransformResponse(**result)

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="PROCESSING_ERROR",
                    message=str(exc),
                )
            ).model_dump(),
        )