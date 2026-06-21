# VoxAid

**VoxAid** is a voice-first accessibility assistant designed for people with dysarthria and motor limitations. It converts unclear or impaired speech into clear intent, routes task-based requests through an agentic AI layer, and speaks the final response back to the user.

The goal is to let users interact with digital systems using only their voice, without needing to type, click, scroll, or manually operate a computer.

## Introduction

People with speech impairments often struggle with traditional voice assistants because automatic speech recognition systems are usually trained on typical speech patterns. VoxAid addresses this by using a custom dysarthria-focused speech recognition pipeline, followed by LLM-based correction and agentic task execution.

The system works as a conversational voice assistant:

1. The user records speech in the browser.
2. A locally fine-tuned Wav2Vec2 model transcribes the dysarthric speech.
3. Claude cleans and rephrases the raw STT output into a clearer sentence or task.
4. The VoxAid middle routing agent decides whether the user is simply communicating or asking for an agentic task.
5. Agentic tasks are sent to ASI.
6. Claude rephrases the ASI output into a clean, spoken-friendly response.
7. Deepgram Aura TTS generates speech from the final response.
8. The frontend displays the conversation in a chatbot-style interface and automatically plays the generated audio.

## System Architecture

```text
User Voice
   ↓
Streamlit Frontend
   ↓
FastAPI Backend
   ↓
Local Wav2Vec2 + LoRA Dysarthria STT
   ↓
Claude Speech Correction
   ↓
VoxAid Middle Routing Agent
   ↓
ASI:One Agentic Task Layer
   ↓
Claude Final Response Rephrasing
   ↓
Deepgram Aura TTS
   ↓
Spoken Response + Chatbot UI
```

## Features

* Browser-based voice input
* Local dysarthria speech-to-text model
* Claude-based transcript correction
* Middle routing agent for intent classification
* ASI integration for agentic task handling
* Claude final response rephrasing
* Deepgram Aura text-to-speech
* Voice-only user interaction
* Chatbot-style UI showing:

  * what the STT model heard
  * what VoxAid deciphered
  * the final assistant response
* FastAPI backend
* Streamlit frontend

## Project Structure

```text
VoxAid/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   └── services/
│   │       ├── audio_preprocess.py
│   │       ├── stt_model.py
│   │       ├── claude_corrector.py
│   │       ├── asi_one_client.py
│   │       ├── agentverse_router.py
│   │       ├── deepgram_tts.py
│   │       └── pipeline.py
│   ├── models/
│   │   └── wav2vec2-torgo-standard-lora/
│   ├── tests/
│   ├── requirements.txt
│   ├── .env.example
│   └── .env
│
├── frontend/
│   ├── main.py
│   ├── app.py
│   ├── examples/
│   └── requirements.txt
│
├── testing-voice/
├── .gitignore
└── README.md
```

## Requirements

Use **Python 3.11** for the backend.

Recommended environment:

```text
Python 3.11
macOS / Linux / Windows
FastAPI backend
Streamlit frontend
```

The project requires API keys for:

* Anthropic Claude
* ASI
* Deepgram

The project also requires the local Wav2Vec2 LoRA model files to be placed inside:

```text
backend/models/wav2vec2-torgo-standard-lora/
```

That folder should contain files like:

```text
adapter_model.safetensors
adapter_config.json
tokenizer_config.json
vocab.json
processor_config.json
training_metadata.json
README.md
```

## API Key Setup

Create a `.env` file inside the `backend/` folder:

```bash
cd backend
cp .env.example .env
```

Then edit `backend/.env`.

Example:

```env
# Server
APP_NAME=Dysarthria Voice Backend
APP_ENV=development
CORS_ALLOW_ORIGINS=http://localhost:8501,http://127.0.0.1:8501

# Local STT model
STT_BASE_MODEL=facebook/wav2vec2-base-960h
STT_ADAPTER_PATH=models/wav2vec2-torgo-standard-lora
STT_SAMPLE_RATE=16000
DEVICE=auto

# Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-sonnet-4-6

# ASI:One
ASI_ONE_API_KEY=your_asi_one_api_key_here
ASI_ONE_BASE_URL=https://api.asi1.ai/v1
ASI_ONE_MODEL=asi1
ASI_ONE_TIMEOUT_SECONDS=60

# Deepgram Aura TTS
DEEPGRAM_API_KEY=your_deepgram_api_key_here
DEEPGRAM_AURA_MODEL=aura-2-thalia-en
DEEPGRAM_ENCODING=mp3
DEEPGRAM_CONTAINER=
DEEPGRAM_SAMPLE_RATE=22050

# TTS limit
MAX_TTS_CHARS=2000
```

### What each key is used for

#### `ANTHROPIC_API_KEY`

Used for:

* correcting raw dysarthric STT output
* cleaning repeated or unclear transcriptions
* classifying whether the user is simply speaking or asking for a task
* rephrasing ASI output into a spoken-friendly final response

#### `ASI_ONE_API_KEY`

Used for:

* handling agentic tasks
* generating useful task responses
* replacing the earlier manual Agentverse relay-agent workflow

#### `DEEPGRAM_API_KEY`

Used for:

* generating speech output using Deepgram Aura TTS

Deepgram is used only for text-to-speech in this project, not for speech-to-text.

## Backend Setup

From the project root:

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are on Windows:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Backend Requirements

Your `backend/requirements.txt` should include the main dependencies:

```txt
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

deepgram-sdk>=7.0.0
anthropic>=0.70.0
openai>=1.60.0

torch
transformers>=4.45.0
peft>=0.19.0
safetensors>=0.4.0

numpy>=1.26.0
soundfile>=0.12.1
librosa>=0.10.0
```

If MP3 decoding fails locally, install FFmpeg.

On macOS:

```bash
brew install ffmpeg
```

## Running the Backend

From the `backend/` folder:

```bash
source .venv/bin/activate
PYTHONPATH=. python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend should start at:

```text
http://localhost:8000
```

You can check the health endpoint:

```text
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "stt_model_loaded": true,
  "claude_ready": true,
  "deepgram_ready": true,
  "agentverse_ready": true
}
```

In this project, `agentverse_ready` represents the ASI agentic layer being available.

## Frontend Setup

Open a new terminal from the project root:

```bash
cd frontend
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `frontend/requirements.txt` is minimal or missing packages, install:

```bash
pip install streamlit requests python-dotenv
```

## Running the Frontend

From the `frontend/` folder:

```bash
source .venv/bin/activate
python -m streamlit run main.py
```

The frontend should open at:

```text
http://localhost:8501
```

Make sure the backend is already running on:

```text
http://localhost:8000
```

## How to Use

1. Start the backend.
2. Start the frontend.
3. Open the Streamlit app in your browser.
4. Record your voice using the microphone input.
5. VoxAid will process the recording automatically.
6. The chat UI will show:

   * the raw STT transcript
   * the corrected user intent
   * the final assistant response
7. The final response will be spoken automatically using Deepgram Aura TTS.

Example request:

```text
Find the nearest library near ASU Tempe campus.
```

Expected behavior:

```text
STT model heard:
find me the nearest library near asu tempe campus

VoxAid deciphered:
Find the nearest library near ASU Tempe campus.

ASI:One / Assistant:
The nearest library near ASU Tempe campus is Hayden Library, located on the Tempe campus.
```

## Voice-Only Interaction

VoxAid is designed so the user does not need to type. The user interacts through voice recordings, and the system responds through both text and generated speech.

The frontend is intentionally designed like a chatbot, but the user input is voice-only.

## Backend API

### Health Check

```http
GET /health
```

### Voice Transform Endpoint

```http
POST /api/v1/voice/transform
```

Form data:

```text
audio_file: recorded audio file
voice_model: optional Deepgram Aura voice model
enable_agentverse: true
```

Example response:

```json
{
  "success": true,
  "raw_transcript": "find me nearest library",
  "corrected_text": "Find the nearest library.",
  "audio_base64": "...",
  "audio_mime_type": "audio/mpeg",
  "voice_model": "aura-2-thalia-en",
  "processing_time_ms": 12345,
  "agentverse": {
    "enabled": true,
    "mode": "agentverse_task",
    "action_status": "completed",
    "agent_text": "I need your current location or ZIP code to find the nearest library.",
    "spoken_summary": "I need your current location or ZIP code to find the nearest library."
  }
}
```

## Model Setup

The local STT model should be placed here:

```text
backend/models/wav2vec2-torgo-standard-lora/
```

The model is a LoRA adapter over:

```text
facebook/wav2vec2-base-960h
```

The backend loads the base Wav2Vec2 model and applies the LoRA adapter at startup.

If the model folder is missing or incomplete, the backend will fail during startup.

## Important Notes

Do not commit your `.env` file.

Do not commit API keys.

Do not commit virtual environments.

Do not commit large model checkpoints unless required for the hackathon submission.

The `.gitignore` should exclude:

```text
.env
.venv/
__pycache__/
testing-voice/
*.mp3
*.wav
backend/models/
models/
*.safetensors
*.pt
*.pth
```

If the model is required for reproducibility, provide a separate download link or include instructions for placing it in the correct folder.

## Troubleshooting

### Backend says port 8000 is already in use

Find and kill the existing process:

```bash
lsof -i :8000
kill -9 <PID>
```

Or run on another port:

```bash
PYTHONPATH=. python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Then update the frontend backend URL if needed.

### Frontend cannot connect to backend

Make sure the backend is running:

```text
http://localhost:8000
```

Also check that the frontend is using the correct backend URL.

### Streamlit microphone error

If `st.audio_input` gives an error related to `sample_rate`, your Streamlit version may not support that argument.

Upgrade Streamlit:

```bash
pip install --upgrade streamlit
```

Or remove the `sample_rate=16000` argument from `st.audio_input`.

### Deepgram MP3 container error

If Deepgram returns an error about MP3 container settings, make sure:

```env
DEEPGRAM_ENCODING=mp3
DEEPGRAM_CONTAINER=
```

For MP3, leave `DEEPGRAM_CONTAINER` empty.

### ASI fails

Check that your `.env` contains:

```env
ASI_ONE_API_KEY=your_real_asi_one_key_here
ASI_ONE_BASE_URL=https://api.asi1.ai/v1
ASI_ONE_MODEL=asi1
```

Then restart the backend.

### Claude correction fails

Check that your `.env` contains:

```env
ANTHROPIC_API_KEY=your_real_anthropic_key_here
ANTHROPIC_MODEL=claude-sonnet-4-6
```

Then restart the backend.

### STT model fails to load

Make sure the model files exist at:

```text
backend/models/wav2vec2-torgo-standard-lora/
```

The folder should contain:

```text
adapter_model.safetensors
adapter_config.json
tokenizer_config.json
vocab.json
processor_config.json
training_metadata.json
```

### Audio decoding fails

Install FFmpeg.

On macOS:

```bash
brew install ffmpeg
```

## Development Tests

The backend includes test scripts inside:

```text
backend/tests/
```

Example:

```bash
cd backend
source .venv/bin/activate
PYTHONPATH=. python tests/00_check_env.py
```

To test audio preprocessing:

```bash
PYTHONPATH=. python tests/01_test_audio_preprocess.py --audio "../testing-voice/Spastic dysarthria example.mp3"
```

To test the full pipeline:

```bash
PYTHONPATH=. python tests/05_test_pipeline.py --audio "../testing-voice/Spastic dysarthria example.mp3"
```

## Summary

VoxAid combines custom dysarthria speech recognition, LLM-based correction, agentic task routing, and text-to-speech to create a voice-only interface for people who may struggle with both speech clarity and physical computer interaction.

The project demonstrates how accessible voice interfaces can act as a gateway to agentic AI systems, allowing users to complete digital tasks through natural speech.
