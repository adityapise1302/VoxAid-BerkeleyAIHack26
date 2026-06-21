# app.py
import streamlit as st
import requests
import base64
import os

# ── Config ────────────────────────────────────────────────────
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(
    page_title="ClearVoice",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ ClearVoice")
st.caption("Speech accessibility tool for people with dysarthria")

# ── Sidebar — backend config ──────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    backend_url = st.text_input(
        "Backend URL",
        value=BACKEND_URL,
        help="URL of the FastAPI backend"
    )
    voice_model = st.selectbox(
        "Voice Model",
        options=[
            "aura-2-thalia-en",
            "aura-2-orion-en",
            "aura-2-luna-en",
            "aura-2-stella-en",
        ],
        index=0,
        help="Deepgram Aura voice model for speech output"
    )

    # ── Health check ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("Backend Status")
    if st.button("Check Backend Health"):
        try:
            resp = requests.get(f"{backend_url}/health", timeout=5)
            if resp.status_code == 200:
                health = resp.json()
                st.success("Backend is online")
                st.json(health)
            else:
                st.error(f"Backend returned status {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach backend — is it running?")
        except Exception as e:
            st.error(f"Health check failed: {e}")

# ── Main UI ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 1 — Record your speech")
st.markdown(
    "Press the button below to start recording. "
    "Press again to stop."
)

audio_value = st.audio_input(
    "Record your speech",
    sample_rate=16000,
)

# ── Playback of original recording ───────────────────────────
if audio_value is not None:
    st.markdown("**Your recording:**")
    st.audio(audio_value, format="audio/wav")

    st.markdown("---")
    st.subheader("Step 2 — Process Recording")

    if st.button("▶️ Process Recording", type="primary", use_container_width=True):
        with st.spinner("Sending to backend and processing..."):
            try:
                audio_bytes = audio_value.read()

                response = requests.post(
                    f"{backend_url}/api/v1/voice/transform",
                    files={
                        "audio_file": ("recording.wav", audio_bytes, "audio/wav")
                    },
                    data={
                        "voice_model": voice_model
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()

                    if result.get("success"):
                        st.markdown("---")
                        st.subheader("Step 3 — Results")

                        # ── Corrected text ────────────────────
                        st.markdown("### ✅ Corrected Speech")
                        st.success(result["corrected_text"])

                        # ── Generated audio ───────────────────
                        st.markdown("### 🔊 Generated Voice Output")
                        audio_b64  = result.get("audio_base64", "")
                        audio_mime = result.get("audio_mime_type", "audio/mpeg")
                        if audio_b64:
                            audio_out = base64.b64decode(audio_b64)
                            st.audio(audio_out, format=audio_mime)
                        else:
                            st.warning("No audio returned from backend.")

                        # ── Raw transcript (debug) ────────────
                        with st.expander("🔍 Raw ASR Transcript (debug)"):
                            st.code(result.get("raw_transcript", "(empty)"), language=None)
                            st.caption(
                                f"Processing time: {result.get('processing_time_ms', 'N/A')} ms  |  "
                                f"Voice model: {result.get('voice_model', 'N/A')}"
                            )

                    else:
                        # Backend returned success: false
                        error = result.get("error", {})
                        st.error(
                            f"Backend error [{error.get('code', 'UNKNOWN')}]: "
                            f"{error.get('message', 'Something went wrong.')}"
                        )

                else:
                    st.error(
                        f"Request failed with status {response.status_code}. "
                        "Check that the backend is running and the URL is correct."
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the backend. "
                    "Make sure it is running and the URL in the sidebar is correct."
                )
            except requests.exceptions.Timeout:
                st.error(
                    "The request timed out after 60 seconds. "
                    "The backend may be overloaded or the audio too long."
                )
            except Exception as e:
                st.error(f"Unexpected error: {e}")