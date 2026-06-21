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
    page_title="VoxAid",
    page_icon="🎙️",
    layout="centered"
)

st.title("VoxAid")
st.caption("Speech accessibility tool for people with dysarthria and other speech impediments")

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

# ── Example comparisons ───────────────────────────────────────
# Replace placeholder text with your actual pre-computed transcripts.
# Replace video paths with your actual files in an examples/ folder.
EXAMPLES = [
    {
        "title": "Example 1 - Lower Severity",
        "video": "examples/demo_example1.mp3",
        "raw":      "AND DA OFO OTER ASREMAIN GENTAMDIN DAN BUT I NEVER DI AN DA IT WAS ONY AN A FAVRIRARY A WEN FOR MY FESICO AND DOCTOR PEERSAN PIK APON MY REC PEECH AND DA I HAD A GOOD HUM THERPED SAID PETER HAMIA IN TAN BEARN SWALLO AND AN TA GOTME AXIOSHIDIN THE TONG AN TATI VANTO",
        "corrected": "And the, uh, other assessments remain, gentleman, and but I never did, and the it was only and a February a, when for my physical and [Doctor Pearson?] picked up on my speech and the I had a good, um, therapist said [Peter Hamia?] in, and brain swallow, and and they got me an, uh, excitation in the tongue and that I want to",
        "deepgram":  "And, about a lot of were asking me to get something done, but I never did. And, it was only in February I went for my physical and Doctor. Peterson picked up on my reading speech and I had a good home therapist that had me ENT and be on swallow. And then she got me exercising the tongue and these muscles.",
        "true":"And, uh, my daughters were after me to get something done..."
    },
    {
        "title": "Example 2",
        "video": "examples/example2.mp4",
        "raw":      "CN YU HEP ME WIT MY MEDICASHUN",
        "corrected": "Can you help me with my medication?",
        "deepgram":  "",
    },
    {
        "title": "Example 3",
        "video": "examples/example3.mp4",
        "raw":      "FAVRIRARY FEJAKO PEERSAN PIGYA PA MA RICH WECH",
        "corrected": "February checkup [Doctor Pierson?] picked up my prescription.",
        "deepgram":  "",
    },
]

st.markdown("---")
st.header("📹 Example Comparisons")
st.markdown(
    "The examples below show how our pipeline handles real dysarthric speech. "
    "Each example displays the raw ASR output, our Claude-corrected version, "
    "and Deepgram's output for reference."
)

for example in EXAMPLES:
    st.markdown(f"### {example['title']}")
    col_video, col_results = st.columns([1, 1])

    with col_video:
        try:
            st.video(example["video"])
        except Exception:
            st.info("📁 Add video file to examples/ folder to display here.")

    with col_results:
        st.markdown("**🔴 Raw ASR Output**")
        st.code(example["raw"], language=None)

        st.markdown("**✅ Claude Corrected**")
        st.success(example["corrected"])

        st.markdown("**🔵 Deepgram Reference**")
        if example["deepgram"]:
            st.info(example["deepgram"])
        else:
            st.warning("Deepgram returned empty transcript for this sample.")

    st.markdown("---")

# ── Live recording section ────────────────────────────────────
st.header("🎤 Try It Live")
st.markdown(
    "Record yourself speaking — or have someone with a speech impairment try it. "
    "The pipeline will transcribe and clean your speech in real time."
)

st.subheader("Step 1 — Record your speech")
audio_value = st.audio_input(
    "Record your speech",
    sample_rate=16000,
)

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

                        st.markdown("### ✅ Corrected Speech")
                        st.success(result["corrected_text"])

                        st.markdown("### 🔊 Generated Voice Output")
                        audio_b64  = result.get("audio_base64", "")
                        audio_mime = result.get("audio_mime_type", "audio/mpeg")
                        if audio_b64:
                            audio_out = base64.b64decode(audio_b64)
                            st.audio(audio_out, format=audio_mime)
                        else:
                            st.warning("No audio returned from backend.")

                        with st.expander("🔍 Raw ASR Transcript (debug)"):
                            st.code(result.get("raw_transcript", "(empty)"), language=None)
                            st.caption(
                                f"Processing time: {result.get('processing_time_ms', 'N/A')} ms  |  "
                                f"Voice model: {result.get('voice_model', 'N/A')}"
                            )

                    else:
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