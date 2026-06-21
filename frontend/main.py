# app.py
import streamlit as st
import requests
import base64
import os
import json
import hashlib

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
        "title": "Example 1",
        "video": "examples/demo_example1.mp3",
        "raw":      "AND DA OFO OTER ASREMAIN GENTAMDIN DAN BUT I NEVER DI AN DA IT WAS ONY AN A FAVRIRARY A WEN FOR MY FESICO AND DOCTOR PEERSAN PIK APON MY REC PEECH AND DA I HAD A GOOD HUM THERPED SAID PETER HAMIA IN TAN BEARN SWALLO AND AN TA GOTME AXIOSHIDIN THE TONG AN TATI VANTO",
        "corrected": "And the, uh, other assessments remain, gentleman, and but I never did, and the it was only and a February a, when for my physical and [Doctor Pearson?] picked up on my speech and the I had a good, um, therapist said [Peter Hamia?] in, and brain swallow, and and they got me an, uh, excitation in the tongue and that I want to",
        "deepgram":  "And, about a lot of were asking me to get something done, but I never did. And, it was only in February I went for my physical and Doctor. Peterson picked up on my reading speech and I had a good home therapist that had me ENT and be on swallow. And then she got me exercising the tongue and these muscles.",
    },
    {
        "title": "Example 2",
        "video": "examples/Dysarthria Example.mp3",
        "raw":      "WELL I E CHAPE I WELL SHAQI HOPO WE CA I WILL GOING AGAIN LAAE CHA",
        "corrected": "Well, I hope we can, I will be going again later, yeah.",
        "deepgram":  "I feel like I'm stopping. I will succeed. Coco, we can. I'm gonna get no ever seven.",
    },
    {
        "title": "Example 3",
        "video": "examples/demo_example3.mp3",
        "raw":      "MA MAY ME AN AD NOB ATA NN",
        "corrected": "Mama, may me and [Anad?] know about it now?",
        "deepgram":  "My baby ain't a nothing. All that is a man.",
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
        st.warning(example["raw"])

        st.markdown("**✅ Claude Corrected**")
        st.success(example["corrected"])

        st.markdown("**🔵 Deepgram Reference**")
        if example["deepgram"]:
            st.info(example["deepgram"])
        else:
            st.warning("Deepgram returned empty transcript for this sample.")

    st.markdown("---")

# ── Voice-only Agentverse chatbot section ──────────────────────

st.header("🎤 Voice Chat with VoxAid")
st.markdown(
    "Use your voice only. VoxAid will show what the STT model heard, what the agent "
    "deciphered, and the Agentverse response. The response is spoken automatically."
)

if "voxaid_chat" not in st.session_state:
    st.session_state.voxaid_chat = []

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None


def render_chat_message(message: dict):
    role = message.get("role")

    if role == "user":
        with st.chat_message("user"):
            st.markdown("#### You")
            st.markdown("**STT model heard:**")
            st.code(message.get("raw_transcript", ""), language=None)

            st.markdown("**VoxAid deciphered:**")
            st.success(message.get("corrected_text", ""))

    elif role == "agent":
        with st.chat_message("assistant"):
            mode = message.get("mode", "speech_only")
            agent_text = message.get("agent_text", "")
            audio_base64 = message.get("audio_base64", "")
            audio_mime_type = message.get("audio_mime_type", "audio/mpeg")

            if mode == "agentverse_task":
                st.markdown("#### Agentverse Agent")
            else:
                st.markdown("#### VoxAid")

            st.markdown(agent_text)

            if audio_base64:
                try:
                    audio_bytes = base64.b64decode(audio_base64)
                    st.audio(audio_bytes, format=audio_mime_type, autoplay=True)
                except TypeError:
                    audio_bytes = base64.b64decode(audio_base64)
                    st.audio(audio_bytes, format=audio_mime_type)

for message in st.session_state.voxaid_chat:
    render_chat_message(message)


st.markdown("---")
st.subheader("Record your next message")

try:
    audio_value = st.audio_input("Speak to VoxAid", sample_rate=16000)
except TypeError:
    audio_value = st.audio_input("Speak to VoxAid")


def process_voice_message(audio_file):
    audio_bytes = audio_file.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    if st.session_state.last_audio_hash == audio_hash:
        return

    st.session_state.last_audio_hash = audio_hash

    with st.spinner("VoxAid is listening, deciphering, routing, and speaking..."):
        response = requests.post(
            f"{backend_url}/api/v1/voice/transform",
            files={
                "audio_file": ("recording.wav", audio_bytes, "audio/wav"),
            },
            data={
                "voice_model": voice_model,
                "enable_agentverse": "true",
            },
            timeout=180,
        )

    if response.status_code != 200:
        try:
            st.error("Backend request failed.")
            st.json(response.json())
        except Exception:
            st.error(response.text)
        return

    result = response.json()

    if not result.get("success"):
        st.error("Backend returned an error.")
        st.json(result)
        return

    agentverse = result.get("agentverse") or {}

    st.session_state.voxaid_chat.append(
        {
            "role": "user",
            "raw_transcript": result.get("raw_transcript", ""),
            "corrected_text": result.get("corrected_text", ""),
        }
    )

    st.session_state.voxaid_chat.append(
        {
            "role": "agent",
            "mode": agentverse.get("mode", "speech_only"),
            "action_status": agentverse.get("action_status", "none"),
            "selected_agent": agentverse.get("selected_agent"),
            "agent_text": agentverse.get("agent_text") or result.get("corrected_text", ""),
            "audio_base64": result.get("audio_base64", ""),
            "audio_mime_type": result.get("audio_mime_type", "audio/mpeg"),
            "voice_model": result.get("voice_model", ""),
        }
    )

    st.rerun()


if audio_value is not None:
    process_voice_message(audio_value)


if st.session_state.voxaid_chat:
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.voxaid_chat = []
        st.session_state.last_audio_hash = None
        st.rerun()