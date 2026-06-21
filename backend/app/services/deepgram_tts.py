from deepgram import DeepgramClient


class DeepgramAuraTTS:
    def __init__(
        self,
        api_key: str,
        default_model: str,
        encoding: str = "mp3",
        container: str | None = None,
        sample_rate: int | None = None,
    ) -> None:
        self.client = DeepgramClient(api_key=api_key)
        self.default_model = default_model
        self.encoding = encoding
        self.container = container if container else None
        self.sample_rate = sample_rate

    @property
    def mime_type(self) -> str:
        if self.encoding == "mp3":
            return "audio/mpeg"
        if self.encoding == "linear16" and self.container == "wav":
            return "audio/wav"
        if self.encoding == "opus" and self.container == "ogg":
            return "audio/ogg"
        if self.encoding == "flac":
            return "audio/flac"
        if self.encoding == "aac":
            return "audio/aac"
        return "application/octet-stream"

    def _response_to_bytes(self, response) -> bytes:
        if isinstance(response, bytes):
            return response

        if isinstance(response, bytearray):
            return bytes(response)

        if hasattr(response, "stream") and hasattr(response.stream, "getvalue"):
            return response.stream.getvalue()

        if hasattr(response, "content") and isinstance(response.content, bytes):
            return response.content

        chunks = []

        try:
            for chunk in response:
                if isinstance(chunk, bytes):
                    chunks.append(chunk)
                elif isinstance(chunk, bytearray):
                    chunks.append(bytes(chunk))
                elif hasattr(chunk, "data") and isinstance(chunk.data, bytes):
                    chunks.append(chunk.data)
                elif hasattr(chunk, "content") and isinstance(chunk.content, bytes):
                    chunks.append(chunk.content)
                else:
                    raise TypeError(
                        f"Unsupported Deepgram audio chunk type: {type(chunk)}"
                    )
        except TypeError as exc:
            raise RuntimeError(
                f"Unsupported Deepgram TTS response type: {type(response)}"
            ) from exc

        return b"".join(chunks)

    def synthesize(self, text: str, voice_model: str | None = None) -> bytes:
        text = (text or "").strip()

        if not text:
            raise ValueError("Cannot synthesize empty text.")

        model = voice_model or self.default_model

        kwargs = {
            "text": text,
            "model": model,
            "encoding": self.encoding,
        }

        # For MP3, do NOT send container or sample_rate.
        # Deepgram treats MP3 as an encoding with no configurable container.
        if self.encoding != "mp3":
            if self.container:
                kwargs["container"] = self.container
            if self.sample_rate:
                kwargs["sample_rate"] = self.sample_rate

        response = self.client.speak.v1.audio.generate(**kwargs)

        audio_bytes = self._response_to_bytes(response)

        if not audio_bytes:
            raise RuntimeError("Deepgram returned empty audio.")

        return audio_bytes