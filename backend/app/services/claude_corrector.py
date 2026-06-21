from anthropic import Anthropic


class ClaudeCorrector:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def correct(self, raw_transcript: str) -> str:
        raw_transcript = (raw_transcript or "").strip()

        if not raw_transcript:
            return ""

        system_prompt = (
            "You are correcting speech-to-text output from a person with dysarthria. "
            "Preserve the speaker's intended meaning. "
            "Fix likely transcription errors, missing words, and grammar only when strongly implied. "
            "Do not add new facts. "
            "Do not ask for confirmation. "
            "Return only one natural sentence that can be spoken aloud. "
            "Do not include explanations, labels, JSON, or quotation marks."
        )

        user_prompt = f"Raw transcript:\n{raw_transcript}\n\nCorrected sentence:"

        message = self.client.messages.create(
            model=self.model,
            max_tokens=120,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        parts = []
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)

        corrected = " ".join(parts).strip()
        corrected = corrected.strip('"').strip("'").strip()

        return corrected or raw_transcript