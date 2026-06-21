from __future__ import annotations

from openai import OpenAI


class ASIOneClient:
    """
    Direct ASI:One chat client.

    This replaces:
    - Agentverse manual marketplace search
    - relay agent
    - mailbox/log polling
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.asi1.ai/v1",
        model: str = "asi1",
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("ASI_ONE_API_KEY is missing.")

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    def complete_task(
        self,
        corrected_text: str,
        task_for_agent: str,
    ) -> str:
        system_prompt = """
You are ASI:One inside VoxAid, a voice-first accessibility assistant for people with dysarthria and motor limitations.

The user can only use voice. They may not be able to type, click, scroll, or manually operate websites.

Your job:
- Understand the user's corrected speech.
- If it is a normal sentence, respond naturally.
- If it is a task, help complete it as much as possible.
- Use agentic reasoning and available agent capabilities when useful.
- Do not mention internal routing, marketplace search, model names, agent addresses, or APIs.
- Do not say "I found an Agentverse agent."
- Do not return raw agent metadata.
- Do not repeat the user's command back.
- Give the final useful answer only.

Safety:
- If the task requires missing information, ask for the missing information conversationally.
- For irreversible actions like sending, buying, booking, submitting forms, or changing accounts, prepare the next step or draft instead of claiming it was completed.
- For location-based tasks like "nearest library", if no location is provided, ask for the user's city, ZIP code, or current location.

Keep the response short, clear, and spoken-friendly.
""".strip()

        user_prompt = f"""
Corrected user speech:
{corrected_text}

Clean task:
{task_for_agent}

Return the answer VoxAid should speak to the user.
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.2,
            max_tokens=700,
        )

        text = response.choices[0].message.content or ""
        return text.strip()