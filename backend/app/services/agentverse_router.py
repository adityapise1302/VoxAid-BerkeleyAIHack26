from __future__ import annotations

import json
import re
from typing import Any

from anthropic import Anthropic

from app.config import Settings
from app.services.asi_one_client import ASIOneClient


AGENT_ADDRESS_RE = re.compile(r"@?(agent1[a-zA-Z0-9]+)")


class AgentverseRouter:
    """
    Backward-compatible router name.

    Actual flow:
    1. Claude cleans/classifies user intent.
    2. ASI:One handles the agentic task.
    3. Claude rephrases ASI:One's output into a clean spoken answer.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.claude = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        self.asi_one = ASIOneClient(
            api_key=settings.ASI_ONE_API_KEY,
            base_url=settings.ASI_ONE_BASE_URL,
            model=settings.ASI_ONE_MODEL,
            timeout_seconds=settings.ASI_ONE_TIMEOUT_SECONDS,
        )

    def _remove_agent_addresses(self, text: str) -> str:
        text = AGENT_ADDRESS_RE.sub("", text or "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _dedupe_repeated_command(self, text: str) -> str:
        text = (text or "").strip()

        if not text:
            return ""

        lowered = text.lower()

        repeated_patterns = [
            "find me the nearest library",
            "find me nearest library",
            "give me the nearest library",
            "give me nearest library",
            "find nearest library",
            "nearest library",
        ]

        matches = [pattern for pattern in repeated_patterns if pattern in lowered]

        if "library" in lowered and matches:
            return "Find the nearest library."

        parts = re.split(r"[.!?]+", text)
        seen = set()
        cleaned_parts = []

        for part in parts:
            normalized = re.sub(r"\s+", " ", part.strip().lower())

            if not normalized:
                continue

            if normalized not in seen:
                seen.add(normalized)
                cleaned_parts.append(part.strip())

        if cleaned_parts:
            return ". ".join(cleaned_parts).strip()

        return text

    def _clean_text(self, text: str) -> str:
        text = self._remove_agent_addresses(text)
        text = self._dedupe_repeated_command(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _empty_result(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": "speech_only",
            "action_status": "none",
            "reason": "",
            "task_for_agent": "",
            "search_query": "",
            "selected_agent": None,
            "agent_text": "",
            "spoken_summary": "",
            "search_error": None,
        }

    def classify_task(self, corrected_text: str) -> dict[str, str]:
        cleaned_text = self._clean_text(corrected_text)

        if not cleaned_text:
            return {
                "mode": "speech_only",
                "reason": "No corrected text was available.",
                "task_for_agent": "",
                "search_query": "",
            }

        system_prompt = """
You are VoxAid's middle routing agent.

The user may have dysarthria and motor limitations. Decide whether the corrected sentence is:
1. speech_only: something the user simply wants spoken aloud
2. agentic_task: something the user wants an assistant/agent to help accomplish

Return strict JSON only.
""".strip()

        user_prompt = f"""
Corrected and cleaned user speech:
{cleaned_text}

Rules:
- speech_only: greetings, emotional statements, direct sentences, simple communication, or anything that just needs to be spoken aloud.
- agentic_task: searching, browsing, finding nearby places, summarizing, comparing, drafting, writing messages/emails, scheduling, filling forms, planning, booking, ordering, buying, opening websites, or getting live/helpful information.
- Make task_for_agent a clean direct instruction.
- Do not include Agentverse addresses.
- Do not repeat the same instruction multiple times.

Return JSON exactly:
{{
  "mode": "speech_only" or "agentic_task",
  "reason": "short reason",
  "task_for_agent": "clean instruction, or empty string"
}}
""".strip()

        try:
            message = self.claude.messages.create(
                model=self.settings.ANTHROPIC_MODEL,
                max_tokens=240,
                temperature=0,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            raw = "".join(
                getattr(block, "text", "")
                for block in message.content
                if getattr(block, "text", None)
            ).strip()

            data = json.loads(raw)

        except Exception:
            lowered = cleaned_text.lower()

            task_markers = [
                "search",
                "find",
                "nearest",
                "near me",
                "look up",
                "browse",
                "summarize",
                "compare",
                "draft",
                "write an email",
                "write a message",
                "schedule",
                "book",
                "order",
                "buy",
                "fill",
                "form",
                "open",
                "website",
                "calendar",
                "apply",
                "reserve",
            ]

            is_task = any(marker in lowered for marker in task_markers)

            data = {
                "mode": "agentic_task" if is_task else "speech_only",
                "reason": "Fallback keyword routing was used.",
                "task_for_agent": cleaned_text if is_task else "",
            }

        mode = data.get("mode", "speech_only")

        if mode not in {"speech_only", "agentic_task"}:
            mode = "speech_only"

        task_for_agent = self._clean_text(
            str(data.get("task_for_agent", "")).strip()
        )

        return {
            "mode": mode,
            "reason": str(data.get("reason", "")).strip(),
            "task_for_agent": task_for_agent,
            "search_query": task_for_agent,
        }

    def _rephrase_asi_one_output(
        self,
        original_corrected_text: str,
        cleaned_user_text: str,
        task_for_agent: str,
        raw_asi_output: str,
    ) -> str:
        """
        Final Claude rephrasing layer.

        ASI:One handles the agentic task.
        Claude converts ASI:One's output into the final answer VoxAid should show/speak.
        """

        raw_asi_output = self._remove_agent_addresses(raw_asi_output or "").strip()

        if not raw_asi_output:
            return "I understood your request, but ASI:One did not return a usable answer."

        system_prompt = """
You are VoxAid's final response agent.

Your job is to rephrase ASI:One's raw output into a clean, natural answer for a voice-only user with dysarthria and motor limitations.

Rules:
- Return only the final answer the user should hear.
- Do not mention ASI:One, Agentverse, marketplace agents, APIs, routing, or internal tools.
- Do not include agent addresses.
- Do not repeat the user's command.
- Do not include debug text or metadata.
- If ASI:One asks for missing information, rephrase that as a short conversational question.
- If ASI:One gives useful information, summarize it clearly.
- If the task is location-based and location is missing, ask for city, ZIP code, or current location.
- Keep it concise, spoken-friendly, and helpful.
""".strip()

        user_prompt = f"""
Original corrected user speech:
{original_corrected_text}

Cleaned user speech:
{cleaned_user_text}

Task sent to ASI:One:
{task_for_agent}

Raw ASI:One output:
{raw_asi_output}

Now produce the final VoxAid response to speak to the user.
""".strip()

        try:
            message = self.claude.messages.create(
                model=self.settings.ANTHROPIC_MODEL,
                max_tokens=400,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            final_text = "".join(
                getattr(block, "text", "")
                for block in message.content
                if getattr(block, "text", None)
            ).strip()

            final_text = self._remove_agent_addresses(final_text)
            final_text = re.sub(r"\s+", " ", final_text).strip()

            return final_text or raw_asi_output

        except Exception:
            return raw_asi_output

    def route(self, corrected_text: str) -> dict[str, Any]:
        original_corrected_text = corrected_text or ""
        cleaned_text = self._clean_text(original_corrected_text)

        result = self._empty_result()

        decision = self.classify_task(cleaned_text)

        result["reason"] = decision["reason"]
        result["task_for_agent"] = decision["task_for_agent"]
        result["search_query"] = decision["search_query"]

        if decision["mode"] == "speech_only":
            result["mode"] = "speech_only"
            result["action_status"] = "none"
            result["agent_text"] = cleaned_text
            result["spoken_summary"] = cleaned_text
            return result

        task_for_agent = decision["task_for_agent"] or cleaned_text

        try:
            raw_asi_response = self.asi_one.complete_task(
                corrected_text=cleaned_text,
                task_for_agent=task_for_agent,
            )

            final_response = self._rephrase_asi_one_output(
                original_corrected_text=original_corrected_text,
                cleaned_user_text=cleaned_text,
                task_for_agent=task_for_agent,
                raw_asi_output=raw_asi_response,
            )

            final_response = self._remove_agent_addresses(final_response)
            final_response = final_response.strip()

            if not final_response:
                final_response = "I understood your request, but I could not generate a useful response."

            result["mode"] = "agentverse_task"
            result["action_status"] = "completed"
            result["selected_agent"] = None
            result["agent_text"] = final_response
            result["spoken_summary"] = final_response
            return result

        except Exception as exc:
            result["mode"] = "agentverse_task"
            result["action_status"] = "asi_one_failed"
            result["search_error"] = str(exc)
            result["agent_text"] = (
                "I could not complete the request. Please check your ASI:One API key and backend logs."
            )
            result["spoken_summary"] = result["agent_text"]
            return result