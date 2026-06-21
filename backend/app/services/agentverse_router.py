from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic

from app.config import Settings
from app.services.agentverse_client import AgentverseSearchClient, AgentverseSearchError
from app.services.agentverse_relay_chat import AgentverseChatError, AgentverseRelayChatClient


class AgentverseRouter:
    """
    VoxAid middle agent.

    This version actually sends the task to the selected Agentverse marketplace agent.
    It does not show marketplace matches.
    It does not ask for confirmation.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.claude = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

        self.search_client = AgentverseSearchClient(
            api_key=settings.agentverse_api_key,
            search_url=settings.AGENTVERSE_SEARCH_URL,
            timeout_seconds=settings.AGENTVERSE_TIMEOUT_SECONDS,
        )

        self.chat_client = AgentverseRelayChatClient(
            api_key=settings.agentverse_api_key,
            wait_seconds=int(getattr(settings, "AGENTVERSE_CHAT_WAIT_SECONDS", 60)),
            relay_agent_address=getattr(settings, "AGENTVERSE_RELAY_AGENT_ADDRESS", ""),
            start_session=bool(getattr(settings, "AGENTVERSE_START_SESSION", False)),
        )

    def classify_task(self, corrected_text: str) -> dict[str, str]:
        text = (corrected_text or "").strip()

        if not text:
            return {
                "mode": "speech_only",
                "reason": "No corrected text was available.",
                "task_for_agent": "",
                "search_query": "",
            }

        system_prompt = (
            "You are VoxAid's accessibility router. The user may have dysarthria "
            "and motor limitations. Decide whether the corrected sentence is just "
            "something to speak aloud, or a digital task that should be sent to an "
            "Agentverse marketplace agent. Return strict JSON only."
        )

        user_prompt = f"""
Corrected user speech:
{text}

Rules:
- Use "speech_only" when the user is expressing a sentence, need, answer, thought, or message that should simply be spoken aloud.
- Use "agentverse_task" when the user asks to search, browse, summarize, compare, draft, write, schedule, fill a form, open/use a website, book/order/buy, or delegate a computer/digital task.
- Make "task_for_agent" a direct instruction to the selected marketplace agent.
- Make "search_query" short and targeted for finding the right marketplace agent.

Return JSON exactly:
{{
  "mode": "speech_only" or "agentverse_task",
  "reason": "short reason",
  "task_for_agent": "instruction for Agentverse agent, or empty string",
  "search_query": "marketplace search query, or empty string"
}}
""".strip()

        try:
            message = self.claude.messages.create(
                model=self.settings.ANTHROPIC_MODEL,
                max_tokens=240,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw = "".join(
                getattr(block, "text", "")
                for block in message.content
                if getattr(block, "text", None)
            ).strip()

            data = json.loads(raw)

        except Exception:
            lowered = text.lower()
            task_markers = [
                "search",
                "find",
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
                "mode": "agentverse_task" if is_task else "speech_only",
                "reason": "Fallback keyword routing was used.",
                "task_for_agent": text if is_task else "",
                "search_query": text if is_task else "",
            }

        mode = data.get("mode", "speech_only")

        if mode not in {"speech_only", "agentverse_task"}:
            mode = "speech_only"

        return {
            "mode": mode,
            "reason": str(data.get("reason", "")).strip(),
            "task_for_agent": str(data.get("task_for_agent", "")).strip(),
            "search_query": str(data.get("search_query", "")).strip(),
        }

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

    def _normalize_agent(self, raw_agent: dict[str, Any]) -> dict[str, Any]:
        address = str(
            raw_agent.get("address")
            or raw_agent.get("agent_address")
            or raw_agent.get("address_prefix")
            or raw_agent.get("identifier")
            or ""
        )

        name = str(
            raw_agent.get("name")
            or raw_agent.get("title")
            or raw_agent.get("agent_name")
            or "Agentverse Agent"
        )

        description = str(
            raw_agent.get("description")
            or raw_agent.get("short_description")
            or raw_agent.get("summary")
            or raw_agent.get("readme")
            or ""
        )

        protocols = raw_agent.get("protocols") or raw_agent.get("protocol_digests") or []

        if not isinstance(protocols, list):
            protocols = []

        return {
            "name": name,
            "address": address,
            "description": description[:700],
            "url": f"https://agentverse.ai/agents/{address}" if address else "https://agentverse.ai/",
            "score": raw_agent.get("score") or raw_agent.get("rating") or raw_agent.get("rank"),
            "status": raw_agent.get("status") or raw_agent.get("state") or raw_agent.get("agent_state"),
            "developer": raw_agent.get("developer") or raw_agent.get("author") or raw_agent.get("owner"),
            "protocols": protocols,
        }

    def route(self, corrected_text: str) -> dict[str, Any]:
        corrected_text = (corrected_text or "").strip()
        decision = self.classify_task(corrected_text)

        result = self._empty_result()
        result.update(
            {
                "mode": decision["mode"],
                "reason": decision["reason"],
                "task_for_agent": decision["task_for_agent"],
                "search_query": decision["search_query"],
            }
        )

        if decision["mode"] == "speech_only":
            result["action_status"] = "none"
            result["agent_text"] = corrected_text
            result["spoken_summary"] = corrected_text
            return result

        search_query = decision["search_query"] or decision["task_for_agent"] or corrected_text
        task_for_agent = decision["task_for_agent"] or corrected_text

        try:
            raw_agent = self.search_client.search_top_agent(
                query=search_query,
                protocol_digest=self.settings.AGENTVERSE_PROTOCOL_DIGEST or None,
            )

            if not raw_agent:
                result["action_status"] = "no_agent_found"
                result["agent_text"] = (
                    "I understood the task, but I could not find an Agentverse marketplace agent for it."
                )
                result["spoken_summary"] = result["agent_text"]
                return result

            selected_agent = self._normalize_agent(raw_agent)
            target_address = selected_agent.get("address", "")

            if not target_address:
                result["action_status"] = "no_agent_found"
                result["agent_text"] = (
                    "I found an Agentverse listing, but it did not include a usable agent address."
                )
                result["spoken_summary"] = result["agent_text"]
                return result

            chat_result = self.chat_client.send_message(
                target_address=target_address,
                message=task_for_agent,
            )

            agent_text = (chat_result.get("agent_text") or "").strip()

            if not agent_text:
                result["action_status"] = "agent_timeout"
                result["selected_agent"] = selected_agent
                result["agent_text"] = (
                    "I routed your request to the selected Agentverse marketplace agent, "
                    "but it did not return a response in time."
                )
                result["spoken_summary"] = result["agent_text"]
                result["search_error"] = json.dumps(chat_result.get("debug_logs", []))[:1000]
                return result

            result["mode"] = "agentverse_task"
            result["action_status"] = "completed"
            result["selected_agent"] = selected_agent
            result["task_for_agent"] = task_for_agent
            result["search_query"] = search_query
            result["agent_text"] = agent_text
            result["spoken_summary"] = agent_text

            return result

        except AgentverseSearchError as exc:
            result["action_status"] = "search_failed"
            result["search_error"] = str(exc)
            result["agent_text"] = "Agentverse search failed. Please check the API key and Agentverse configuration."
            result["spoken_summary"] = result["agent_text"]
            return result

        except AgentverseChatError as exc:
            result["action_status"] = "agent_chat_failed"
            result["search_error"] = str(exc)
            result["agent_text"] = "I found an Agentverse agent, but the chat handoff failed."
            result["spoken_summary"] = result["agent_text"]
            return result