from __future__ import annotations

import ast
import json
import re
import time
from typing import Any, Optional

import requests


AGENTVERSE_HOSTING_URL = "https://agentverse.ai/v1/hosting/agents"
RELAY_AGENT_PREFIX = "voxaid-agentverse-relay"

_UUID_RE = re.compile(r"\bUUID\('([0-9a-f-]+)'\)")


class AgentverseChatError(RuntimeError):
    pass


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _list_hosted_agents(api_key: str) -> list[dict[str, Any]]:
    response = requests.get(
        AGENTVERSE_HOSTING_URL,
        headers=_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):
        items = data.get("items", data.get("agents", data.get("data", [])))
        return items if isinstance(items, list) else []

    if isinstance(data, list):
        return data

    return []


def _find_relay_agent(api_key: str) -> Optional[str]:
    try:
        agents = _list_hosted_agents(api_key)
    except Exception:
        return None

    for agent in agents:
        name = str(agent.get("name", ""))
        address = agent.get("address")

        if name.startswith(RELAY_AGENT_PREFIX) and address:
            return address

    return None


def _create_relay_agent(api_key: str) -> str:
    payload = {
        "name": RELAY_AGENT_PREFIX,
    }

    response = requests.post(
        AGENTVERSE_HOSTING_URL,
        headers=_headers(api_key),
        json=payload,
        timeout=30,
    )

    if response.status_code not in (200, 201):
        raise AgentverseChatError(
            f"Could not create Agentverse relay agent: "
            f"{response.status_code} {response.text[:300]}"
        )

    data = response.json()
    address = data.get("address")

    if not address:
        raise AgentverseChatError(
            f"Agentverse relay creation succeeded but no address was returned: {data}"
        )

    return address


def _find_or_create_relay(api_key: str) -> str:
    relay = _find_relay_agent(api_key)

    if relay:
        return relay

    return _create_relay_agent(api_key)


def _stop_agent(api_key: str, address: str) -> None:
    try:
        requests.post(
            f"{AGENTVERSE_HOSTING_URL}/{address}/stop",
            headers=_headers(api_key),
            timeout=30,
        )
    except Exception:
        pass


def _start_agent(api_key: str, address: str) -> None:
    response = requests.post(
        f"{AGENTVERSE_HOSTING_URL}/{address}/start",
        headers=_headers(api_key),
        timeout=30,
    )

    if response.status_code not in (200, 201, 202):
        raise AgentverseChatError(
            f"Could not start relay agent: {response.status_code} {response.text[:300]}"
        )


def _upload_code(api_key: str, address: str, code: str) -> None:
    files = [
        {
            "language": "python",
            "name": "agent.py",
            "value": code,
        }
    ]

    payload = {
        "code": json.dumps(files),
    }

    response = requests.put(
        f"{AGENTVERSE_HOSTING_URL}/{address}/code",
        headers=_headers(api_key),
        json=payload,
        timeout=30,
    )

    if response.status_code not in (200, 201, 204):
        raise AgentverseChatError(
            f"Could not upload relay agent code: "
            f"{response.status_code} {response.text[:300]}"
        )


def _get_logs(api_key: str, address: str) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            f"{AGENTVERSE_HOSTING_URL}/{address}/logs/latest",
            headers=_headers(api_key),
            timeout=30,
        )

        if response.status_code != 200:
            return []

        data = response.json()

        return data if isinstance(data, list) else []

    except Exception:
        return []


def _parse_result_entry(result_str: str) -> Any:
    try:
        return json.loads(result_str)
    except Exception:
        pass

    cleaned = _UUID_RE.sub(r"'\1'", result_str)

    try:
        return ast.literal_eval(cleaned)
    except Exception:
        return result_str


def _extract_results(logs: list[dict[str, Any]]) -> list[Any]:
    results = []

    for entry in sorted(logs, key=lambda x: x.get("log_timestamp", "")):
        msg = str(entry.get("log_entry", ""))

        if msg.startswith("RESULT:"):
            results.append(_parse_result_entry(msg[len("RESULT:"):]))

    return results


def _extract_text_from_result(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        if item.get("type") == "text" and item.get("text"):
            return str(item["text"]).strip()

        if item.get("text"):
            return str(item["text"]).strip()

        content = item.get("content")

        if isinstance(content, list):
            parts = [_extract_text_from_result(x) for x in content]
            return "\n".join([x for x in parts if x]).strip()

        if isinstance(content, str):
            return content.strip()

        if item.get("resource"):
            public_url = item.get("public_url")
            if public_url:
                return f"I generated a resource for you: {public_url}"

    return ""


def _extract_final_text(results: list[Any]) -> str:
    texts = []

    for item in results:
        text = _extract_text_from_result(item)

        if text:
            texts.append(text)

    return "\n".join(texts).strip()


def _build_relay_code(target_address: str, message: str, start_session: bool = False) -> str:
    escaped_message = (
        message.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )

    if start_session:
        startup_body = """
    ctx.logger.info("CHAT_STATUS:sending_session_start")
    await ctx.send(
        TARGET,
        ChatMessage(
            timestamp=datetime.now(),
            msg_id=uuid4(),
            content=[StartSessionContent(type="start-session")],
        ),
    )
    ctx.logger.info("CHAT_STATUS:session_start_sent")
    await asyncio.sleep(2)
    ctx.logger.info("CHAT_STATUS:sending")
    await ctx.send(
        TARGET,
        ChatMessage(
            timestamp=datetime.now(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=MESSAGE)],
        ),
    )
    ctx.logger.info("CHAT_STATUS:sent")
"""
        extra_imports = "import asyncio\n"
        start_session_import = "StartSessionContent,"
    else:
        startup_body = """
    ctx.logger.info("CHAT_STATUS:sending")
    await ctx.send(
        TARGET,
        ChatMessage(
            timestamp=datetime.now(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=MESSAGE)],
        ),
    )
    ctx.logger.info("CHAT_STATUS:sent")
"""
        extra_imports = ""
        start_session_import = ""

    return f'''
{extra_imports}
from datetime import datetime
from uuid import uuid4

from uagents import Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    {start_session_import}
    TextContent,
    chat_protocol_spec,
)

TARGET = "{target_address}"
MESSAGE = "{escaped_message}"

protocol = Protocol(spec=chat_protocol_spec)


@agent.on_event("startup")
async def send_msg(ctx: Context):
{startup_body}


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info("CHAT_STATUS:ack_received")


@protocol.on_message(ChatMessage)
async def handle_response(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info("CHAT_STATUS:response_received")

    for item in msg.content:
        try:
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif hasattr(item, "dict"):
                item_dict = item.dict()
            else:
                item_dict = str(item)

            ctx.logger.info("RESULT:" + str(item_dict))
        except Exception:
            ctx.logger.info("RESULT:" + repr(item))


agent.include(protocol, publish_manifest=True)
'''


class AgentverseRelayChatClient:
    def __init__(
        self,
        api_key: str,
        wait_seconds: int = 60,
        relay_agent_address: str = "",
        start_session: bool = False,
    ) -> None:
        self.api_key = api_key
        self.wait_seconds = wait_seconds
        self.relay_agent_address = relay_agent_address
        self.start_session = start_session

    def send_message(self, target_address: str, message: str) -> dict[str, Any]:
        if not self.api_key:
            raise AgentverseChatError("AGENTVERSE_API_KEY is missing.")

        if not target_address:
            raise AgentverseChatError("No target Agentverse agent address was provided.")

        if not message.strip():
            raise AgentverseChatError("Cannot send an empty message to Agentverse.")

        relay = self.relay_agent_address or _find_or_create_relay(self.api_key)

        _stop_agent(self.api_key, relay)
        time.sleep(2)

        relay_code = _build_relay_code(
            target_address=target_address,
            message=message,
            start_session=self.start_session,
        )

        _upload_code(self.api_key, relay, relay_code)
        time.sleep(1)

        _start_agent(self.api_key, relay)

        elapsed = 0
        poll_interval = 5
        results = []

        while elapsed < self.wait_seconds:
            time.sleep(poll_interval)
            elapsed += poll_interval

            logs = _get_logs(self.api_key, relay)
            results = _extract_results(logs)

            if results:
                break

        _stop_agent(self.api_key, relay)

        if not results:
            logs = _get_logs(self.api_key, relay)
            last_logs = [
                str(x.get("log_entry", ""))
                for x in sorted(logs, key=lambda y: y.get("log_timestamp", ""))
            ][-10:]

            return {
                "status": "timeout",
                "relay_agent": relay,
                "target_agent": target_address,
                "agent_text": "",
                "raw_results": [],
                "debug_logs": last_logs,
            }

        final_text = _extract_final_text(results)

        return {
            "status": "success",
            "relay_agent": relay,
            "target_agent": target_address,
            "agent_text": final_text,
            "raw_results": results,
            "debug_logs": [],
        }