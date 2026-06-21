from __future__ import annotations

from typing import Any

import httpx


class AgentverseSearchError(RuntimeError):
    pass


class AgentverseSearchClient:
    """
    Searches Agentverse Marketplace and returns the top marketplace agent.

    This does not show matches to the user. The router uses the first result only.
    """

    def __init__(
        self,
        api_key: str = "",
        search_url: str = "https://agentverse.ai/v1/search/agents",
        timeout_seconds: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.search_url = search_url
        self.timeout_seconds = timeout_seconds

    def search_top_agent(
        self,
        query: str,
        protocol_digest: str | None = None,
    ) -> dict[str, Any] | None:
        agents = self.search_agents(
            query=query,
            limit=1,
            protocol_digest=protocol_digest,
        )
        return agents[0] if agents else None

    def search_agents(
        self,
        query: str,
        limit: int = 1,
        protocol_digest: str | None = None,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()

        if not query:
            return []

        limit = max(1, min(int(limit), 5))

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-Api-Key"] = self.api_key

        filters: dict[str, Any] = {}

        if protocol_digest:
            filters["protocol_digest"] = [protocol_digest]

        payload = {
            "search_text": query,
            "sort": "relevancy",
            "filters": filters,
            "direction": "asc",
            "offset": 0,
            "limit": limit,
        }

        try:
            response = httpx.post(
                self.search_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AgentverseSearchError(
                f"Agentverse search failed with status "
                f"{exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.RequestError as exc:
            raise AgentverseSearchError(
                f"Agentverse search request failed: {exc}"
            ) from exc

        data = response.json()

        if isinstance(data, list):
            return data

        if not isinstance(data, dict):
            return []

        agents = (
            data.get("agents")
            or data.get("ais")
            or data.get("results")
            or data.get("data")
            or []
        )

        if not isinstance(agents, list):
            return []

        return agents