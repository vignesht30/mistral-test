import json
import httpx
from dataclasses import dataclass
from typing import List, Dict, Any

DDG_API_URL = "https://api.duckduckgo.com/"


@dataclass
class DuckDoc:
    title: str
    text: str
    url: str


def retrieve_duckduckgo(query: str, timeout: float = 8.0) -> List[DuckDoc]:
    """
    Retrieve information from DuckDuckGo Instant Answer API.
    """
    params = {
        "q": query,
        "format": "json",
        "no_redirect": 1,
        "no_html": 1,
    }

    headers = {
        "User-Agent": "MisRAGPractice/1.0",
        "Accept": "application/json",
    }

    with httpx.Client(timeout=timeout, headers=headers) as client:
        resp = client.get(DDG_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    docs: List[DuckDoc] = []

    # Primary abstract (best signal)
    if data.get("AbstractText"):
        docs.append(
            DuckDoc(
                title=data.get("Heading", "DuckDuckGo Result"),
                text=data["AbstractText"],
                url=data.get("AbstractURL", ""),
            )
        )

    # Fallback: related topics
    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and topic.get("Text"):
            docs.append(
                DuckDoc(
                    title=topic.get("Text", "")[:80],
                    text=topic.get("Text", ""),
                    url=topic.get("FirstURL", ""),
                )
            )

        if len(docs) >= 3:
            break

    return docs


def duckduckgo_tool(query: str) -> str:
    """
    Tool wrapper: returns JSON string for LLM tool calling.
    """
    docs = retrieve_duckduckgo(query)

    payload: Dict[str, Any] = {
        "query": query,
        "results": [
            {
                "title": d.title,
                "url": d.url,
                "content": d.text,
            }
            for d in docs
        ],
    }

    return json.dumps(payload, ensure_ascii=False)
