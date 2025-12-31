import os
import argparse
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import httpx
import random

from dotenv import load_dotenv
from mistralai import Mistral  # only Mistral, no AsyncClient

import json

################################################################

DDG_API_URL = "https://api.duckduckgo.com/"
#DDG_API_URL = os.getenv("DDG_API_URL","https://api.duckduckgo.com/").

@dataclass()
class ddg:
    title: str
    text: str
    url: str


def strip_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


async def get_json_with_retries(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    timeout: float = 8.0,
    max_retries: int = 3,
) -> Dict[str, Any]:
    back_off = 0.5

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.get(url, params=params, headers=headers)

                if resp.status_code in {429, 500, 502, 503, 504}:
                    if attempt == max_retries:
                        resp.raise_for_status()
                    await asyncio.sleep(back_off + random.uniform(0, 0.2))
                    back_off = back_off * 2
                    continue

                resp.raise_for_status()
                return resp.json()

            except (httpx.TimeoutException, httpx.ConnectError):
                if attempt == max_retries:
                    raise
                await asyncio.sleep(back_off + random.uniform(0, 0.2))
                back_off = back_off * 2

            except ValueError:
                raise RuntimeError(f"Failed to parse JSON from {url}")

    raise RuntimeError(f"Failed to connect to {url}")


async def retrieve_ddg(
    query: str,
    top_k: int = 3,
    timeout: float = 8.0,
) -> List[ddg]:
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

    data = await get_json_with_retries(DDG_API_URL, params, headers, timeout=timeout)

    docs: List[ddg] = []

    # Best signal: Abstract
    abstract_text = strip_whitespace(data.get("AbstractText", ""))
    if abstract_text:
        docs.append(
            ddg(
                title=strip_whitespace(data.get("Heading", "")) or "DuckDuckGo Abstract",
                text=abstract_text,
                url=strip_whitespace(data.get("AbstractURL", "")),
            )
        )

    # Fallback: RelatedTopics (can be nested)
    def add_topic(t: dict) -> None:
        nonlocal docs
        if len(docs) >= top_k:
            return
        text = strip_whitespace(t.get("Text", ""))
        if not text:
            return
        docs.append(
            ddg(
                title=(text[:80] + "…") if len(text) > 80 else text,
                text=text,
                url=strip_whitespace(t.get("FirstURL", "")),
            )
        )

    for topic in data.get("RelatedTopics", []) or []:
        if len(docs) >= top_k:
            break
        if isinstance(topic, dict) and "Topics" in topic and isinstance(topic["Topics"], list):
            for sub in topic["Topics"]:
                if len(docs) >= top_k:
                    break
                if isinstance(sub, dict):
                    add_topic(sub)
        elif isinstance(topic, dict):
            add_topic(topic)

    return docs


async def retrieve_ddg_tool(query: str) -> str:
    docs = await retrieve_ddg(query)
    return json.dumps(
        {
            "results": [
                {"title": d.title, "url": d.url, "content": d.text}
                for d in docs
            ]
        }
    )



################################################################



class MistralClient:
    def __init__(self, api_key: str, model: str = "mistral-large-latest") -> None:
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is missing!")
        self.client = Mistral(api_key=api_key)
        self.model = model

    async def answer_with_llm(self, question: str) -> str:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": question},
        ]

        # Call the async method
        res = await self.client.chat.complete_async(
            model=self.model,
            messages=messages,
            temperature=0.7,
            stream=False,
        )

        # Parse content
        return res.choices[0].message.content

async def main_async() -> None:
    print("INFO: Running async main")

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("ERROR: No API key provided")
        return

    parser = argparse.ArgumentParser(description="mistral async demo")
    parser.add_argument("--mode", choices=["llm", "rag"], required=True)
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    client = MistralClient(api_key=api_key)

    if args.mode == "llm":
        answer = await client.answer_with_llm(question=args.question)
    else:
        raise RuntimeError("RAG mode not implemented")

    print(f"ANSWER:\n{answer}")

if __name__ == "__main__":
    asyncio.run(main_async())
