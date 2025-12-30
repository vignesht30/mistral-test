import os
import sys
import re
import random
import time
import json
import argparse
import asyncio

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from mistralai import Mistral
import httpx

DDG_API_URL = "https://api.duckduckgo.com/"

#####################################################################################
# Data model
#####################################################################################

@dataclass
class DuckDoc:
    title: str
    text: str
    url: str

def strip_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

#####################################################################################
# Async HTTP helper (kept, but we will return static response for reliability)
#####################################################################################

async def get_json_with_retries_async(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    timeout: float = 8.0,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Async HTTP GET with retries + exponential backoff.
    (In this interview-friendly version, retrieve_duckduckgo_async uses a static fallback,
     but we keep this function as requested.)
    """
    back_off = 0.5
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.get(url, params=params, headers=headers)

                if resp.status_code in {429, 500, 502, 503, 504}:
                    if attempt == max_retries:
                        resp.raise_for_status()
                    await asyncio.sleep(back_off + random.uniform(0, 0.2))
                    back_off *= 2
                    continue

                resp.raise_for_status()
                return resp.json()

            except (httpx.TimeoutException, httpx.ConnectError):
                if attempt == max_retries:
                    raise
                await asyncio.sleep(back_off + random.uniform(0, 0.2))
                back_off *= 2

            except ValueError:
                raise RuntimeError(f"Failed to parse JSON from {url}")

    raise RuntimeError(f"Failed to fetch JSON from {url}")

#####################################################################################
# Retrieval (static fallback for interview reliability)
#####################################################################################

async def retrieve_duckduckgo_async(query: str, top_k: int = 3) -> List[DuckDoc]:
    """
    For interview reliability:
      - Return deterministic/static docs for common demo questions (no dependency on DDG uptime/schema).
      - Optionally try real DDG if you want later (kept easy to toggle).
    """
    q = strip_whitespace(query).lower()

    # STATIC DEMO RESPONSES (add more if you want)
    if "capital of india" in q or "what is capital of india" in q:
        return [
            DuckDoc(
                title="Capital of India",
                text="The capital of India is New Delhi.",
                url="https://en.wikipedia.org/wiki/New_Delhi",
            )
        ][:top_k]

    # Generic fallback static doc (still lets RAG flow work)
    return [
        DuckDoc(
            title="No static hit",
            text="No static retrieval match was found for this query in the interview demo mode.",
            url="",
        )
    ][:top_k]

    # If you want to try real DDG later, uncomment and parse:
    # params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
    # headers = {"User-Agent": "MisRAGPractice/1.0", "Accept": "application/json"}
    # data = await get_json_with_retries_async(DDG_API_URL, params, headers)
    # ...parse AbstractText/RelatedTopics into DuckDoc...

#####################################################################################
# Context building + prompts
#####################################################################################

def build_context_from_docs(docs: List[DuckDoc], max_chars: int = 2000) -> str:
    blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        blocks.append(
            f"[Source {i}]\n"
            f"Title: {d.title}\n"
            f"URL: {d.url}\n"
            f"Content: {d.text}\n"
        )
    return "\n".join(blocks).strip()[:max_chars]

def system_prompt_docs() -> str:
    return (
        "You are a helpful assistant.\n"
        "Use ONLY the provided context to answer.\n"
        "If the context does not contain enough information, say what is missing.\n"
        "Cite sources like [Source 1], [Source 2] next to the claims they support.\n"
        "Do not invent citations."
    )

#####################################################################################
# Mistral client wrapper (async orchestration)
#####################################################################################

class MistralClient:
    def __init__(self, api_key: str, model: str = "mistral-large-latest") -> None:
        self.client = Mistral(api_key=api_key)
        self.model = model

    async def answer_llm_async(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        # Wrap sync SDK call so event loop stays non-blocking
        res = await asyncio.to_thread(
            self.client.chat.complete,
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return res.choices[0].message.content or ""

    async def answer_rag_async(self, question: str) -> str:
        docs = await retrieve_duckduckgo_async(question)
        context = build_context_from_docs(docs)

        messages = [
            {"role": "system", "content": system_prompt_docs()},
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": question},
        ]

        res = await asyncio.to_thread(
            self.client.chat.complete,
            model=self.model,
            messages=messages,
            temperature=0.3,
        )
        return res.choices[0].message.content or ""

#####################################################################################
# CLI
#####################################################################################

async def main_async() -> None:
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Missing MISTRAL_API_KEY environment variable")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Async LLM + RAG demo (static retrieval)")
    parser.add_argument("--mode", choices=["llm", "rag"], required=True, help="Execution mode")
    parser.add_argument("--question", type=str, required=True, help="Question")

    args = parser.parse_args()
    client = MistralClient(api_key=api_key)

    if args.mode == "llm":
        answer = await client.answer_llm_async(args.question)
    else:
        answer = await client.answer_rag_async(args.question)

    print("\n----- ANSWER -----\n")
    print(answer)

def main() -> None:
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
