import os
import sys
import re
import random
import time
import json
import argparse
import math
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

@dataclass()
class ddg:
    title: str
    text: str
    url: str

def strip_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

#####################################################################################
# Async HTTP utilities
#####################################################################################

async def get_json_with_retries_async(
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

async def retrieve_ddg_async(
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

    data = await get_json_with_retries_async(DDG_API_URL, params, headers, timeout=timeout)
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

async def retrieve_ddg_tool_async(query: str) -> str:
    docs = await retrieve_ddg_async(query)
    return json.dumps({
        "results": [
            {"title": d.title, "url": d.url, "content": d.text}
            for d in docs
        ]
    })

#####################################################################################
# Calculator (sync is fine; it's CPU-local)
#####################################################################################

def calculator_tool(expression: str) -> str:
    """
    Dummy calculator tool.
    VERY IMPORTANT: never use eval in production.
    This is purely for interview/demo purposes.
    """
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"expression": expression, "error": str(e)})

#####################################################################################
# Prompt helpers
#####################################################################################

def system_prompt_docs() -> str:
    return (
        "You are a helpful assistant.\n"
        "Use ONLY the provided context to answer.\n"
        "If the context does not contain enough information, say what is missing.\n"
        "Cite sources like [Source 1], [Source 2] next to the claims they support.\n"
        "Do not invent citations."
    )

def system_prompt_tools() -> str:
    return (
        "You are a helpful assistant.\n"
        "Use ONLY the provided context to answer.\n"
        "If the context does not contain enough information, say what is missing.\n"
        "Cite sources like [Source 1], [Source 2] next to the claims they support.\n"
        "Use tools when required.\n"
        "Do not invent citations."
    )

def build_context_from_docs(docs: List[ddg], max_char: int = 2000) -> str:
    blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        url = d.url or ""
        blocks.append(f"[Source {i}]:\nTitle:{d.title}\nText:{d.text}\nURL:{url}")
    return "\n".join(blocks).strip()[:max_char]

#####################################################################################
# Mistral client wrapper (async orchestration)
#####################################################################################

class MistralClient:
    def __init__(self, api_key: str, model: str = "mistral-large-latest") -> None:
        self.client = Mistral(api_key=api_key)
        self.model = model

    async def answer_with_llm_async(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "you are helpful assistant"},
            {"role": "user", "content": question},
        ]

        # If SDK is sync-only in your env, run in a thread:
        res = await asyncio.to_thread(
            self.client.chat.complete,
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return res.choices[0].message.content

    async def answer_with_context_async(self, question: str, context: str) -> str:
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
        return res.choices[0].message.content

    async def answer_with_tools_async(self, question: str) -> str:
        TOOL_REGISTRY_ASYNC = {
            "search": retrieve_ddg_tool_async,   # async tool
            "calculator": None,                 # sync tool handled separately
        }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search DuckDuckGo Instant Answer API for factual information.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a mathematical expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            },
        ]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt_tools()},
            {"role": "user", "content": question},
        ]

        for _ in range(5):
            res = await asyncio.to_thread(
                self.client.chat.complete,
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                response_format={"type": "text"},
                parallel_tool_calls=False,
            )

            msg = res.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            # Append assistant message in cookbook shape
            assistant_entry: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_entry)

            if not tool_calls:
                return msg.content or ""

            # Execute tool calls
            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments or "{}")

                if fn_name == "calculator":
                    tool_output = calculator_tool(**fn_args)
                else:
                    tool_fn = TOOL_REGISTRY_ASYNC.get(fn_name)
                    if tool_fn is None:
                        tool_output = json.dumps({"error": f"Unsupported tool: {fn_name}"})
                    else:
                        tool_output = await tool_fn(**fn_args)

                messages.append(
                    {
                        "role": "tool",
                        "name": fn_name,
                        "tool_call_id": tc.id,
                        "content": tool_output,
                    }
                )

        return "I couldn't finish tool execution within the allowed steps."

#####################################################################################
# CLI
#####################################################################################

async def main_async() -> None:
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Missing MISTRAL_API_KEY environment variable")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="mistral tools (async)")
    parser.add_argument("--mode", choices=["llm", "rag", "tool"], required=True)
    parser.add_argument("--question", type=str, required=True)

    args = parser.parse_args()

    client = MistralClient(api_key=api_key)

    if args.mode == "llm":
        answer = await client.answer_with_llm_async(args.question)

    elif args.mode == "rag":
        docs = await retrieve_ddg_async(args.question)
        context = build_context_from_docs(docs)
        answer = await client.answer_with_context_async(args.question, context)

    elif args.mode == "tool":
        answer = await client.answer_with_tools_async(args.question)

    else:
        raise RuntimeError("Invalid mode")

    print("\n----- ANSWER -----\n")
    print(answer)

def main() -> None:
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
