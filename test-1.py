import os
import sys
import re
import random
import time
import json
import argparse
import math

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

from mistralai import Mistral
import httpx

DDG_API_URL = "https://api.duckduckgo.com/"
#DDG_API_URL = os.getenv("DDG_API_URL","https://api.duckduckgo.com/").

#####################################################################################
# MistralClient custom class
#####################################################################################

class MistralClient:
    def __init__(self, api_key: str, model:str = "mistral-large-latest") -> None:
        self.client = Mistral(api_key=api_key)
        self.model = model

    def answer_with_llm(self, question:str) -> str:

        messages = [
            {"role":"system", "content": "you are helpful assistant"},
            {"role":"user", "content": question},
        ]

        res = self.client.chat.complete(
            model = self.model,
            messages = messages,
            temperature=0.7,
        )

        return res.choices[0].message.content

    def answer_with_context(self, question:str, context:str) -> str:

        messages = [
            {"role":"system", "content": system_prompt_docs()},
            {"role":"system", "content": f"Context:\n {context}"},
            {"role":"user", "content": question},
        ]

        res = self.client.chat.complete(
            model = self.model,
            messages = messages,
            temperature=0.3,
        )

        return res.choices[0].message.content

    def answer_with_tools(self, question: str) -> str:
        TOOL_REGISTRY = {
            "search": retrieve_ddg_tool,
            "calculator": calculator_tool,
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

        # Cookbook-style loop: keep calling until model stops requesting tools
        for _ in range(5):  # safety cap to avoid infinite loops
            print("looping...")
            res = self.client.chat.complete(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # change to "any" if you want to force a tool
                temperature=0.2,
                response_format={"type": "text"},
                parallel_tool_calls=False,  # keeps things simpler/consistent
            )

            msg = res.choices[0].message
            print("==============================================")
            print(msg)
            tool_calls = getattr(msg, "tool_calls", None) or []

            # Append assistant message exactly like cookbook does
            # (this preserves tool_calls in the assistant message)
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ] if tool_calls else None,
                }
            )
            # Remove tool_calls key if none (cleaner payload)
            if messages[-1].get("tool_calls") is None:
                messages[-1].pop("tool_calls", None)

            # If no tool calls, we're done
            if not tool_calls:
                return msg.content or ""

            # Execute each tool call and append tool results
            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments or "{}")

                tool_fn = TOOL_REGISTRY.get(fn_name)
                if tool_fn is None:
                    # Return an error tool message so model can recover
                    tool_output = json.dumps({"error": f"Unsupported tool: {fn_name}"})
                else:
                    tool_output = tool_fn(**fn_args)

                print(f"tool_output -->> {tool_output}")
                messages.append(
                    {
                        "role": "tool",
                        "name": fn_name,
                        "tool_call_id": tc.id,
                        "content": tool_output,
                    }
                )

        # If we hit the safety cap, return a graceful message
        return "I couldn't finish tool execution within the allowed steps."

#####################################################################################
# retrieve data - call api functions
#####################################################################################

@dataclass()
class ddg:
    title:str
    text:str
    url:str

def strip_whitespace(s: str) -> str:
    return re.sub(r"\s+"," ",(s or "")).strip()

def get_json_with_retries(
        url:str,
        params:Optional[Dict[str, Any]]=None,
        headers:Optional[Dict[str, Any]]=None,
        timeout:float=8.0,
        max_retries:int=3,
) -> Dict[str, Any]:
    back_off = 0.5

    with httpx.Client(timeout=timeout) as client:
        for attempt in range(1, max_retries +1 ):
            try:
                resp = client.get(url, params=params, headers=headers)

                if resp.status_code in {429, 500,502,503,504}:
                    if attempt == max_retries:
                        resp.raise_for_status()
                    time.sleep(back_off + random.uniform(0,0.2))
                    back_off = back_off * 2
                    continue

                resp.raise_for_status()
                return resp.json()

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt == max_retries:
                    raise
                time.sleep(back_off + random.uniform(0, 0.2))
                back_off = back_off * 2

            except ValueError:
                raise RuntimeError(f"Failed to connect to {url}")

    raise  RuntimeError(f"Failed to connect to {url}")

def retrieve_ddg(
        query:str,
        top_k: int = 3,
        timeout:float=8.0,
) -> List[ddg]:

    ############### simple one ####################
    # params = {
    #     "q": query,
    #     "format": "json",
    # }
    # headers = {
    #     "Accept": "application/json",
    # }
    # data = get_json_with_retries(DDG_API_URL, params, headers)
    # docs:List[ddg] = []

    # if not data:
    #     print("INFO: no data retrieved")
    # docs.append(
    #     ddg(
    #         title=strip_whitespaces(data.get("Heading", "")),
    #         text=strip_whitespaces(data.get("AbstractText", "")),
    #         url=strip_whitespaces(data.get("AbstractURL", ""))
    #     )
    # )
    # return docs
    ############### simple one ####################

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

    data = get_json_with_retries(DDG_API_URL, params, headers)

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

def retrieve_ddg_tool(query: str) -> str:
    docs = retrieve_ddg(query)
    return json.dumps({
        "results": [
            {
                "title": d.title,
                "url": d.url,
                "content": d.text,
            }
            for d in docs
        ]
    })

def calculator_tool(expression: str) -> str:
    """
    Dummy calculator tool.
    VERY IMPORTANT: never use eval in production.
    This is purely for interview/demo purposes.
    """
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return json.dumps(
            {
                "expression": expression,
                "result": result,
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "expression": expression,
                "error": str(e),
            }
        )
#####################################################################################
# build context, system prompt helper functions
#####################################################################################

def build_context_from_docs(
    docs: List[ddg],
    max_char: int = 2000
) -> str:

    blocks:List[str] = []
    for i,d in enumerate(docs, start=1):
        url = d.url or  ""
        blocks.append(
            f"[Source {i}]: \nTitle:{d.title} \nText:{d.text} \nURL:{url}"
        )
    return "\n".join(blocks).strip()[:max_char]


def build_context_from_tool_json(tool_json_str: str, max_chars: int = 2000) -> str:
    data = json.loads(tool_json_str)
    blocks = []

    for i, r in enumerate(data.get("results", []), start=1):
        blocks.append(
            f"""[Source {i}]
Title: {r.get("title", "")}
URL: {r.get("url", "")}
Content: {r.get("content", "")}
"""
        )

    context = "\n".join(blocks).strip()
    return context[:max_chars]

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
        "Use tools when required."
        "Do not invent citations."
    )

#####################################################################################
# main
#####################################################################################

def main() ->  None:
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Missing API_KEY environment variable")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="mistral tools")
    parser.add_argument(
        "--mode",
        choices = ["llm","rag","tool"],
        required=True,
        help = "Execution mode",
    )

    parser.add_argument(
        "--question",
        type = str,
        help = "Question",
    )

    args = parser.parse_args()

    client = MistralClient(api_key = api_key)

    if args.mode == "llm":
        answer = client.answer_with_llm(args.question)

    elif args.mode == "rag":
        print("rag mode==========================")
        docs = retrieve_ddg(args.question)
        context = build_context_from_docs(docs)
        print(f"build_context_from_docs output -->> {context} ")
        answer = client.answer_with_context(args.question, context)

    elif args.mode == "tool":
        print("tool mode==========================")
        answer = client.answer_with_tools(args.question)

    else:
        raise RuntimeError("Invalid mode")


    print("\n ----- ANSWER ------ \n")
    print(answer)

if __name__ == "__main__":
    main()


