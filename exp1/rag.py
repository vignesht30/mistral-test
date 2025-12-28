import json
from typing import List, Dict, Any
from retriever import DuckDoc


def build_context_from_docs(docs: List[DuckDoc], max_chars: int = 2000) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(
            f"""[Source {i}]
Title: {doc.title}
URL: {doc.url}
Content: {doc.text}
"""
        )

    context = "\n".join(blocks).strip()
    return context[:max_chars]


def build_context_from_tool_json(tool_json_str: str, max_chars: int = 2000) -> str:
    data = json.loads(tool_json_str)
    blocks = []

    for i, r in enumerate(data.get("results", []), start=1):
        blocks.append(
            f"""[Source {i}]
Title: {r.get("title","")}
URL: {r.get("url","")}
Content: {r.get("content","")}
"""
        )

    context = "\n".join(blocks).strip()
    return context[:max_chars]



def system_prompt() -> str:
    return (
        "You are a helpful assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the context is insufficient, say so.\n"
        "Cite sources like [Source 1]."
    )
