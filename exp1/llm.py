import json
from typing import Any, Dict, List, Optional

from mistralai import Mistral

from retriever import duckduckgo_tool
from rag import system_prompt, build_context_from_tool_json


class MistralClient:
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        self.client = Mistral(api_key=api_key)
        self.model = model

    def answer_with_context(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": question},
        ]

        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content

    def answer_with_tools(self, question: str) -> str:
        """
        Tool calling flow:
        1) LLM decides whether retrieval is needed
        2) External API is called for factual context
        3) Retrieved data is injected into the prompt
        4) Final answer is grounded in retrieved context
        """

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for factual information to answer the user's question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for retrieving relevant information"
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": question},
        ]

        # Step 1: ask model, allow tool calls
        res1 = self.client.chat.complete(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            response_format={"type": "text"},
        )

        msg = res1.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        # If model didn't call a tool, just return its answer
        if not tool_calls:
            return msg.content

        # Step 2: execute tool calls (usually just 1 in interviews)
        # Add the assistant message that contains the tool call(s)
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Execute each tool call and append tool responses
        for tc in tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments or "{}")

            if fn_name != "search":
                raise RuntimeError(f"Unexpected tool requested: {fn_name}")

            # Light query rewrite to improve recall
            query = fn_args.get("query") or f"{question} history background purpose"

            tool_output = duckduckgo_tool(query=query)

            # Provide tool output (string) back to the model
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                }
            )

        # Step 3: final answer, grounded using tool output as context
        # (We reinforce grounding by inserting a context message derived from tool output)
        last_tool_content = messages[-1]["content"]
        ctx = build_context_from_tool_json(last_tool_content)

        messages.insert(
            1,  # right after system prompt
            {"role": "system", "content": f"Context:\n{ctx}"},
        )

        res2 = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "text"},
        )
        return res2.choices[0].message.content
