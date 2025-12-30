import os
import argparse
import asyncio
from typing import List, Dict, Any

from dotenv import load_dotenv
from mistralai import Mistral  # only Mistral, no AsyncClient

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
