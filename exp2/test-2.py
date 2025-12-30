import os
import re
import sys
import time
import random
from typing import List, Dict, Any, Optional
import argparse

from dotenv import load_dotenv
from mistralai import Mistral

class MistralClient:
    def __init__(self, api_key:str, model:str= "mistral-large-latest") -> None:
        self.client = Mistral(api_key=api_key)
        self.model = model

    def answer_with_llm(self, question:str) -> str:
        messages: List [Dict[str, Any]] = [
            {"role":"system", "content":"you are helpful assistant"},
            {"role":"user", "content": question },
        ]

        res = self.client.chat.complete(
            model = self.model,
            messages=messages,
            temperature= 0.7,
            response_format={"type": "text"},
        )

        return res.choices[0].message.content

def main() -> None:
    print("INFO: Entering Main()")

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("INFO: No API key provided")

    parser = argparse.ArgumentParser(description= "mistral demo")

    parser.add_argument(
        "--mode",
        choices = ["llm", "rag"],
        required = True,
        help = "mode"
    )

    parser.add_argument(
        "--question",
        type = str,
        required = True,
        help="mode"
    )

    arg = parser.parse_args()

    client = MistralClient(api_key=api_key)

    if arg.mode == "llm":
        answer = client.answer_with_llm(question=arg.question)
    else:
        raise RuntimeError("Invalid mode")

    print(f" ANSWER  ------>>>>>> \n {answer}")


if __name__ == "__main__":
    main()

