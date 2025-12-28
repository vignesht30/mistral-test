import os
import sys
from dotenv import load_dotenv

from retriever import retrieve_duckduckgo
from rag import build_context_from_docs
from llm import MistralClient


def main():
    if len(sys.argv) < 2:
        print('Usage:\n  python main.py "your question"\n  python main.py --tools "your question"')
        sys.exit(1)

    use_tools = False
    args = sys.argv[1:]

    if args[0] == "--tools":
        use_tools = True
        args = args[1:]

    if not args:
        print('Usage:\n  python main.py "your question"\n  python main.py --tools "your question"')
        sys.exit(1)

    question = args[0].strip()

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY missing in environment or .env")

    llm = MistralClient(api_key)

    if use_tools:
        print("→ Mode: TOOL CALLING (model decides retrieval)")
        print("→ Calling Mistral (tools enabled)...\n")
        answer = llm.answer_with_tools(question)
        print(answer)
        return

    # Direct RAG mode
    print("→ Mode: DIRECT RAG (always retrieve)")
    print("→ Retrieving external data...")
    docs = retrieve_duckduckgo(question)

    if not docs:
        print("No external data found.")
        return

    context = build_context_from_docs(docs)

    print("→ Calling Mistral API...\n")
    answer = llm.answer_with_context(question, context)
    print(answer)


if __name__ == "__main__":
    main()
