"""Print every message the agent exchanges with the model for one question.

Run: python debug_agent.py "How often are secrets rotated?"
"""
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from agent import Agent, ollama_chat
from rag_service import DEFAULT_RERANKER, RAGService
from vector_store import PgVectorStore


async def main(question: str) -> None:
    load_dotenv()
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")
    real_chat = ollama_chat(url, model)
    turn = 0

    async def logging_chat(messages, tools):
        nonlocal turn
        turn += 1
        print(f"\n===== turn {turn}: last message sent =====")
        print(json.dumps(messages[-1], indent=2)[:800])
        reply = await real_chat(messages, tools)
        print(f"----- turn {turn}: model replied -----")
        print(json.dumps(reply, indent=2)[:800])
        return reply

    service = RAGService(reranker_model_name=DEFAULT_RERANKER, store=PgVectorStore())
    result = await Agent(service, logging_chat).run(question)
    print("\n===== result =====")
    print(json.dumps({k: result[k] for k in ("answerable", "answer", "citations",
                                             "refusal_reason", "model_answer")
                      if k in result}, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "How often are secrets rotated?"))
