"""Query decomposition: a fixed workflow instead of a free-running agent.

  1. The model splits the question into 1-3 search queries (one JSON call).
  2. Code, not the model, runs a search for the original question and each query.
  3. The merged passages go through the same grounded-answer step as /ask.

The model makes one small, well-defined decision; the code does the rest.
Small models are much more reliable at this than at choosing tools in a loop.
"""
import os
from typing import Awaitable, Callable, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError

MAX_QUERIES = 3
MAX_PASSAGES = 6

PLAN_PROMPT = """Split the user's question into the separate things it asks about,
as 1 to {max_queries} short search queries. If it asks about one thing only,
return a single query. Do not answer the question.

Question: {question}

Respond with JSON only: {{"queries": ["...", "..."]}}
"""


class Plan(BaseModel):
    queries: list[str] = Field(description="1-3 short search queries")


PlanChat = Callable[[str], Awaitable[str]]


def parse_plan(raw: str) -> Optional[list]:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        plan = Plan.model_validate_json(text[start : end + 1])
    except ValidationError:
        return None
    queries = [q.strip() for q in plan.queries if isinstance(q, str) and q.strip()]
    return queries[:MAX_QUERIES] or None


def merge_passages(result_lists: list, limit: int = MAX_PASSAGES) -> list:
    """Combine search results, one copy per chunk at its best score, best first."""
    best = {}
    for results in result_lists:
        for passage in results:
            current = best.get(passage["chunk_id"])
            if current is None or passage["score"] > current["score"]:
                best[passage["chunk_id"]] = passage
    return sorted(best.values(), key=lambda p: p["score"], reverse=True)[:limit]


class DecomposingRAG:
    def __init__(self, rag, ollama_url: str, model_name: str, chat: Optional[PlanChat] = None):
        self.rag = rag
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.chat = chat or self._ollama_plan_chat

    async def plan(self, question: str) -> list:
        raw = await self.chat(PLAN_PROMPT.format(max_queries=MAX_QUERIES, question=question))
        return parse_plan(raw) or []

    async def answer(self, question: str) -> dict:
        try:
            sub_queries = await self.plan(question)
        except httpx.HTTPError:
            sub_queries = []   # planning is an optimisation; fall back to one search

        # Always search the original question too, so decomposition can only add.
        queries = list(dict.fromkeys([question, *sub_queries]))
        passages = merge_passages([self.rag.retrieve_relevant(q) for q in queries])

        result = await self.rag.answer_from_passages(
            question, passages, self.ollama_url, self.model_name
        )
        result["sub_queries"] = queries
        return result

    async def _ollama_plan_chat(self, prompt: str) -> str:
        timeout = float(os.environ.get("OLLAMA_TIMEOUT", "120"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.ollama_url.rstrip('/')}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "format": Plan.model_json_schema(),
                    "options": {"temperature": 0, "num_predict": 200},
                },
            )
            response.raise_for_status()
        return response.json()["message"]["content"]
