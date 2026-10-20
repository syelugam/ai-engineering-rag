"""A hand-written tool-calling agent over the RAG service.

The loop, in plain words:
  1. Send the conversation and the tool menu to the model.
  2. If the model asks for tools, run them and append the results.
  3. Repeat until the model writes a final answer, or the step limit is hit.
  4. Check that every citation in the answer points at a passage a tool
     actually returned. If none do, refuse.

No framework is used, so every decision the agent makes is visible here.
"""
import json
import os
import re
from typing import Awaitable, Callable, Optional

import httpx

MAX_STEPS = 6
CITATION = re.compile(r"\[([^\[\]\s]+#\d+)\]")
NOT_FOUND = "NOT_FOUND"

SYSTEM_PROMPT = f"""You answer questions about our platform documentation.

How to work:
- Use the search_docs tool to find evidence. You may call it several times with
  different queries, for example once for each part of a multi-part question.
- Use list_documents if you need to know which documents and sections exist.
- Answer only from tool results, never from your own knowledge.
- Cite every fact with the passage id in square brackets, e.g. [platform_runbook.md#4].
- If the documents do not contain the answer, reply with exactly: {NOT_FOUND}
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the platform documentation. Returns the most relevant "
                "passages, each starting with its id in square brackets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, in plain words.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional file name to search within, e.g. platform_runbook.md",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List every document and its section headings.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

ChatFn = Callable[[list, list], Awaitable[dict]]


class ToolError(Exception):
    """A tool call the agent should be told about, not crash on."""


class Agent:
    def __init__(self, rag, chat: ChatFn, max_steps: int = MAX_STEPS):
        self.rag = rag
        self.chat = chat
        self.max_steps = max_steps

    async def run(self, question: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        seen_passages = {}   # chunk_id -> passage, from every search this run
        trace = []           # what the agent did, for debugging and the API
        done_calls = set()   # (tool, args) already executed
        nudged = False

        for _step in range(self.max_steps):
            message = await self.chat(messages, TOOLS)
            messages.append(message)
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                answer = (message.get("content") or "").strip()
                # Guardrail: an answer before any search is a guess. Push back once.
                if not trace and not nudged and NOT_FOUND not in answer:
                    nudged = True
                    messages.append({
                        "role": "user",
                        "content": "Search the documentation with search_docs before answering.",
                    })
                    continue
                return self._finish(question, answer, seen_passages, trace)

            for call in tool_calls:
                name, args = parse_tool_call(call)
                key = (name, json.dumps(args, sort_keys=True))
                if key in done_calls:
                    output = "You already made this exact call; use its earlier result."
                else:
                    done_calls.add(key)
                    try:
                        output = self._run_tool(name, args, seen_passages)
                    except ToolError as error:
                        output = f"Error: {error}"
                trace.append({"tool": name, "args": args, "result": output[:300]})
                messages.append({"role": "tool", "tool_name": name, "content": output})

        return refusal(question, "step_limit", trace)

    def _run_tool(self, name: str, args: dict, seen_passages: dict) -> str:
        if name == "search_docs":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ToolError("search_docs needs a non-empty 'query' string.")
            source = args.get("source") or None
            if source is not None and self.rag.store is None:
                source = None   # in-memory mode cannot filter by file
            passages = self.rag.retrieve_relevant(query, source=source)
            if not passages:
                return "No relevant passages found. Try different words."
            for passage in passages:
                seen_passages[passage["chunk_id"]] = passage
            return "\n\n".join(f"[{p['chunk_id']}] {p['text']}" for p in passages)

        if name == "list_documents":
            if self.rag.store is not None:
                sections = self.rag.store.list_sections()
            else:
                sections = list(dict.fromkeys((c.source, c.section) for c in self.rag.chunks))
            return "\n".join(f"{source}: {section}" for source, section in sections)

        raise ToolError(f"Unknown tool '{name}'. Available: search_docs, list_documents.")

    def _finish(self, question, answer, seen_passages, trace) -> dict:
        cited_ids = list(dict.fromkeys(CITATION.findall(answer)))
        valid = [cid for cid in cited_ids if cid in seen_passages]
        invented = [cid for cid in cited_ids if cid not in seen_passages]

        if NOT_FOUND in answer and not valid:
            return refusal(question, "model_said_unanswerable", trace)
        if not valid:
            return refusal(question, "no_valid_citations", trace, invented)

        return {
            "question": question,
            "answerable": True,
            "answer": CITATION.sub("", answer).strip(),
            "citations": valid,
            "sources": [seen_passages[cid] for cid in valid],
            "refusal_reason": None,
            "invented_citations": invented,
            "steps": trace,
        }


def parse_tool_call(call: dict) -> tuple:
    function = call.get("function", {})
    args = function.get("arguments") or {}
    if isinstance(args, str):   # some models send arguments as a JSON string
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return function.get("name", ""), args


def refusal(question, reason, trace, invented=()) -> dict:
    return {
        "question": question,
        "answerable": False,
        "answer": "I don't know based on the available documentation.",
        "citations": [],
        "sources": [],
        "refusal_reason": reason,
        "invented_citations": list(invented),
        "steps": trace,
    }


def ollama_chat(ollama_url: str, model_name: str) -> ChatFn:
    """A chat function that talks to Ollama's /api/chat with tools."""
    timeout = float(os.environ.get("OLLAMA_TIMEOUT", "120"))

    async def chat(messages: list, tools: list) -> dict:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{ollama_url.rstrip('/')}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 600},
                },
            )
            response.raise_for_status()
        return response.json()["message"]

    return chat
