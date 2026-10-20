"""Agent loop tests. The model is a scripted fake, so these run offline and fast."""
import asyncio
from unittest.mock import MagicMock

from agent import Agent, parse_tool_call

ROLLBACK = {"text": "Rollback\nRe-route the listener to blue.", "score": 0.9,
            "source": "runbook.md", "section": "Rollback", "chunk_id": "runbook.md#4"}
LAMBDA = {"text": "Lambda\nMove the live alias back.", "score": 0.8,
          "source": "runbook.md", "section": "Lambda", "chunk_id": "runbook.md#3"}


def tool_call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def answer(text):
    return {"role": "assistant", "content": text}


class ScriptedModel:
    """Returns the given replies in order and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def __call__(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)


def make_rag(results_by_query=None):
    rag = MagicMock()
    rag.store = None
    results_by_query = results_by_query or {}
    rag.retrieve_relevant.side_effect = lambda q, source=None: results_by_query.get(q, [])
    rag.chunks = [MagicMock(source="runbook.md", section="Rollback"),
                  MagicMock(source="runbook.md", section="Lambda")]
    return rag


def run(agent, question="How do I roll back?"):
    return asyncio.run(agent.run(question))


def test_search_then_answer_with_citation():
    rag = make_rag({"fargate rollback": [ROLLBACK]})
    model = ScriptedModel(
        tool_call("search_docs", query="fargate rollback"),
        answer("Re-route the listener to blue [runbook.md#4]."),
    )

    result = run(Agent(rag, model))

    assert result["answerable"] is True
    assert result["citations"] == ["runbook.md#4"]
    assert result["sources"] == [ROLLBACK]
    assert result["answer"] == "Re-route the listener to blue ."
    assert [s["tool"] for s in result["steps"]] == ["search_docs"]
    # the tool result went back to the model, labelled with the passage id
    assert "[runbook.md#4]" in model.calls[1][-1]["content"]


def test_multi_part_question_uses_two_searches():
    rag = make_rag({"fargate": [ROLLBACK], "lambda": [LAMBDA]})
    model = ScriptedModel(
        tool_call("search_docs", query="fargate"),
        tool_call("search_docs", query="lambda"),
        answer("Fargate: re-route [runbook.md#4]. Lambda: move alias [runbook.md#3]."),
    )

    result = run(Agent(rag, model))

    assert result["citations"] == ["runbook.md#4", "runbook.md#3"]
    assert len(result["steps"]) == 2


def test_answer_before_any_search_is_pushed_back_once():
    rag = make_rag({"rollback": [ROLLBACK]})
    model = ScriptedModel(
        answer("Just redeploy."),                      # guess without evidence
        tool_call("search_docs", query="rollback"),
        answer("Re-route [runbook.md#4]."),
    )

    result = run(Agent(rag, model))

    assert result["answerable"] is True
    assert "search_docs" in model.calls[1][-1]["content"]   # the nudge


def test_invented_citation_is_refused():
    rag = make_rag({"rollback": [ROLLBACK]})
    model = ScriptedModel(
        tool_call("search_docs", query="rollback"),
        answer("Use kubectl rollout undo [made_up.md#9]."),
    )

    result = run(Agent(rag, model))

    assert result["answerable"] is False
    assert result["refusal_reason"] == "no_valid_citations"
    assert result["invented_citations"] == ["made_up.md#9"]


def test_not_found_is_a_refusal():
    model = ScriptedModel(tool_call("search_docs", query="database"), answer("NOT_FOUND"))

    result = run(Agent(make_rag(), model), "What database do we use?")

    assert result["refusal_reason"] == "model_said_unanswerable"


def test_unknown_tool_and_bad_arguments_are_reported_to_the_model():
    rag = make_rag({"rollback": [ROLLBACK]})
    model = ScriptedModel(
        tool_call("delete_everything"),
        tool_call("search_docs"),                    # missing query
        tool_call("search_docs", query="rollback"),
        answer("Re-route [runbook.md#4]."),
    )

    result = run(Agent(rag, model))

    assert result["answerable"] is True
    assert result["steps"][0]["result"].startswith("Error: Unknown tool")
    assert result["steps"][1]["result"].startswith("Error: search_docs needs")


def test_repeated_identical_call_is_not_run_twice():
    rag = make_rag({"rollback": [ROLLBACK]})
    model = ScriptedModel(
        tool_call("search_docs", query="rollback"),
        tool_call("search_docs", query="rollback"),
        answer("Re-route [runbook.md#4]."),
    )

    run(Agent(rag, model))

    assert rag.retrieve_relevant.call_count == 1


def test_step_limit_stops_a_looping_agent():
    model = ScriptedModel(*[tool_call("search_docs", query=f"q{i}") for i in range(10)])

    result = run(Agent(make_rag(), model, max_steps=3))

    assert result["refusal_reason"] == "step_limit"
    assert len(result["steps"]) == 3


def test_list_documents_in_memory_mode():
    model = ScriptedModel(tool_call("list_documents"), answer("NOT_FOUND"))

    result = run(Agent(make_rag(), model))

    assert result["steps"][0]["result"] == "runbook.md: Rollback\nrunbook.md: Lambda"


def test_arguments_sent_as_json_string_are_parsed():
    call = {"function": {"name": "search_docs", "arguments": '{"query": "x"}'}}

    assert parse_tool_call(call) == ("search_docs", {"query": "x"})
