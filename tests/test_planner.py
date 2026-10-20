import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx

from planner import DecomposingRAG, merge_passages, parse_plan


def passage(cid, score):
    return {"chunk_id": cid, "score": score, "text": cid, "section": cid}


def make_rag(results):
    rag = MagicMock()
    rag.retrieve_relevant.side_effect = lambda q: results.get(q, [])
    rag.answer_from_passages = AsyncMock(return_value={"answerable": True})
    return rag


def test_parse_plan_limits_and_cleans_queries():
    assert parse_plan('{"queries": ["a", " ", "b", "c", "d"]}') == ["a", "b", "c"]
    assert parse_plan("no json here") is None
    assert parse_plan('{"queries": []}') is None


def test_merge_keeps_best_score_per_chunk_and_sorts():
    merged = merge_passages([
        [passage("x#1", 0.4), passage("x#2", 0.9)],
        [passage("x#1", 0.7)],
    ])

    assert [(p["chunk_id"], p["score"]) for p in merged] == [("x#2", 0.9), ("x#1", 0.7)]


def test_searches_original_question_and_each_sub_query():
    rag = make_rag({
        "Lambda vs Fargate canary?": [passage("r#2", 0.3)],
        "Lambda canary duration": [passage("r#3", 0.8)],
        "Fargate canary duration": [passage("r#2", 0.9)],
    })
    plan = AsyncMock(return_value='{"queries": ["Lambda canary duration", "Fargate canary duration"]}')
    decomposer = DecomposingRAG(rag, "http://x", "m", chat=plan)

    result = asyncio.run(decomposer.answer("Lambda vs Fargate canary?"))

    passages = rag.answer_from_passages.call_args.args[1]
    assert [p["chunk_id"] for p in passages] == ["r#2", "r#3"]
    assert result["sub_queries"] == [
        "Lambda vs Fargate canary?", "Lambda canary duration", "Fargate canary duration",
    ]


def test_falls_back_to_a_single_search_when_planning_fails():
    rag = make_rag({"q": [passage("r#1", 0.5)]})
    decomposer = DecomposingRAG(rag, "http://x", "m", chat=AsyncMock(return_value="garbage"))

    result = asyncio.run(decomposer.answer("q"))

    assert result["sub_queries"] == ["q"]

    broken = DecomposingRAG(
        rag, "http://x", "m", chat=AsyncMock(side_effect=httpx.ConnectError("down"))
    )
    assert asyncio.run(broken.answer("q"))["sub_queries"] == ["q"]
