"""End-to-end answer evaluation: retrieval + the language model, on the golden set.

Run:  python evaluate_answers.py                       (fixed pipeline, golden set)
      python evaluate_answers.py --mode agent          (tool-calling agent)
      python evaluate_answers.py --mode decompose      (split question, search each part)
      python evaluate_answers.py --cases evals/multihop.jsonl --mode agent
(Needs Ollama running and the Postgres index built with index_documents.py.)

For answerable questions it checks the model answered and cited a passage from
an expected section. For unanswerable ones it checks the system refused.
"""
import argparse
import asyncio
import os
import sys
import time

from evaluate import load_cases


def score_answer(case: dict, result: dict) -> dict:
    expected = set(case["sections"])
    cited_sections = {source["section"] for source in result["sources"]}
    answerable = bool(expected)

    if answerable and case.get("require_all"):
        # multi-part question: every part's section must be cited
        correct = result["answerable"] and expected <= cited_sections
    elif answerable:
        correct = result["answerable"] and bool(cited_sections & expected)
    else:
        correct = not result["answerable"]

    return {
        "question": case["question"],
        "answerable": answerable,
        "correct": correct,
        "answered": result["answerable"],
        "cited_sections": sorted(cited_sections),
        "refusal_reason": result["refusal_reason"],
        "invented": result.get("invented_citations", []),
        "answer": result["answer"],
        "model_answer": result.get("model_answer", ""),
    }


def summarize_answers(rows: list) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]

    def rate(group, key):
        return sum(1 for row in group if row[key]) / len(group) if group else 0.0

    return {
        "answered_correctly": rate(answerable, "correct"),
        "answerable_refused": sum(1 for r in answerable if not r["answered"]) / len(answerable)
        if answerable else 0.0,
        "unanswerable_refused": rate(unanswerable, "correct"),
        "invented_citation_rows": sum(1 for row in rows if row["invented"]),
    }


async def run(cases_path: str, mode: str) -> int:
    from dotenv import load_dotenv

    from rag_service import DEFAULT_RERANKER, RAGService
    from vector_store import PgVectorStore

    load_dotenv()
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model_name = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

    service = RAGService(reranker_model_name=DEFAULT_RERANKER, store=PgVectorStore())
    rows, timings = [], []

    errors = []
    if mode == "agent":
        from agent import Agent, ollama_chat

        agent = Agent(service, ollama_chat(ollama_url, model_name))

        async def ask(question):
            return await agent.run(question)
    elif mode == "decompose":
        from planner import DecomposingRAG

        decomposer = DecomposingRAG(service, ollama_url, model_name)

        async def ask(question):
            return await decomposer.answer(question)
    else:
        async def ask(question):
            return await service.answer(question, ollama_url, model_name)

    steps = []
    for case in load_cases(cases_path):
        started = time.perf_counter()
        try:
            result = await ask(case["question"])
        except Exception as exc:   # one slow or broken call shouldn't end the run
            errors.append((case["question"], f"{type(exc).__name__}: {exc}"))
            print("E", end="", flush=True)
            continue
        timings.append(time.perf_counter() - started)
        rows.append(score_answer(case, result))
        steps.append(len(result.get("steps", [])))
        print("." if rows[-1]["correct"] else "F", end="", flush=True)

    if errors:
        print(f"\n\n{len(errors)} question(s) errored and are excluded from the scores:")
        for question, message in errors:
            print(f"  {question}\n    {message}")
    if not rows:
        print("\nNo question completed; check that Ollama is running and responsive.")
        return 1

    summary = summarize_answers(rows)
    print(f"\n\nMode: {mode}   Cases: {cases_path}   Model: {model_name}")
    print(f"Answerable answered with a correct citation: {summary['answered_correctly']:.0%}")
    print(f"Answerable wrongly refused:                  {summary['answerable_refused']:.0%}")
    print(f"Unanswerable correctly refused:              {summary['unanswerable_refused']:.0%}")
    print(f"Answers containing invented citations:       {summary['invented_citation_rows']}")
    print(f"Average time per question:                   {sum(timings) / len(timings):.1f}s")
    if mode == "agent":
        print(f"Average tool calls per question:             {sum(steps) / len(steps):.1f}")

    failures = [row for row in rows if not row["correct"]]
    if failures:
        print("\nFailures:")
        for row in failures:
            print(f"  {row['question']}")
            print(f"    answered={row['answered']} reason={row['refusal_reason']} "
                  f"cited={row['cited_sections']}")
            print(f"    answer: {row['answer'][:120]}")
            if row.get("model_answer"):
                print(f"    model wrote: {row['model_answer'][:200]!r}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end answer evaluation.")
    parser.add_argument("--cases", default="evals/golden.jsonl")
    parser.add_argument("--mode", choices=["pipeline", "agent", "decompose"], default="pipeline")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.cases, args.mode)))
