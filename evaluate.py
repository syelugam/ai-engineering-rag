"""Retrieval evaluation.

Run:     python evaluate.py
Rerank:  python evaluate.py --rerank       (compare against the plain run)
Postgres: python evaluate.py --rerank --pg (search the pgvector index instead of memory)
Gate:    python evaluate.py --min-hit3 0.85   (exits 1 if hit@3 is below the floor)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

TOP_K = 3   # same number of passages the /ask endpoint sends to the model

# Cosine scores bunch between 0.1 and 0.7; reranker scores spread from ~0 to ~1.
COSINE_THRESHOLDS = [t / 100 for t in range(20, 65, 5)]
RERANK_THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def load_cases(path: str = "evals/golden.jsonl") -> list:
    cases = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        case = json.loads(line)
        if "question" not in case or "sections" not in case:
            raise ValueError(f"{path} line {number}: need 'question' and 'sections'")
        cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def first_hit(results: list, expected_sections: list) -> Optional[dict]:
    """Rank (1-based) and score of the first retrieved chunk from an expected section."""
    for rank, result in enumerate(results, start=1):
        if result["section"] in expected_sections:
            return {"rank": rank, "score": result["score"]}
    return None


def score_cases(cases: list, retrieve: Callable, top_k: int = TOP_K) -> list:
    rows = []
    for case in cases:
        results = retrieve(case["question"], top_k=top_k)
        hit = first_hit(results, case["sections"])
        rows.append(
            {
                "question": case["question"],
                "answerable": bool(case["sections"]),
                "rank": hit["rank"] if hit else None,
                "hit_score": hit["score"] if hit else None,
                "top_score": results[0]["score"] if results else 0.0,
                "top_section": results[0]["section"] if results else None,
            }
        )
    return rows


def summarize(rows: list) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    if not answerable:
        raise ValueError("The evaluation set has no answerable questions.")

    def hit_rate(k: int) -> float:
        hits = [row for row in answerable if row["rank"] and row["rank"] <= k]
        return len(hits) / len(answerable)

    return {
        "answerable": len(answerable),
        "unanswerable": len(rows) - len(answerable),
        "hit@1": hit_rate(1),
        "hit@3": hit_rate(3),
        # Mean reciprocal rank: 1.0 if the right chunk is always first,
        # 0.5 if always second, 0 for a miss.
        "mrr": sum(1 / row["rank"] for row in answerable if row["rank"])
        / len(answerable),
    }


def threshold_sweep(rows: list, thresholds: list) -> list:
    """For each threshold: how many good answers survive, how many bad questions are refused."""
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]

    sweep = []
    for threshold in thresholds:
        kept = [
            row for row in answerable
            if row["hit_score"] is not None and row["hit_score"] >= threshold
        ]
        refused = [row for row in unanswerable if row["top_score"] < threshold]
        kept_rate = len(kept) / len(answerable) if answerable else 0.0
        refused_rate = len(refused) / len(unanswerable) if unanswerable else 0.0
        sweep.append(
            {
                "threshold": threshold,
                "answerable_kept": kept_rate,
                "unanswerable_refused": refused_rate,
                "balanced": (kept_rate + refused_rate) / 2,
            }
        )
    return sweep


def best_threshold(sweep: list) -> float:
    return max(sweep, key=lambda item: item["balanced"])["threshold"]


def print_report(rows: list, summary: dict, sweep: list) -> None:
    print(f"\n{summary['answerable']} answerable, {summary['unanswerable']} unanswerable questions\n")
    print(f"hit@1  {summary['hit@1']:.0%}   right section ranked first")
    print(f"hit@3  {summary['hit@3']:.0%}   right section in the top 3")
    print(f"MRR    {summary['mrr']:.2f}")

    problems = [row for row in rows if row["answerable"] and row["rank"] != 1]
    if problems:
        print("\nNot ranked first:")
        for row in problems:
            rank = f"rank {row['rank']}" if row["rank"] else "MISS  "
            print(f"  {rank}  {row['question']}")
            print(f"          got: {row['top_section']} ({row['top_score']:.2f})")

    print("\nUnanswerable questions, top score (lower is better):")
    for row in rows:
        if not row["answerable"]:
            print(f"  {row['top_score']:.2f}  {row['question']}")

    print("\nthreshold  answerable kept  unanswerable refused")
    for item in sweep:
        print(
            f"  {item['threshold']:.2f}        {item['answerable_kept']:>5.0%}"
            f"             {item['unanswerable_refused']:>5.0%}"
        )
    print(f"\nBest balance at min_score = {best_threshold(sweep):.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--cases", default="evals/golden.jsonl")
    parser.add_argument("--min-hit3", type=float, default=None)
    parser.add_argument(
        "--rerank", action="store_true", help="rerank with a cross-encoder"
    )
    parser.add_argument(
        "--pg", action="store_true", help="search Postgres/pgvector instead of memory"
    )
    args = parser.parse_args()

    # Imported here so the tests don't load any models.
    from rag_service import DEFAULT_RERANKER, RAGService

    store = None
    if args.pg:
        from vector_store import PgVectorStore

        store = PgVectorStore()

    service = RAGService(
        reranker_model_name=DEFAULT_RERANKER if args.rerank else None,
        store=store,
    )
    thresholds = RERANK_THRESHOLDS if args.rerank else COSINE_THRESHOLDS
    print(
        "Mode: "
        + ("Postgres/pgvector" if args.pg else "in-memory")
        + (f" + rerank ({DEFAULT_RERANKER})" if args.rerank else " vector search")
    )

    rows = score_cases(load_cases(args.cases), service.retrieve)
    summary = summarize(rows)
    sweep = threshold_sweep(rows, thresholds)
    print_report(rows, summary, sweep)

    if args.min_hit3 is not None and summary["hit@3"] < args.min_hit3:
        print(f"\nFAIL: hit@3 {summary['hit@3']:.0%} is below {args.min_hit3:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
