import pytest

from evaluate import (
    best_threshold,
    first_hit,
    load_cases,
    score_cases,
    summarize,
    threshold_sweep,
)
from ingest import ingest

CASES = [
    {"question": "ranked first", "sections": ["A"]},
    {"question": "ranked second", "sections": ["B"]},
    {"question": "missed", "sections": ["C"]},
    {"question": "not in the docs", "sections": []},
]

FAKE_RESULTS = {
    "ranked first": [("A", 0.80), ("X", 0.30)],
    "ranked second": [("X", 0.60), ("B", 0.50)],
    "missed": [("X", 0.45), ("Y", 0.40)],
    "not in the docs": [("X", 0.25), ("Y", 0.20)],
}


def fake_retrieve(question, top_k=3):
    return [
        {"section": section, "score": score}
        for section, score in FAKE_RESULTS[question][:top_k]
    ]


@pytest.fixture
def rows():
    return score_cases(CASES, fake_retrieve)


def test_first_hit_reports_rank_and_score():
    results = fake_retrieve("ranked second")

    assert first_hit(results, ["B"]) == {"rank": 2, "score": 0.50}
    assert first_hit(results, ["C"]) is None


def test_summary_metrics(rows):
    summary = summarize(rows)

    assert summary["answerable"] == 3
    assert summary["unanswerable"] == 1
    assert summary["hit@1"] == pytest.approx(1 / 3)
    assert summary["hit@3"] == pytest.approx(2 / 3)
    assert summary["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)


def test_threshold_sweep_trades_coverage_for_refusals(rows):
    low, mid, high = threshold_sweep(rows, [0.20, 0.40, 0.70])

    # Too low: the unanswerable question slips through.
    assert low["unanswerable_refused"] == 0.0
    # Middle: both correct answers kept, unanswerable refused.
    assert mid["answerable_kept"] == pytest.approx(2 / 3)
    assert mid["unanswerable_refused"] == 1.0
    # Too high: a correct answer (score 0.50) is thrown away.
    assert high["answerable_kept"] == pytest.approx(1 / 3)


def test_best_threshold_picks_the_balanced_one(rows):
    assert best_threshold(threshold_sweep(rows, [0.20, 0.40, 0.70])) == 0.40


def test_golden_set_points_at_sections_that_exist():
    """Catches typos and stale section names after a document is edited."""
    real_sections = {chunk.section for chunk in ingest("knowledge")}

    for case in load_cases():
        for section in case["sections"]:
            assert section in real_sections, f"{case['question']!r} -> {section!r}"


def test_multihop_set_points_at_sections_that_exist():
    real_sections = {chunk.section for chunk in ingest("knowledge")}

    for case in load_cases("evals/multihop.jsonl"):
        for section in case["sections"]:
            assert section in real_sections, f"{case['question']!r} -> {section!r}"
