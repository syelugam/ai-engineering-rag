from evaluate_answers import score_answer, summarize_answers


def result(answerable, sections=(), invented=()):
    return {
        "answerable": answerable,
        "answer": "x",
        "sources": [{"section": s} for s in sections],
        "refusal_reason": None if answerable else "model_said_unanswerable",
        "invented_citations": list(invented),
    }


def test_answerable_needs_answer_and_matching_citation():
    case = {"question": "q", "sections": ["Deployment"]}

    assert score_answer(case, result(True, ["Deployment"]))["correct"]
    assert not score_answer(case, result(True, ["Security"]))["correct"]
    assert not score_answer(case, result(False))["correct"]


def test_unanswerable_is_correct_only_when_refused():
    case = {"question": "q", "sections": []}

    assert score_answer(case, result(False))["correct"]
    assert not score_answer(case, result(True, ["Security"]))["correct"]


def test_summary_rates():
    rows = [
        score_answer({"question": "a", "sections": ["D"]}, result(True, ["D"])),
        score_answer({"question": "b", "sections": ["D"]}, result(False)),
        score_answer({"question": "c", "sections": []}, result(False)),
        score_answer({"question": "d", "sections": []}, result(True, ["S"], ["x#1"])),
    ]
    summary = summarize_answers(rows)

    assert summary["answered_correctly"] == 0.5
    assert summary["answerable_refused"] == 0.5
    assert summary["unanswerable_refused"] == 0.5
    assert summary["invented_citation_rows"] == 1


def test_multi_part_case_needs_every_section_cited():
    case = {"question": "q", "sections": ["Fargate", "Lambda"], "require_all": True}

    assert score_answer(case, result(True, ["Fargate", "Lambda"]))["correct"]
    assert not score_answer(case, result(True, ["Fargate"]))["correct"]
