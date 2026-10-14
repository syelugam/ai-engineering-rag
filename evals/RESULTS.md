# Retrieval evaluation results

Question set: `evals/golden.jsonl` (23 answerable, 6 unanswerable). Top 3 passages go to the model.
Run with `python evaluate.py` (vector only) or `python evaluate.py --rerank`.

| Run | Retrieval | hit@1 | hit@3 | MRR | Cutoff | Answerable kept | Unanswerable refused |
|---|---|---|---|---|---|---|---|
| Day 5 | Vector search (all-MiniLM-L6-v2) | 78% | 96% | 0.87 | 0.30 cosine | 87% | 50% |
| Day 6 | Vector top 10 + cross-encoder rerank (ms-marco-MiniLM-L-6-v2) | 91% | 100% | 0.96 | 0.05 rerank | 96% | 100% |
| Day 7 | Same as Day 6, index stored in Postgres/pgvector (HNSW) | 91% | 100% | 0.96 | 0.05 rerank | 96% | 100% |

## Notes

- Day 7: moving the index to Postgres changed no metric, confirming the migration preserved retrieval behaviour.

- Day 5: no single cosine cutoff separated the two groups. Some correct passages scored 0.26 while an unanswerable question scored 0.41.
- Day 5 end-to-end: the two unanswerable questions that passed the cutoff (AWS bill, Kubernetes version) were still refused by the model, which made it a working second check.
- Day 6: reranker scores for unanswerable questions were at most 0.02, while all but one answerable question had its correct passage at 0.10 or above. A cutoff of 0.05 sits between the two.
- Remaining misses (rank 2): "How do I undo a bad Lambda release?" (wording differs from "roll back") and "Does a message in the dead-letter queue page the on-call engineer?".
- 29 questions is a small set: one question moves a rate by about 4 points.

# End-to-end answer evaluation (Day 8)

Run with `python evaluate_answers.py`: retrieval + rerank + `gemma4:e2b` via Ollama, structured JSON output with citation checks.
"Correct" means the system answered and cited a passage from an expected section; unanswerable questions must be refused.

| Change | Answered correctly | Wrongly refused | Unanswerable refused | Errors | Time/question |
|---|---|---|---|---|---|
| JSON schema, model thinking on | — | — | — | run timed out (>120s on one call) | — |
| Thinking off, 400-token cap | 10/23 completed correctly | — | 100% | 12 invalid JSON | 1.4s |
| + free-form retry, tolerant parser | 74% | 26% | 100% | 0 | 0.3s |
| + schema field order: answer, citations, answerable | 96% | 4% | 100% | 0 | 0.3s |

## Notes

- Thinking: Gemma reasoned for hundreds of tokens before each answer; on some questions it ran past the timeout. Disabling it removed the timeouts.
- Field order: with `answerable` first, the model had to decide before writing any answer and skewed to refusing. Moving it last (draft answer, cite, then judge) cut wrong refusals from 26% to 4% with no change in model, data or retrieval.
- Remaining failure: "How do I undo a bad Lambda release?" — the runbook says "roll back", not "undo".
- No invented citations in any run; citation validation would have refused such answers anyway.
