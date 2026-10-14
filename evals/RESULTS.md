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
