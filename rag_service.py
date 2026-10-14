import math
from typing import Optional

import httpx
from sentence_transformers import SentenceTransformer, util

from ingest import ingest

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Score cutoffs tuned on evals/golden.jsonl. The two scores are on different
# scales, so each retrieval mode has its own.
COSINE_MIN_SCORE = 0.30     # Day 5
RERANK_MIN_SCORE = 0.05     # Day 6: unanswerable max 0.02, answerable mostly >= 0.10


class RAGService:
    def __init__(
        self,
        document_path: str = "knowledge",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: Optional[str] = None,
        rerank_candidates: int = 10,
        min_score: Optional[float] = None,
        store=None,
    ):
        # With a store (e.g. PgVectorStore), chunks and embeddings live in the
        # database and are loaded by index_documents.py. Without one, the
        # documents are chunked and embedded in memory at startup.
        self.store = store
        if store is None:
            # document_path can be a single file or a folder of .md/.txt files.
            self.chunks = ingest(document_path)

        # Reranking is off unless a cross-encoder model name is given.
        self.reranker_model_name = reranker_model_name
        self.rerank_candidates = rerank_candidates
        self._reranker = None
        if min_score is None:
            min_score = COSINE_MIN_SCORE if reranker_model_name is None else RERANK_MIN_SCORE
        self.min_score = min_score

        self.embedding_model = SentenceTransformer(embedding_model_name)
        if store is None:
            self.document_embeddings = self.embedding_model.encode(
                [chunk.text for chunk in self.chunks],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

    def retrieve(
        self, question: str, top_k: int = 3, source: Optional[str] = None
    ) -> list[dict]:
        """Best passages for the question; reranked when a reranker is configured.

        source limits the search to one file (database mode only).
        """
        if self.reranker_model_name is None:
            return self.vector_search(question, top_k, source=source)

        # Stage 1: cheap vector search casts a wide net.
        candidates = self.vector_search(
            question, top_k=max(top_k, self.rerank_candidates), source=source
        )
        # Stage 2: the cross-encoder reads question and passage together.
        return self.rerank(question, candidates)[:top_k]

    def vector_search(
        self, question: str, top_k: int = 3, source: Optional[str] = None
    ) -> list[dict]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        if self.store is not None:
            query_embedding = self.embedding_model.encode(
                question, normalize_embeddings=True
            )
            return self.store.search(query_embedding, top_k=top_k, source=source)

        if source is not None:
            raise ValueError("Filtering by source needs the database store.")

        question_embedding = self.embedding_model.encode(
            question,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        scores = util.cos_sim(
            question_embedding, self.document_embeddings
        )[0]

        results = scores.topk(min(top_k, len(self.chunks)))

        return [
            {
                "text": self.chunks[index].text,
                "score": float(score),
                "source": self.chunks[index].source,
                "section": self.chunks[index].section,
                "chunk_id": self.chunks[index].chunk_id,
            }
            for score, index in zip(
                results.values.tolist(),
                results.indices.tolist(),
            )
        ]

    @property
    def reranker(self):
        # Loaded on first use, so code paths without reranking never download it.
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def rerank(self, question: str, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        # Ask for raw logits and convert them ourselves, so the score is
        # always a 0-1 relevance value whatever the model's own default is.
        logits = self.reranker.predict(
            [(question, candidate["text"]) for candidate in candidates],
            activation_fct=lambda x: x,
            show_progress_bar=False,
        )

        reranked = [
            {
                **candidate,
                "vector_score": candidate["score"],
                "score": 1 / (1 + math.exp(-float(logit))),
            }
            for candidate, logit in zip(candidates, logits)
        ]
        return sorted(reranked, key=lambda item: item["score"], reverse=True)

    def retrieve_relevant(
        self,
        question: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
        source: Optional[str] = None,
    ) -> list[dict]:
        threshold = self.min_score if min_score is None else min_score
        results = self.retrieve(question, top_k=top_k, source=source)
        return [
            item for item in results
            if item["score"] >= threshold
        ]

    async def answer(
        self,
        question: str,
        ollama_url: str,
        model_name: str,
    ) -> dict:
        passages = self.retrieve_relevant(question)

        if not passages:
            return {
                "question": question,
                "answer": "I don't know based on the available documentation.",
                "model": model_name,
                "sources": [],
            }

        context = "\n\n".join(
            f"[{passage['chunk_id']}] {passage['text']}"
            for passage in passages
        )

        prompt = f"""
You are an assistant answering questions about our AWS architecture.
Use only the supplied context to answer.
If the context does not contain the answer, say you don't know.
Do not invent architecture details.

Each context passage starts with an id in square brackets.

Context:
{context}

Question:
{question}

Give a brief answer, then cite the ids of the passages you used, like [platform_runbook.md#2].
"""

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{ollama_url.rstrip('/')}/api/chat",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()

        data = response.json()

        return {
            "question": question,
            "answer": data["message"]["content"],
            "model": model_name,
            "sources": passages,
        }
