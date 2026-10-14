import math
import os
from typing import Optional

import httpx
from pydantic import BaseModel, Field, ValidationError
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
        source: Optional[str] = None,
    ) -> dict:
        passages = self.retrieve_relevant(question, source=source)

        if not passages:
            return refusal(question, model_name, "no_relevant_passages")

        context = "\n\n".join(
            f"[{passage['chunk_id']}] {passage['text']}"
            for passage in passages
        )
        prompt = ANSWER_PROMPT.format(context=context, question=question)

        # Attempt 1: output constrained to the JSON schema.
        # Attempt 2: unconstrained, relying on the prompt and a tolerant parser,
        # because some models loop or stall under schema-constrained decoding.
        parsed, raw = None, ""
        for constrained in (True, False):
            raw = await self._chat(prompt, ollama_url, model_name, constrained)
            parsed = parse_model_answer(raw)
            if parsed is not None:
                break
        if parsed is None:
            raise ModelOutputError(
                f"The model did not return valid JSON. Last reply: {raw[:300]!r}"
            )

        if not parsed.answerable:
            return refusal(question, model_name, "model_said_unanswerable")

        # Keep only citations that point at passages we actually supplied.
        by_id = {passage["chunk_id"]: passage for passage in passages}
        cited = [cid for cid in dict.fromkeys(parsed.citations) if cid in by_id]
        invented = [cid for cid in parsed.citations if cid not in by_id]

        if not cited:
            # An answer we cannot trace back to a passage is not returned.
            return refusal(
                question, model_name, "no_valid_citations", invented_citations=invented
            )

        return {
            "question": question,
            "answerable": True,
            "answer": parsed.answer.strip(),
            "model": model_name,
            "citations": cited,
            "sources": [by_id[cid] for cid in cited],
            "refusal_reason": None,
            "invented_citations": invented,
        }

    async def _chat(
        self, prompt: str, ollama_url: str, model_name: str, constrained: bool = True
    ) -> str:
        timeout = float(os.environ.get("OLLAMA_TIMEOUT", "120"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # No hidden reasoning: it is slow, can run on for minutes,
                # and its tokens count against num_predict below.
                "think": False,
                # Cap the reply length so a runaway generation fails fast.
                "options": {"temperature": 0, "num_predict": 400},
            }
            if constrained:
                # Ollama constrains the output to this JSON schema.
                body["format"] = ModelAnswer.model_json_schema()
            response = await client.post(
                f"{ollama_url.rstrip('/')}/api/chat", json=body
            )
            response.raise_for_status()
        return response.json()["message"]["content"]


class ModelOutputError(Exception):
    """The language model's reply could not be used."""


class ModelAnswer(BaseModel):
    # Field order matters: the model writes fields in this order, so it drafts
    # the answer and finds its citations before it judges answerability.
    answer: str = Field(description="brief answer, or empty if not answerable")
    citations: list[str] = Field(
        default_factory=list,
        description="ids of the passages the answer is based on",
    )
    answerable: bool = Field(
        description="true if the answer above is supported by the cited passages"
    )


ANSWER_PROMPT = """You answer questions about our AWS architecture using only the context below.
Each passage starts with its id in square brackets.

Rules:
- First look for the answer in the context. If it is there, even stated in different words, give a brief answer, list in "citations" the ids of every passage you used (e.g. "platform_runbook.md#2"), and set "answerable" to true.
- Only if the context really does not contain the answer: "answer" is "", "citations" is [] and "answerable" is false.
- Never use knowledge that is not in the context. Never invent ids.

Respond with JSON only: {{"answer": "...", "citations": ["..."], "answerable": true/false}}

Context:
{context}

Question:
{question}
"""


def parse_model_answer(raw: str) -> Optional[ModelAnswer]:
    """Parse the model's JSON reply; None if it is unusable.

    Tolerates code fences and text around the JSON object.
    """
    text = raw.strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return ModelAnswer.model_validate_json(candidate)
        except ValidationError:
            continue
    return None


def refusal(question: str, model_name: str, reason: str, **extra) -> dict:
    return {
        "question": question,
        "answerable": False,
        "answer": "I don't know based on the available documentation.",
        "model": model_name,
        "citations": [],
        "sources": [],
        "refusal_reason": reason,
        "invented_citations": extra.get("invented_citations", []),
    }
