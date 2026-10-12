
import os

import logging

import httpx
from functools import lru_cache
from pydantic import BaseModel, Field
from rag_service import DEFAULT_RERANKER, RAGService
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

settings = Settings()
app = FastAPI(title="AI Engineering Assistant API")


OLLAMA_URL = settings.ollama_url.rstrip("/")
MODEL_NAME = settings.ollama_model

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService(reranker_model_name=DEFAULT_RERANKER)


@app.post("/ask")
async def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        settings = Settings()
        rag = get_rag_service()

        return await rag.answer(
            question=question,
            ollama_url=settings.ollama_url,
            model_name=settings.ollama_model,
        )
    except httpx.HTTPError:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=502,
            detail="The local language model service failed.",
        )

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    model: str


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": request.question,
            }
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            logger.info("Sending chat request to model=%s", settings.ollama_model)
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

        answer = result["message"]["content"]

        return ChatResponse(
            answer=answer,
            model=MODEL_NAME,
        )

    except httpx.TimeoutException:
        logger.warning("Ollama request timed out")
        raise HTTPException(
            status_code=504,
            detail="The AI model took too long to respond.",
        )
    except httpx.HTTPError:
        logger.exception("Ollama HTTP request failed")
        raise HTTPException(
            status_code=502,
            detail="Could not get a response from Ollama.",
        )
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an unexpected response.",
        )
