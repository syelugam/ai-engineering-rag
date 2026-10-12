import httpx

from pathlib import Path

from sentence_transformers import SentenceTransformer, util

DOCUMENT_PATH = Path("knowledge/aws_architecture.txt")

text = DOCUMENT_PATH.read_text(encoding="utf-8")

# Split the document into small sections.
chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]

# Load a small, local embedding model.
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Convert each document section into a vector.
document_embeddings = embedding_model.encode(
    chunks,
    convert_to_tensor=True,
    normalize_embeddings=True,
)

question = "How do we safely release a new application version?"

# Convert the question into a vector.
question_embedding = embedding_model.encode(
    question,
    convert_to_tensor=True,
    normalize_embeddings=True,
)

# Find the most relevant sections by semantic similarity.
scores = util.cos_sim(
    question_embedding,
    document_embeddings,
)[0]

top_k = min(2, len(chunks))
results = scores.topk(top_k)

context = "\n\n".join(
    chunks[index]
    for index in results.indices.tolist()
)

prompt = f"""
You are an assistant answering questions about our AWS architecture.

Use only the supplied context to answer the question.
If the context does not contain the answer, say you don't know.
Do not invent architecture details.

Context:
{context}

Question:
{question}

Answer with a brief explanation and identify the relevant section.
"""

response = httpx.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "gemma4:e2b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
    },
    timeout=120.0,
)
response.raise_for_status()

answer = response.json()["message"]["content"]

print("\nRAG answer:\n")
print(answer)
