"""Chunk storage and vector search in Postgres with pgvector."""
import os
from typing import Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

try:
    from dotenv import load_dotenv

    load_dotenv()   # lets DATABASE_URL live in .env
except ImportError:
    pass

# Default matches docker-compose.yml. Set DATABASE_URL (e.g. in .env) to use
# another Postgres, such as Postgres.app on port 5432.
DEFAULT_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag"
)
EMBEDDING_DIM = 384   # all-MiniLM-L6-v2 produces 384 numbers per text


def as_vector(values) -> np.ndarray:
    """pgvector's adapter sends NumPy float32 arrays as the vector type."""
    return np.asarray(values, dtype=np.float32)


class PgVectorStore:
    def __init__(self, dsn: str = DEFAULT_DSN, dim: int = EMBEDDING_DIM):
        self.dsn = dsn
        self.dim = dim
        self.ensure_schema()

    def _connect(self):
        conn = psycopg.connect(self.dsn, autocommit=True)
        register_vector(conn)
        return conn

    def ensure_schema(self) -> None:
        # The extension must exist before register_vector can find the type.
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id     text PRIMARY KEY,
                    source       text NOT NULL,
                    section      text NOT NULL,
                    text         text NOT NULL,
                    content_hash text NOT NULL,
                    embedding    vector({self.dim}) NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source)"
            )
            # HNSW: approximate nearest-neighbour index for cosine distance.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )

    def source_hashes(self) -> dict:
        """{source file: hash of its content when it was last indexed}"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT source, content_hash FROM chunks"
            ).fetchall()
        return {source: content_hash for source, content_hash in rows}

    def replace_source(self, source: str, content_hash: str, chunks, embeddings) -> None:
        """Swap all chunks of one file in a single transaction."""
        with self._connect() as conn:
            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks "
                        "(chunk_id, source, section, text, content_hash, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        [
                            (c.chunk_id, c.source, c.section, c.text, content_hash, as_vector(e))
                            for c, e in zip(chunks, embeddings)
                        ],
                    )

    def delete_source(self, source: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE source = %s", (source,))

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def search(
        self,
        query_embedding,
        top_k: int = 3,
        source: Optional[str] = None,
    ) -> list:
        # <=> is cosine distance; 1 - distance = the cosine similarity used before.
        sql = (
            "SELECT chunk_id, source, section, text, "
            "1 - (embedding <=> %(q)s) AS score FROM chunks "
        )
        params = {"q": as_vector(query_embedding), "k": top_k}
        if source:
            sql += "WHERE source = %(source)s "
            params["source"] = source
        sql += "ORDER BY embedding <=> %(q)s LIMIT %(k)s"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            {
                "text": text,
                "score": float(score),
                "source": src,
                "section": section,
                "chunk_id": chunk_id,
            }
            for chunk_id, src, section, text, score in rows
        ]

    def list_sections(self) -> list:
        """[(source, section), ...] in document order, for the agent's table of contents."""
        with self._connect() as conn:
            rows = conn.execute(
                # order by the chunk number after '#', numerically (#10 after #9)
                "SELECT source, section, min(split_part(chunk_id, '#', 2)::int) AS n "
                "FROM chunks GROUP BY source, section ORDER BY source, n"
            ).fetchall()
        return [(source, section) for source, section, _ in rows]
