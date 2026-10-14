"""Integration tests against the real Postgres from docker-compose.yml.

Skipped automatically when the database isn't running.
"""
import math

import pytest

psycopg = pytest.importorskip("psycopg")

from ingest import Chunk  # noqa: E402
from vector_store import DEFAULT_DSN, PgVectorStore  # noqa: E402

TEST_SOURCES = ("pytest_a.md", "pytest_b.md")


def database_available():
    try:
        psycopg.connect(DEFAULT_DSN, connect_timeout=2).close()
        return True
    except psycopg.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not database_available(), reason="Postgres not running (docker compose up -d)"
)


def unit(*values):
    """A 384-dimension vector pointing mostly along the given leading values."""
    vector = list(values) + [0.0] * (384 - len(values))
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


@pytest.fixture
def store():
    store = PgVectorStore()
    for source in TEST_SOURCES:
        store.delete_source(source)
    yield store
    for source in TEST_SOURCES:
        store.delete_source(source)


def chunk(source, n, section):
    return Chunk(text=f"{section} text", source=source, section=section,
                 chunk_id=f"{source}#{n}")


def test_search_ranks_by_cosine_similarity(store):
    store.replace_source(
        "pytest_a.md", "h1",
        [chunk("pytest_a.md", 0, "Close"), chunk("pytest_a.md", 1, "Far")],
        [unit(1, 0.1), unit(0, 1)],
    )

    results = store.search(unit(1, 0), top_k=2, source="pytest_a.md")

    assert [r["section"] for r in results] == ["Close", "Far"]
    assert results[0]["score"] > 0.99
    assert results[0]["chunk_id"] == "pytest_a.md#0"


def test_source_filter_and_replace(store):
    store.replace_source("pytest_a.md", "h1", [chunk("pytest_a.md", 0, "A")], [unit(1)])
    store.replace_source("pytest_b.md", "h1", [chunk("pytest_b.md", 0, "B")], [unit(1)])

    only_b = store.search(unit(1), top_k=5, source="pytest_b.md")
    assert [r["source"] for r in only_b] == ["pytest_b.md"]

    # Replacing a source swaps its rows and records the new hash.
    store.replace_source("pytest_a.md", "h2", [chunk("pytest_a.md", 0, "A2")], [unit(1)])
    assert store.source_hashes()["pytest_a.md"] == "h2"
    assert [r["section"] for r in store.search(unit(1), 5, "pytest_a.md")] == ["A2"]
