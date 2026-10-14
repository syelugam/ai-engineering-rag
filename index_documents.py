"""Load the knowledge folder into Postgres, re-embedding only files that changed.

Run:    python index_documents.py
Force:  python index_documents.py --force   (re-embed everything, e.g. after
                                             changing chunk size)
"""
import argparse
import hashlib

from ingest import chunk_document, load_documents

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sync(store, embed, folder: str = "knowledge", force: bool = False) -> dict:
    """Bring the store in line with the folder. Returns {file: what happened}."""
    documents = load_documents(folder)
    indexed = store.source_hashes()
    report = {}

    for name, text in documents.items():
        digest = content_hash(text)
        if not force and indexed.get(name) == digest:
            report[name] = "unchanged"
            continue

        chunks = chunk_document(name, text)
        embeddings = embed([chunk.text for chunk in chunks]) if chunks else []
        store.replace_source(name, digest, chunks, embeddings)
        report[name] = f"indexed {len(chunks)} chunks"

    for name in sorted(set(indexed) - set(documents)):
        store.delete_source(name)
        report[name] = "removed (file no longer in folder)"

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="knowledge")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    from vector_store import PgVectorStore

    store = PgVectorStore()
    model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(texts):
        return model.encode(texts, normalize_embeddings=True)

    for name, outcome in sync(store, embed, args.folder, args.force).items():
        print(f"{name}: {outcome}")
    print(f"\n{store.count()} chunks in the database")


if __name__ == "__main__":
    main()
