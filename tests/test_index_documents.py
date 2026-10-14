"""Incremental indexing logic, tested with an in-memory fake store."""
from index_documents import content_hash, sync


class FakeStore:
    def __init__(self):
        self.rows = {}        # source -> (hash, chunks)
        self.writes = []      # sources written, in order

    def source_hashes(self):
        return {source: digest for source, (digest, _) in self.rows.items()}

    def replace_source(self, source, digest, chunks, embeddings):
        assert len(chunks) == len(embeddings)
        self.rows[source] = (digest, chunks)
        self.writes.append(source)

    def delete_source(self, source):
        del self.rows[source]


def fake_embed(texts):
    return [[0.0] * 3 for _ in texts]


def write(folder, name, text):
    (folder / name).write_text(text, encoding="utf-8")


def test_first_run_indexes_every_file(tmp_path):
    write(tmp_path, "a.md", "# A\nAlpha.")
    write(tmp_path, "b.md", "# B\nBeta.")
    store = FakeStore()

    report = sync(store, fake_embed, str(tmp_path))

    assert report == {"a.md": "indexed 1 chunks", "b.md": "indexed 1 chunks"}


def test_second_run_skips_unchanged_files(tmp_path):
    write(tmp_path, "a.md", "# A\nAlpha.")
    store = FakeStore()
    sync(store, fake_embed, str(tmp_path))

    report = sync(store, fake_embed, str(tmp_path))

    assert report == {"a.md": "unchanged"}
    assert store.writes == ["a.md"]          # embedded once, not twice


def test_edited_file_is_reindexed(tmp_path):
    write(tmp_path, "a.md", "# A\nAlpha.")
    store = FakeStore()
    sync(store, fake_embed, str(tmp_path))

    write(tmp_path, "a.md", "# A\nAlpha, revised.")
    report = sync(store, fake_embed, str(tmp_path))

    assert report["a.md"] == "indexed 1 chunks"
    assert store.rows["a.md"][0] == content_hash("# A\nAlpha, revised.")


def test_deleted_file_is_removed_from_the_store(tmp_path):
    write(tmp_path, "a.md", "# A\nAlpha.")
    write(tmp_path, "b.md", "# B\nBeta.")
    store = FakeStore()
    sync(store, fake_embed, str(tmp_path))

    (tmp_path / "b.md").unlink()
    report = sync(store, fake_embed, str(tmp_path))

    assert "removed" in report["b.md"]
    assert set(store.rows) == {"a.md"}


def test_force_reindexes_even_when_unchanged(tmp_path):
    write(tmp_path, "a.md", "# A\nAlpha.")
    store = FakeStore()
    sync(store, fake_embed, str(tmp_path))

    report = sync(store, fake_embed, str(tmp_path), force=True)

    assert report == {"a.md": "indexed 1 chunks"}
