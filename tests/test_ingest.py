import pytest

from ingest import chunk_document, ingest

DOC = """# Platform Guide

Overview of the platform.

## Deployment

We use blue-green deployments.

Traffic shifts after health checks pass.

Rollback is one command.

## Security

IAM roles follow least privilege.
"""


def test_headings_become_section_trails():
    sections = [chunk.section for chunk in chunk_document("guide.md", DOC)]

    assert sections == [
        "Platform Guide",
        "Platform Guide > Deployment",
        "Platform Guide > Security",
    ]


def test_every_chunk_has_body_text_not_just_a_heading():
    for chunk in chunk_document("guide.md", DOC):
        assert len(chunk.text.splitlines()) > 1


def test_long_section_is_split_with_overlap():
    chunks = chunk_document("guide.md", DOC, max_chars=75, overlap=1)
    deployment = [c for c in chunks if c.section.endswith("Deployment")]

    assert len(deployment) == 2
    assert "Traffic shifts" in deployment[0].text
    assert "Traffic shifts" in deployment[1].text   # repeated as overlap
    assert "Rollback" in deployment[1].text


def test_chunk_ids_are_unique_and_carry_the_source():
    chunks = chunk_document("guide.md", DOC, max_chars=75)

    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.source == "guide.md" for c in chunks)


def test_ingest_reads_supported_files_only(tmp_path):
    (tmp_path / "a.md").write_text(DOC, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Plain text, no headings.", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    chunks = ingest(str(tmp_path))

    assert {c.source for c in chunks} == {"a.md", "notes.txt"}
    assert any(c.section == "notes" for c in chunks)


def test_ingest_fails_loudly_on_empty_folder(tmp_path):
    with pytest.raises(ValueError):
        ingest(str(tmp_path))
