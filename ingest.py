"""Document ingestion: load files from a folder and split them into chunks with metadata."""
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt"}
HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass(frozen=True)
class Chunk:
    text: str       # what gets embedded and sent to the model
    source: str     # file name
    section: str    # heading trail, e.g. "Platform Guide > Deployment"
    chunk_id: str   # stable reference, e.g. "guide.md#3"


def load_documents(path: str = "knowledge") -> dict:
    """Read one file, or every .md/.txt file in a folder."""
    root = Path(path)
    if root.is_file():
        return {root.name: root.read_text(encoding="utf-8")}
    if not root.is_dir():
        raise FileNotFoundError(f"Knowledge path not found: {root}")

    documents = {
        file.name: file.read_text(encoding="utf-8")
        for file in sorted(root.iterdir())
        if file.suffix.lower() in SUPPORTED_SUFFIXES
    }
    if not documents:
        raise ValueError(f"No .md or .txt files found in {root}")
    return documents


def split_sections(text: str, default_section: str) -> list:
    """Return (heading_trail, body) pairs, one per heading."""
    sections = []
    trail = {}      # heading level -> title
    lines = []

    def flush():
        body = "\n".join(lines).strip()
        if body:
            title = " > ".join(trail[level] for level in sorted(trail))
            sections.append((title or default_section, body))
        lines.clear()

    for line in text.splitlines():
        match = HEADING.match(line.strip())
        if match:
            flush()
            level = len(match.group(1))
            trail = {k: v for k, v in trail.items() if k < level}
            trail[level] = match.group(2).strip()
        else:
            lines.append(line)
    flush()
    return sections


def split_long_paragraph(paragraph: str, max_chars: int) -> list:
    """Break one oversized paragraph at sentence boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]

    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def pack_paragraphs(paragraphs: list, max_chars: int, overlap: int) -> list:
    """Group paragraphs into pieces of up to max_chars, repeating the last
    `overlap` paragraphs at the start of the next piece when they fit."""
    pieces, current = [], []
    for paragraph in paragraphs:
        if current and sum(len(p) for p in current) + len(paragraph) > max_chars:
            pieces.append("\n\n".join(current))
            carried = current[-overlap:] if overlap else []
            if sum(len(p) for p in carried) + len(paragraph) > max_chars:
                carried = []
            current = list(carried)
        current.append(paragraph)
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_document(
    name: str,
    text: str,
    max_chars: int = 800,
    overlap: int = 1,
) -> list:
    chunks = []
    for section, body in split_sections(text, default_section=Path(name).stem):
        paragraphs = []
        for paragraph in re.split(r"\n\s*\n", body):
            if paragraph.strip():
                paragraphs.extend(
                    split_long_paragraph(paragraph.strip(), max_chars)
                )
        for piece in pack_paragraphs(paragraphs, max_chars, overlap):
            chunks.append(
                Chunk(
                    text=f"{section}\n{piece}",
                    source=name,
                    section=section,
                    chunk_id=f"{name}#{len(chunks)}",
                )
            )
    return chunks


def ingest(path: str = "knowledge", max_chars: int = 800, overlap: int = 1) -> list:
    chunks = []
    for name, text in load_documents(path).items():
        chunks.extend(chunk_document(name, text, max_chars, overlap))
    if not chunks:
        raise ValueError("Knowledge documents contain no text.")
    return chunks


if __name__ == "__main__":
    all_chunks = ingest()
    print(f"{len(all_chunks)} chunks\n")
    for chunk in all_chunks:
        print(f"[{chunk.chunk_id}] {chunk.section} ({len(chunk.text)} chars)")
