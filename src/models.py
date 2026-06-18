from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    chapter_number: int
    title: str
    href: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    book_id: str
    chapter_number: int
    chapter_title: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
