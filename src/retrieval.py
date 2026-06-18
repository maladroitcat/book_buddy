import json
from pathlib import Path

import numpy as np

from src.models import Chunk, RetrievalResult


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def rank_chunks(
    query_embedding: list[float] | np.ndarray,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    current_chapter: int,
    top_k: int = 5,
) -> list[RetrievalResult]:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count and embedding count do not match.")
    allowed_indices = [i for i, chunk in enumerate(chunks) if chunk.chapter_number <= current_chapter]
    if not allowed_indices:
        return []

    query = np.asarray(query_embedding, dtype=np.float32)
    matrix = embeddings[allowed_indices].astype(np.float32)
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = matrix_norms * query_norm
    scores = np.divide(matrix @ query, denom, out=np.zeros_like(matrix_norms), where=denom != 0)
    order = np.argsort(scores)[::-1][:top_k]
    return [
        RetrievalResult(chunks[allowed_indices[int(i)]], float(scores[int(i)]))
        for i in order
    ]


def build_context(results: list[RetrievalResult], max_chars: int = 9000) -> str:
    blocks: list[str] = []
    total = 0
    for result in results:
        chunk = result.chunk
        block = (
            f"[Chapter {chunk.chapter_number}: {chunk.chapter_title}]\n"
            f"{chunk.text}"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def citation_titles(results: list[RetrievalResult]) -> list[str]:
    titles: list[str] = []
    seen: set[tuple[int, str]] = set()
    for result in results:
        key = (result.chunk.chapter_number, result.chunk.chapter_title)
        if key in seen:
            continue
        seen.add(key)
        titles.append(f"Chapter {result.chunk.chapter_number}: {result.chunk.chapter_title}")
    return titles
