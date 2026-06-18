import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gemini_client import GeminiClient
from src.models import RetrievalResult
from src.retrieval import build_context, citation_titles, load_chunks, rank_chunks


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
QUESTIONS_PATH = ANALYSIS_DIR / "questions.csv"
RAG_LIMITED_CSV = ANALYSIS_DIR / "rag_limited_results.csv"
RAG_UNLIMITED_CSV = ANALYSIS_DIR / "rag_unlimited_results.csv"
NO_RAG_CSV = ANALYSIS_DIR / "no_rag_results.csv"
SUMMARY_JSON = ANALYSIS_DIR / "summary.json"
SUMMARY_MD = ANALYSIS_DIR / "summary.md"
CHUNKS_PATH = ROOT / "artifacts" / "chunks.jsonl"
EMBEDDINGS_PATH = ROOT / "artifacts" / "embeddings.npy"


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rank_unlimited(query_embedding: list[float], chunks: list, embeddings: np.ndarray, top_k: int = 5) -> list[RetrievalResult]:
    query = np.asarray(query_embedding, dtype=np.float32)
    matrix = embeddings.astype(np.float32)
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = np.divide(matrix @ query, denom, out=np.zeros_like(denom), where=denom != 0)
    order = np.argsort(scores)[::-1][:top_k]
    return [RetrievalResult(chunks[int(i)], float(scores[int(i)])) for i in order]


def generate_no_rag_answer(client: GeminiClient, question: str, current_chapter: int) -> str:
    prompt = (
        "You are answering a reader's question about a fiction book. "
        "The reader says where they are in the book and asks not to be spoiled. "
        "Answer without retrieved book context.\n\n"
        f"Current chapter: {current_chapter}\n"
        f"Question: {question}\n"
    )
    response = client.client.models.generate_content(
        model=client.generation_model,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=1200, temperature=0.2),
    )
    return normalize(response.text or "I could not generate an answer.")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    questions = load_questions()
    gcp_project = os.environ.get("GCP_PROJECT_ID")
    gcp_location = os.environ.get("GCP_LOCATION")
    if not gcp_project or not gcp_location:
        raise RuntimeError("Set GCP_PROJECT_ID and GCP_LOCATION before running analysis.")
    client = GeminiClient(use_vertexai=True, project=gcp_project, location=gcp_location)

    rag_limited_rows: list[dict] = []
    rag_unlimited_rows: list[dict] = []
    no_rag_rows: list[dict] = []

    for question_row in questions:
        group = question_row["group"]
        question = question_row["question"]
        current_chapter = int(question_row["current_chapter"])
        query_embedding = client.embed_query(question)

        limited_results = rank_chunks(query_embedding, chunks, embeddings, current_chapter, top_k=5)
        unlimited_results = rank_unlimited(query_embedding, chunks, embeddings, top_k=5)
        limited_chapters = [result.chunk.chapter_number for result in limited_results]
        unlimited_chapters = [result.chunk.chapter_number for result in unlimited_results]
        limited_violation = any(chapter > current_chapter for chapter in limited_chapters)
        unlimited_violation = any(chapter > current_chapter for chapter in unlimited_chapters)

        limited_context = build_context(limited_results)
        unlimited_context = build_context(unlimited_results)

        if limited_results:
            limited_response = normalize(
                client.generate_answer(question, limited_context, current_chapter, citation_titles(limited_results))
            )
        else:
            limited_response = f"Based on chapters 1-{current_chapter}, I do not have enough context to answer that."

        if unlimited_results:
            unlimited_response = normalize(
                client.generate_answer(question, unlimited_context, current_chapter, citation_titles(unlimited_results))
            )
        else:
            unlimited_response = f"Based on chapters 1-{current_chapter}, I do not have enough context to answer that."

        no_rag_response = generate_no_rag_answer(client, question, current_chapter)

        common = {
            "group": group,
            "current_chapter": current_chapter,
            "question": question,
            "limited_retrieved_chapters": json.dumps(limited_chapters),
            "unlimited_retrieved_chapters": json.dumps(unlimited_chapters),
            "limited_retrieval_violation": limited_violation,
            "unlimited_retrieval_violation": unlimited_violation,
        }
        rag_limited_rows.append(
            {
                **common,
                "condition": "rag_limited",
                "response": limited_response,
                "safety_label": "",
                "safety_reason": "",
            }
        )
        rag_unlimited_rows.append(
            {
                **common,
                "condition": "rag_unlimited",
                "response": unlimited_response,
                "safety_label": "",
                "safety_reason": "",
            }
        )
        no_rag_rows.append(
            {
                **common,
                "condition": "no_rag",
                "response": no_rag_response,
                "safety_label": "",
                "safety_reason": "",
            }
        )

    write_csv(RAG_LIMITED_CSV, rag_limited_rows)
    write_csv(RAG_UNLIMITED_CSV, rag_unlimited_rows)
    write_csv(NO_RAG_CSV, no_rag_rows)
    print(f"Wrote {RAG_LIMITED_CSV}")
    print(f"Wrote {RAG_UNLIMITED_CSV}")
    print(f"Wrote {NO_RAG_CSV}")


if __name__ == "__main__":
    main()
