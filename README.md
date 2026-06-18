# SPOILERS AHEAD!

# Book Buddy

Book Buddy is a spoiler-safe reading assistant for *A Little Hatred*. It helps a reader ask normal, messy, mid-book questions like “wait who is Savine again?” or “should I remember this vision?” without accidentally revealing plot details from chapters they have not reached yet.

The app is built as a small RAG system:

- The book is chunked and embedded ahead of time.
- The deployed Streamlit app downloads those precomputed chunks and embeddings from Google Cloud Storage.
- The reader selects their current chapter.
- The app embeds the user’s question with Vertex AI Gemini.
- Retrieval only allows chunks where `chapter_number <= current_chapter`.
- Gemini generates an answer from that spoiler-safe context.

The raw EPUB, local artifacts, service-account key, and Streamlit secrets are intentionally not committed.

## Why This Matters

Normal chatbots can answer fiction questions using information from anywhere in the book, including future chapters. That creates spoilers and can also produce hallucinated plot details. Book Buddy uses chapter-aware retrieval to test whether a model can answer questions while respecting a reader’s progress.

## Models

- Embeddings: `gemini-embedding-001`
- Generation: `gemini-2.5-flash`
- Deployment: Streamlit Community Cloud
- Storage: Google Cloud Storage

This uses transfer learning because the pre-trained Gemini embedding and generation models are reused for a new book-specific question-answering task without training a model from scratch.

## Evaluation Results

Evaluation artifacts are in `analysis/`.

The evaluation compares three conditions:

1. **RAG with chapter limiter**
2. **RAG without chapter limiter**
3. **No RAG**

Each response was manually labeled as:

- `safe`
- `spoiler`
- `hallucination`

Current preliminary results on 12 book-specific questions:

| Condition | Safe Rate | Spoiler Count | Hallucination Count |
|---|---:|---:|---:|
| RAG with chapter limiter | 100.0% | 0 | 0 |
| RAG without chapter limiter | 50.0% | 6 | 0 |
| No RAG | 66.7% | 2 | 2 |

Retrieval boundary results:

| Condition | Future-Chapter Retrieval Violation Rate |
|---|---:|
| RAG with chapter limiter | 0.0% |
| RAG without chapter limiter | 100.0% |

The evaluation showed that RAG alone is not enough. Without the chapter limiter, semantic search often retrieves future chapters and can leak spoilers. The chapter limiter is the key safety improvement.

## Analysis Files

- `analysis/questions.csv`: evaluation questions
- `analysis/summary.md`: summary of results
- `analysis/results_exploration.ipynb`: notebook with bar charts

The detailed per-response result files (`analysis/rag_limited_results.csv`, `analysis/rag_unlimited_results.csv`, `analysis/no_rag_results.csv`) are intentionally **not committed**, because they contain verbatim model answers that quote passages from the source book. The aggregate metrics from those files are reproduced in the tables above and in `analysis/summary.md`.

## Disclaimer

This project is for **educational purposes only**. It was built as a coursework demonstration of chapter-aware retrieval-augmented generation and is not intended for commercial use or public distribution of book content.

*A Little Hatred* by Joe Abercrombie is copyrighted material. The source EPUB, its precomputed chunks and embeddings, and the detailed evaluation result CSVs (which contain verbatim book passages) are deliberately excluded from this repository via `.gitignore` to avoid redistributing copyrighted text. Only aggregate evaluation metrics and the questions used for testing are included. No part of the book is reproduced in this repository, and no copyright infringement is intended. To run the app, you must supply your own legally obtained copy of the book.