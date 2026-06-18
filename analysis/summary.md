# Book Buddy Evaluation Results

## Summary

- Questions evaluated: 12
- RAG with chapter limiter safe response rate: 100.0%
- RAG without chapter limiter safe response rate: 50.0%
- No-RAG safe response rate: 66.7%
- RAG with chapter limiter retrieval violation rate: 0.0%
- RAG without chapter limiter retrieval violation rate: 100.0%

## Label Counts

- RAG with limiter: {'safe': 12}
- RAG without limiter: {'spoiler': 6, 'safe': 6}
- No RAG: {'safe': 8, 'spoiler': 2, 'hallucination': 2}

## Interpretation

The three-way comparison isolates the value of retrieval and the value of the chapter limiter. RAG without the limiter can retrieve future chapters and leak spoilers. No-RAG often refuses safely, but can hallucinate or imply future plot. RAG with the chapter limiter produced grounded, spoiler-safe answers on this test set.
