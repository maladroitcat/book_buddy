from collections.abc import Iterable

from google.auth.credentials import Credentials
from google import genai
from google.genai import types


DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        generation_model: str = DEFAULT_GENERATION_MODEL,
        use_vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        credentials: Credentials | None = None,
    ) -> None:
        if use_vertexai:
            if not project or not location:
                raise ValueError("Vertex AI requires both project and location.")
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                credentials=credentials,
            )
        else:
            self.client = genai.Client(api_key=api_key)
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.use_vertexai = use_vertexai
        self.project = project
        self.location = location

    def embed_document(self, title: str, text: str) -> list[float]:
        content = f"title: {title or 'none'} | text: {text}"
        return self._embed(content)

    def embed_query(self, query: str) -> list[float]:
        content = f"task: question answering | query: {query}"
        return self._embed(content)

    def embed_documents(self, documents: Iterable[tuple[str, str]]) -> list[list[float]]:
        return [self.embed_document(title, text) for title, text in documents]

    def generate_answer(
        self,
        question: str,
        context: str,
        current_chapter: int,
        citation_titles: list[str],
    ) -> str:
        system_instruction = (
            "You are Book Buddy, a spoiler-safe fiction reading assistant. "
            "Answer only from the provided context. Never use knowledge from later chapters, "
            "general world knowledge, or guesses. If the question asks about future events, "
            "refuse briefly and say what is known so far from the provided context. "
            "Do not reveal or infer anything beyond the reader's current chapter. "
            "Keep answers concise and grounded."
        )
        citation_text = "; ".join(citation_titles)
        prompt = (
            f"Reader current chapter: {current_chapter}\n"
            f"Allowed citation chapters: {citation_text}\n\n"
            f"Context from chapters 1-{current_chapter}:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with this exact prefix: "
            f"Based on chapters 1-{current_chapter}, "
        )
        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=1200,
                temperature=0.2,
            ),
        )
        return response.text or "I could not generate an answer from the available context."

    def _embed(self, content: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=content,
        )
        if not result.embeddings:
            raise RuntimeError("Gemini returned no embeddings.")
        return list(result.embeddings[0].values)
