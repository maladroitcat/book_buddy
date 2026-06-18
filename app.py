import json
from pathlib import Path

import numpy as np
import streamlit as st

from src.gcs_store import credentials_from_info, download_file, storage_client_from_info
from src.gemini_client import GeminiClient
from src.retrieval import build_context, citation_titles, load_chunks, rank_chunks


LOCAL_ARTIFACT_DIR = Path("artifacts")
DEFAULT_TOP_K = 5
DEFAULT_CHUNKS_BLOB = "processed/chunks.jsonl"
DEFAULT_EMBEDDINGS_BLOB = "processed/embeddings.npy"
DEFAULT_MANIFEST_BLOB = "processed/manifest.json"


st.set_page_config(page_title="Book Buddy", layout="wide")


def _service_account_info() -> dict | None:
    if "gcp_service_account" not in st.secrets:
        return None
    info = dict(st.secrets["gcp_service_account"])
    if "private_key" in info:
        info["private_key"] = info["private_key"].replace("\\n", "\n")
    return info


def make_gemini_client() -> GeminiClient:
    service_account_info = _service_account_info()
    gcp_project = st.secrets.get("GCP_PROJECT_ID", "")
    gcp_location = st.secrets.get("GCP_LOCATION", "")
    use_vertexai = bool(st.secrets.get("USE_VERTEX_AI", True))
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if use_vertexai:
        if not gcp_project:
            raise RuntimeError("Missing GCP_PROJECT_ID in Streamlit secrets.")
        if not gcp_location:
            raise RuntimeError("Missing GCP_LOCATION in Streamlit secrets.")
        return GeminiClient(
            use_vertexai=True,
            project=gcp_project,
            location=gcp_location,
            credentials=credentials_from_info(service_account_info),
        )
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in Streamlit secrets.")
    return GeminiClient(api_key=api_key)


@st.cache_resource(show_spinner="Loading Book Buddy index...")
def load_book_index() -> tuple[list, np.ndarray, dict]:
    bucket_name = st.secrets.get("GCS_BUCKET_NAME", "")
    if not bucket_name:
        raise RuntimeError("Missing GCS_BUCKET_NAME in Streamlit secrets.")
    chunks_blob = st.secrets.get("CHUNKS_GCS_BLOB", DEFAULT_CHUNKS_BLOB)
    embeddings_blob = st.secrets.get("EMBEDDINGS_GCS_BLOB", DEFAULT_EMBEDDINGS_BLOB)
    manifest_blob = st.secrets.get("MANIFEST_GCS_BLOB", DEFAULT_MANIFEST_BLOB)
    client = storage_client_from_info(_service_account_info())

    LOCAL_ARTIFACT_DIR.mkdir(exist_ok=True)
    chunks_path = download_file(bucket_name, chunks_blob, LOCAL_ARTIFACT_DIR / "chunks.jsonl", client)
    embeddings_path = download_file(bucket_name, embeddings_blob, LOCAL_ARTIFACT_DIR / "embeddings.npy", client)
    manifest_path = download_file(bucket_name, manifest_blob, LOCAL_ARTIFACT_DIR / "manifest.json", client)
    chunks = load_chunks(chunks_path)
    embeddings = np.load(embeddings_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not chunks or embeddings.size == 0:
        raise RuntimeError("Downloaded index artifacts are empty.")
    return chunks, embeddings, manifest


def main() -> None:
    st.title("Book Buddy")
    st.caption("Spoiler-safe reading assistant for one private EPUB")

    try:
        chunks, embeddings, manifest = load_book_index()
    except Exception as exc:
        st.error(f"Could not load the book index: {exc}")
        st.stop()

    try:
        client = make_gemini_client()
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    chapters = manifest.get("chapters", [])
    chapter_labels = [f"{chapter['chapter_number']}. {chapter['title']}" for chapter in chapters]

    with st.sidebar:
        st.header("Reading progress")
        selected_index = st.selectbox(
            "Current chapter",
            options=list(range(len(chapter_labels))),
            format_func=lambda i: chapter_labels[i],
            index=0,
        )
        current_chapter = int(chapters[selected_index]["chapter_number"])
        st.divider()
        st.metric("Available chapters", len(chapters))
        st.metric("Indexed chunks", len(chunks))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("Citations"):
                    for citation in message["citations"]:
                        st.write(citation)

    question = st.chat_input("Ask a spoiler-safe question...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking what you know so far..."):
            query_embedding = client.embed_query(question)
            results = rank_chunks(query_embedding, chunks, embeddings, current_chapter, top_k=DEFAULT_TOP_K)
            if not results:
                answer = f"Based on chapters 1-{current_chapter}, I do not have enough context to answer that."
                citations = []
            else:
                context = build_context(results)
                citations = citation_titles(results)
                answer = client.generate_answer(question, context, current_chapter, citations)
            st.markdown(answer)
            if citations:
                with st.expander("Citations"):
                    for citation in citations:
                        st.write(citation)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )


if __name__ == "__main__":
    main()
