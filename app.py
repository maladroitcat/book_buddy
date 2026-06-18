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


st.set_page_config(
    page_title="Book Buddy",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    """Grimdark, Age-of-Madness-inspired styling: iron, leather, and ember."""
    st.markdown(
        """
        <style>
        :root {
            --bb-ember: #a8322d;
            --bb-ember-bright: #c84a3f;
            --bb-bone: #ddd2c1;
            --bb-ash: #8c8170;
            --bb-iron-edge: #3a2f26;
        }

        /* Warm, smoky backdrop */
        .stApp {
            background:
                radial-gradient(1200px 600px at 80% -10%, rgba(168, 50, 45, 0.10), transparent 60%),
                linear-gradient(180deg, #16120f 0%, #120f0c 100%);
        }

        /* Thematic header */
        .bb-header { padding: 0.25rem 0 1.1rem 0; border-bottom: 1px solid var(--bb-iron-edge); margin-bottom: 1.2rem; }
        .bb-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 2.9rem; font-weight: 700; letter-spacing: 0.06em;
            color: var(--bb-bone); margin: 0; line-height: 1.1;
            text-shadow: 0 2px 14px rgba(168, 50, 45, 0.35);
        }
        .bb-title .bb-mark { color: var(--bb-ember-bright); }
        .bb-tagline { color: var(--bb-ash); font-size: 1.02rem; letter-spacing: 0.03em; margin-top: 0.35rem; }
        .bb-quote {
            color: var(--bb-ember); font-style: italic; font-size: 0.92rem;
            margin-top: 0.55rem; opacity: 0.85;
        }

        /* Chat bubbles: weathered leather cards */
        [data-testid="stChatMessage"] {
            background: linear-gradient(180deg, #241d18 0%, #1d1712 100%);
            border: 1px solid var(--bb-iron-edge);
            border-left: 3px solid var(--bb-ember);
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
            margin-bottom: 0.7rem;
        }

        /* Citations expander */
        [data-testid="stExpander"] {
            border: 1px solid var(--bb-iron-edge);
            border-radius: 8px;
            background: rgba(35, 28, 23, 0.5);
        }
        [data-testid="stExpander"] summary { color: var(--bb-ash); font-size: 0.88rem; }

        /* Sidebar: iron panel */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1b1510 0%, #14100d 100%);
            border-right: 1px solid var(--bb-iron-edge);
        }
        .bb-side-head {
            font-family: Georgia, serif; font-size: 1.25rem; color: var(--bb-bone);
            letter-spacing: 0.05em; margin-bottom: 0.2rem;
        }
        .bb-side-sub { color: var(--bb-ash); font-size: 0.82rem; margin-bottom: 0.8rem; }

        /* Progress bar in ember */
        [data-testid="stSidebar"] [role="progressbar"] > div > div {
            background: linear-gradient(90deg, var(--bb-ember) 0%, var(--bb-ember-bright) 100%) !important;
        }

        /* Chat input */
        [data-testid="stChatInput"] textarea { color: var(--bb-bone); }
        [data-testid="stChatInput"] { border-top: 1px solid var(--bb-iron-edge); }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    inject_styles()
    st.markdown(
        """
        <div class="bb-header">
            <div class="bb-title">Book <span class="bb-mark">Buddy</span></div>
            <div class="bb-tagline">A spoiler-safe companion for the Age of Madness</div>
            <div class="bb-quote">"The Long Eye sees only what it chooses to show."</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        st.markdown(
            """
            <div class="bb-side-head">⚔ How far have you read?</div>
            <div class="bb-side-sub">Book Buddy will never speak of pages you've yet to turn.</div>
            """,
            unsafe_allow_html=True,
        )
        selected_index = st.selectbox(
            "Current chapter",
            options=list(range(len(chapter_labels))),
            format_func=lambda i: chapter_labels[i],
            index=0,
        )
        current_chapter = int(chapters[selected_index]["chapter_number"])

        total_chapters = len(chapters)
        progress = (selected_index + 1) / total_chapters if total_chapters else 0.0
        st.progress(progress)
        st.caption(f"Chapter {selected_index + 1} of {total_chapters} — {int(progress * 100)}% through the tale")

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Chapters", total_chapters)
        col2.metric("Indexed passages", len(chunks))

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
