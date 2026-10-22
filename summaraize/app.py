"""SummarAIze - Streamlit UI: PDF -> slides, and ask questions about the PDF."""
import hashlib
import io
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
try:  # Streamlit Cloud: secrets -> env vars (st.secrets raises if no secrets file)
    for _key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

from core.extractive import divide_into_topics, extractive_sentences
from core.pdf_utils import extract_pages, full_text
from core.rag import Embedder, answer, build_index, default_min_score
from core.slides import build_presentation, extractive_slide_specs, llm_slide_specs

TEMPLATE = "organic_template.pptx"  # optional: used if the file exists next to app.py
MIME_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

st.set_page_config(page_title="SummarAIze", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def get_embedder():
    return Embedder()


@st.cache_data(show_spinner=False)
def read_pdf(data: bytes):
    return extract_pages(io.BytesIO(data))


st.title("SummarAIze")
st.caption("Turn a PDF into a slide deck, or ask it questions with page citations.")

upload = st.file_uploader("Upload a PDF", type=["pdf"])
if upload is None:
    st.stop()

data = upload.getvalue()
pages = read_pdf(data)
if not pages:
    st.error("No text could be extracted (the PDF may be scanned images).")
    st.stop()
doc_id = hashlib.md5(data).hexdigest()
st.write(f"**{upload.name}**: {len(pages)} pages with text")

tab_slides, tab_ask = st.tabs(["Generate slides", "Ask the document"])

# ---------------------------------------------------------------- slides tab
with tab_slides:
    mode = st.radio(
        "Slide writer",
        ["Extractive (no API key needed)", "LLM-written (cleaner titles and bullets)"],
        horizontal=True,
    )
    col1, col2 = st.columns(2)
    ratio = col1.slider("Summary ratio (share of sentences kept)", 0.1, 1.0, 0.2)
    n_topics = col2.slider("Number of topics (slides groups)", 1, 10, 3)
    title = st.text_input("Presentation title", value=os.path.splitext(upload.name)[0])
    subtitle = st.text_input("Subtitle")

    if st.button("Generate PowerPoint", type="primary"):
        try:
            with st.spinner("Summarizing and building slides..."):
                sentences = extractive_sentences(full_text(pages), ratio)
                if not sentences:
                    st.error("Could not find enough sentences to summarize.")
                    st.stop()
                topics, keywords = divide_into_topics(sentences, n_topics)
                if mode.startswith("LLM"):
                    specs = llm_slide_specs(topics, keywords)
                else:
                    specs = extractive_slide_specs(topics, keywords)
                st.session_state["deck"] = {
                    "doc": doc_id,
                    "specs": specs,
                    "pptx": build_presentation(specs, title, subtitle, TEMPLATE),
                }
        except RuntimeError as e:  # e.g. missing API key
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

    deck = st.session_state.get("deck")
    if deck and deck["doc"] == doc_id:
        fallbacks = sum(1 for s in deck["specs"] if s.get("fallback"))
        if fallbacks:
            st.warning(f"{fallbacks} slide(s) used extractive text because the model's output was unusable.")
        st.download_button("Download presentation", deck["pptx"], file_name="summary.pptx", mime=MIME_PPTX)
        with st.expander("Preview slide text"):
            for s in deck["specs"]:
                st.markdown(f"**{s['title']}**")
                for b in s["bullets"]:
                    st.markdown(f"- {b}")

# -------------------------------------------------------------------- ask tab
with tab_ask:
    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, 4)

    if st.session_state.get("index_doc") != doc_id:
        with st.spinner("Indexing document (chunking, embedding, storing in ChromaDB)..."):
            st.session_state["index"] = build_index(pages, get_embedder())
            st.session_state["index_doc"] = doc_id
            st.session_state["history"] = []

    for turn in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            with st.expander("Sources"):
                for h in turn["sources"]:
                    st.markdown(f"**Page {h['page']}** (similarity {h['score']})")
                    st.caption(h["text"])

    question = st.chat_input("Ask something about the document")
    if question:
        try:
            with st.spinner("Searching and answering..."):
                result = answer(question, st.session_state["index"], k=top_k, min_score=default_min_score())
            st.session_state["history"].append({"question": question, **result})
            st.rerun()
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
