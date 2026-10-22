"""SummarAIze REST API.   Run: uvicorn api:app --reload   (docs at /docs)"""
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

load_dotenv()

from core.extractive import divide_into_topics, extractive_sentences
from core.pdf_utils import extract_pages, full_text
from core.rag import DocumentIndex, Embedder, answer, build_index, default_min_score
from core.slides import extractive_slide_specs, llm_slide_specs

app = FastAPI(title="SummarAIze API", version="1.0")

_embedder = None
INDEXES: dict[str, DocumentIndex] = {}  # demo-grade in-memory store (lost on restart)


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _read_pdf(file: UploadFile) -> list[dict]:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a .pdf file.")
    try:
        pages = extract_pages(file.file)
    except Exception:
        raise HTTPException(400, "Could not read this PDF.")
    if not pages:
        raise HTTPException(422, "No text found in the PDF (is it a scanned image?).")
    return pages


class AskRequest(BaseModel):
    doc_id: str
    question: str = Field(min_length=3)
    k: int = Field(4, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize")
def summarize(
    file: UploadFile = File(...),
    ratio: float = Form(0.2, gt=0, le=1),
    n_topics: int = Form(3, ge=1, le=10),
    mode: str = Form("llm"),
):
    """PDF in -> slide content (title + bullets per topic) as JSON."""
    if mode not in ("llm", "extractive"):
        raise HTTPException(400, "mode must be 'llm' or 'extractive'.")
    pages = _read_pdf(file)
    sentences = extractive_sentences(full_text(pages), ratio)
    if not sentences:
        raise HTTPException(422, "Not enough text to summarize.")
    topics, keywords = divide_into_topics(sentences, n_topics)
    try:
        specs = llm_slide_specs(topics, keywords) if mode == "llm" else extractive_slide_specs(topics, keywords)
    except RuntimeError as e:  # missing LLM configuration
        raise HTTPException(503, str(e))
    return {"mode": mode, "slides": specs}


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    """Index a PDF for question answering. Returns a doc_id to use with /ask."""
    pages = _read_pdf(file)
    index = build_index(pages, get_embedder())
    doc_id = uuid4().hex
    INDEXES[doc_id] = index
    return {"doc_id": doc_id, "pages": len(pages), "chunks": index.collection.count()}


@app.post("/ask")
def ask(req: AskRequest):
    """Answer a question about an indexed document, with page citations."""
    index = INDEXES.get(req.doc_id)
    if index is None:
        raise HTTPException(404, "Unknown doc_id. Upload the PDF to /documents first.")
    try:
        return answer(req.question, index, k=req.k, min_score=default_min_score())
    except RuntimeError as e:
        raise HTTPException(503, str(e))
