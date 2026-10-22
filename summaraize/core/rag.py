"""Retrieval-Augmented Generation over an uploaded PDF.

Pipeline: pages -> chunks (with page numbers) -> embeddings -> ChromaDB
          -> top-k retrieval -> LLM answer that must cite pages.
"""
import os
import uuid

import chromadb
from chromadb.config import Settings

from core.llm import chat

NOT_FOUND = "I couldn't find that in the document."

RAG_SYSTEM = (
    "You answer questions about a document using ONLY the context passages provided. "
    "Each passage starts with its page, like [p. 3]. Cite the page after each claim, "
    "for example 'The fee is 5% [p. 3].' End every sentence and every bullet point "
    "with its citation, and use plain square brackets exactly like [p. 3]. "
    f"If the passages do not contain the answer, reply exactly: {NOT_FOUND} "
    "Never use outside knowledge."
)

_client = None


def _chroma():
    global _client
    if _client is None:
        _client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    return _client


class Embedder:
    """Local sentence-transformers model (free, no API key). Downloads once (~90 MB)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def chunk_pages(pages: list[dict], chunk_words: int = 180, overlap: int = 40) -> list[dict]:
    """Sliding word window with overlap. Chunks never cross a page, so every
    chunk has exactly one page number to cite."""
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    step = chunk_words - overlap
    chunks = []
    for page in pages:
        words = page["text"].split()
        start, index = 0, 0
        while start < len(words):
            piece = words[start : start + chunk_words]
            chunks.append({"id": f"p{page['page']}-c{index}", "page": page["page"], "text": " ".join(piece)})
            if start + chunk_words >= len(words):
                break
            start += step
            index += 1
    return chunks


class DocumentIndex:
    def __init__(self, chunks: list[dict], embedder):
        if not chunks:
            raise ValueError("No text chunks to index.")
        self.embedder = embedder
        self.collection = _chroma().create_collection(
            name=f"doc_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
        )
        for i in range(0, len(chunks), 64):
            batch = chunks[i : i + 64]
            texts = [c["text"] for c in batch]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=texts,
                embeddings=embedder.encode(texts),
                metadatas=[{"page": c["page"]} for c in batch],
            )

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        result = self.collection.query(
            query_embeddings=self.embedder.encode([query]),
            n_results=min(k, self.collection.count()),
        )
        hits = []
        for id_, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"id": id_, "text": text, "page": meta["page"], "score": round(1 - dist, 4)})
        return hits


def build_index(pages: list[dict], embedder, chunk_words: int = 180, overlap: int = 40) -> DocumentIndex:
    return DocumentIndex(chunk_pages(pages, chunk_words, overlap), embedder)


def normalize_citations(text: str) -> str:
    """Some models emit full-width brackets (e.g. 【p. 6】); show plain [p. 6]."""
    return text.replace("\u3010", "[").replace("\u3011", "]")


def default_min_score():
    """Optional similarity cutoff from the RAG_MIN_SCORE env var (unset = no cutoff)."""
    value = os.getenv("RAG_MIN_SCORE")
    return float(value) if value else None


def answer(question: str, index: DocumentIndex, k: int = 4, llm=chat, min_score=None) -> dict:
    """Retrieve, then ask the LLM. If the best passage's similarity is below
    `min_score`, skip the LLM call and return the not-found message (saves cost)."""
    hits = index.retrieve(question, k)
    if min_score is not None and (not hits or hits[0]["score"] < min_score):
        return {"answer": NOT_FOUND, "sources": hits, "llm_skipped": True}
    context = "\n\n".join(f"[p. {h['page']}] {h['text']}" for h in hits)
    messages = [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return {"answer": normalize_citations(llm(messages)), "sources": hits}
