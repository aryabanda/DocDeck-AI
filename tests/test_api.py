import os

import pytest
from fastapi.testclient import TestClient

import api
from core import rag
from tests.test_core import FakeEmbedder

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "docs", "sample.pdf")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(api, "answer", lambda q, idx, k=4: rag.answer(q, idx, k, llm=lambda m: "fake answer [p. 1]"))
    return TestClient(api.app)


def _pdf():
    return {"file": ("sample.pdf", open(SAMPLE, "rb"), "application/pdf")}


def test_health_and_summarize_extractive(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/summarize", files=_pdf(), data={"mode": "extractive", "n_topics": 3})
    assert r.status_code == 200 and r.json()["slides"]
    assert {"title", "bullets"} <= set(r.json()["slides"][0])


def test_summarize_validation(client):
    assert client.post("/summarize", files=_pdf(), data={"mode": "nope"}).status_code == 400
    assert client.post("/summarize", files=_pdf(), data={"ratio": 5}).status_code == 422
    bad = {"file": ("notes.txt", b"hello", "text/plain")}
    assert client.post("/summarize", files=bad).status_code == 400


def test_summarize_llm_mode_reports_missing_config(client, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    r = client.post("/summarize", files=_pdf(), data={"mode": "llm"})
    assert r.status_code == 503 and "not set" in r.json()["detail"]


def test_upload_then_ask(client):
    doc = client.post("/documents", files=_pdf()).json()
    assert doc["chunks"] > 0
    r = client.post("/ask", json={"doc_id": doc["doc_id"], "question": "what is this about?"})
    assert r.status_code == 200 and r.json()["answer"] == "fake answer [p. 1]"
    assert r.json()["sources"]
    assert client.post("/ask", json={"doc_id": "missing", "question": "hello there"}).status_code == 404
