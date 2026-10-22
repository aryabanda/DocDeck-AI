"""PDF text extraction that keeps page numbers (needed for citations)."""
from pypdf import PdfReader


def extract_pages(file) -> list[dict]:
    """Return [{"page": 1, "text": "..."}, ...] for every page that has text.

    `file` can be a path or any file-like object (e.g. Streamlit's UploadedFile).
    """
    reader = PdfReader(file)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""  # one bad page should not kill the whole document
        text = " ".join(text.split())  # collapse newlines / repeated spaces
        if text:
            pages.append({"page": number, "text": text})
    return pages


def full_text(pages: list[dict]) -> str:
    return " ".join(p["text"] for p in pages)
