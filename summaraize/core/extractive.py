"""Your original NLP pipeline (spaCy frequency scoring + TF-IDF/KMeans topics),
cleaned up: model cached, sentence order preserved, no `split('. ')` hack,
and topic keywords are matched to the right cluster."""
from functools import lru_cache
from heapq import nlargest
from string import punctuation

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS

MAX_SENTENCE_WORDS = 50  # very long sentences score high just because they are long


@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def extractive_sentences(text: str, ratio: float = 0.2) -> list[str]:
    """Pick the top `ratio` of sentences by word-frequency score, in document order."""
    doc = get_nlp()(text)
    freqs: dict[str, int] = {}
    for token in doc:
        w = token.text.lower()
        if w not in STOP_WORDS and w not in punctuation and not token.is_space:
            freqs[w] = freqs.get(w, 0) + 1
    if not freqs:
        return []
    top = max(freqs.values())
    freqs = {w: c / top for w, c in freqs.items()}

    sentences = [s for s in doc.sents if 4 <= len(s) <= MAX_SENTENCE_WORDS]
    scores = {}
    for sent in sentences:
        scores[sent] = sum(freqs.get(t.text.lower(), 0) for t in sent)

    k = max(1, int(len(sentences) * ratio))
    chosen = nlargest(k, scores, key=scores.get)
    chosen.sort(key=lambda s: s.start)  # keep the document's original order
    return [s.text.strip() for s in chosen]


def summarize_text(text: str, ratio: float = 0.2) -> str:
    return " ".join(extractive_sentences(text, ratio))


def divide_into_topics(sentences: list[str], n_topics: int = 3):
    """Cluster sentences with TF-IDF + KMeans.

    Returns (topics, keywords): both dicts keyed by cluster label.
    """
    n_topics = max(1, min(n_topics, len(sentences)))
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10).fit(X)

    terms = vectorizer.get_feature_names_out()
    keywords = {}
    for label in range(n_topics):
        top_idx = kmeans.cluster_centers_[label].argsort()[-3:][::-1]
        keywords[label] = ", ".join(terms[i] for i in top_idx)

    topics: dict[int, list[str]] = {}
    for sentence, label in zip(sentences, kmeans.labels_):
        topics.setdefault(int(label), []).append(sentence)
    return topics, keywords
