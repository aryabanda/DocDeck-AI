"""Measure retrieval quality and compare chunk sizes.

Usage:
  python -m eval.eval_retrieval --pdf docs/my.pdf --questions eval/questions.json \
         --chunk-words 100 180 300 --overlap 40 --k 5 --by-type
  # fair comparison: give every chunk size the same amount of retrieved text
  python -m eval.eval_retrieval ... --context-words 600     # k = round(600 / chunk_words)

questions.json: {"questions": [{"type": "factual", "question": "...", "page": 3,
                                "evidence": "short exact phrase"}]}
  - "evidence" (recommended): a 3-6 word phrase copied from the document that answers it.
    A retrieved chunk counts as a hit if it contains the phrase (case-insensitive).
  - without "evidence", a hit means a chunk from the correct "page" was retrieved.
  - "type" is optional and only used for the --by-type breakdown.
Metrics: hit@1, hit@3, hit@k (share of questions answered within the top-n) and MRR.
"""
import argparse
import json

from core.pdf_utils import extract_pages
from core.rag import Embedder, build_index


def is_relevant(hit: dict, q: dict) -> bool:
    if q.get("evidence"):
        return " ".join(q["evidence"].lower().split()) in hit["text"].lower()
    return hit["page"] == q["page"]


def _rank(index, q: dict, k: int):
    retrieved = index.retrieve(q["question"], k)
    return next((i for i, h in enumerate(retrieved, 1) if is_relevant(h, q)), None)


def _metrics(ranks: list, k: int) -> dict:
    total = len(ranks)
    return {
        "hit@1": sum(1 for r in ranks if r == 1) / total,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / total,
        "hit@k": sum(1 for r in ranks if r) / total,
        "mrr": sum(1 / r for r in ranks if r) / total,
    }


def evaluate(pages, questions, embedder, chunk_words=180, overlap=40, k=5, by_type=False) -> dict:
    index = build_index(pages, embedder, chunk_words, overlap)
    ranks = [_rank(index, q, k) for q in questions]
    result = _metrics(ranks, k)
    result.update(chunks=index.collection.count(), k=k)
    if by_type:
        groups: dict[str, list] = {}
        for q, r in zip(questions, ranks):
            groups.setdefault(q.get("type", "untyped"), []).append(r)
        result["by_type"] = {t: {**_metrics(rs, k), "n": len(rs)} for t, rs in groups.items()}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--chunk-words", type=int, nargs="+", default=[100, 180, 300])
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--context-words", type=int, default=0,
                    help="if set, use k = round(context_words / chunk_words) for each chunk size")
    ap.add_argument("--by-type", action="store_true")
    args = ap.parse_args()

    pages = extract_pages(args.pdf)
    questions = json.load(open(args.questions, encoding="utf-8"))["questions"]
    embedder = Embedder()
    total_words = sum(len(p["text"].split()) for p in pages)
    print(f"{len(questions)} questions, {len(pages)} pages, ~{total_words} words\n")

    rows = []
    for cw in args.chunk_words:
        k = max(1, round(args.context_words / cw)) if args.context_words else args.k
        rows.append((cw, evaluate(pages, questions, embedder, cw, args.overlap, k, args.by_type)))

    print("| chunk_words | k | chunks | hit@1 | hit@3 | hit@k | MRR |")
    print("|---|---|---|---|---|---|---|")
    for cw, r in rows:
        print(f"| {cw} | {r['k']} | {r['chunks']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['hit@k']:.2f} | {r['mrr']:.2f} |")

    if args.by_type:
        for cw, r in rows:
            print(f"\nBy question type (chunk_words={cw}, k={r['k']})\n")
            print("| type | n | hit@1 | hit@3 | hit@k | MRR |")
            print("|---|---|---|---|---|---|")
            for t, m in sorted(r["by_type"].items()):
                print(f"| {t} | {m['n']} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | {m['hit@k']:.2f} | {m['mrr']:.2f} |")


if __name__ == "__main__":
    main()
