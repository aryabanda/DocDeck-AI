"""Does the system refuse questions the document can't answer, without refusing ones it can?

Usage:
  python -m eval.eval_abstention --pdf docs/my.pdf --questions eval/questions.json \
         --out-of-doc eval/out_of_doc.json [--chunk-words 180] [--k 4] [--llm]

Step 1 (no API key needed): compare the top-1 similarity score of in-document questions
with out-of-document questions and suggest a cutoff (RAG_MIN_SCORE).
Step 2 (--llm): run the full pipeline and count refusals, i.e. how often the model
replies with the not-found message.
"""
import argparse
import json

from core.pdf_utils import extract_pages
from core.rag import NOT_FOUND, Embedder, answer, build_index


def top_scores(index, questions: list[dict]) -> list[float]:
    scores = []
    for q in questions:
        hits = index.retrieve(q["question"], 1)
        scores.append(hits[0]["score"] if hits else 0.0)
    return scores


def best_threshold(in_scores: list[float], out_scores: list[float]):
    """Cutoff t maximising balanced accuracy, where score >= t means 'answerable'.
    Returns (t, share of in-doc questions kept, share of out-of-doc questions blocked)."""
    best = None
    for t in sorted(set(in_scores + out_scores)):
        kept = sum(s >= t for s in in_scores) / len(in_scores)
        blocked = sum(s < t for s in out_scores) / len(out_scores)
        score = (kept + blocked) / 2
        if best is None or score > best[0] + 1e-9:
            best = (score, t, kept, blocked)
    return best[1], best[2], best[3]


def is_refusal(text: str) -> bool:
    return NOT_FOUND.lower() in text.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-of-doc", required=True)
    ap.add_argument("--chunk-words", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--llm", action="store_true", help="also run the LLM and count refusals")
    args = ap.parse_args()

    in_q = json.load(open(args.questions, encoding="utf-8"))["questions"]
    out_q = json.load(open(args.out_of_doc, encoding="utf-8"))["questions"]
    index = build_index(extract_pages(args.pdf), Embedder(), args.chunk_words, args.overlap)

    in_s, out_s = top_scores(index, in_q), top_scores(index, out_q)
    print("\nTop-1 similarity per question (lowest first)\n")
    print("| group | score | question |\n|---|---|---|")
    rows = [("in-doc", s, q["question"]) for s, q in zip(in_s, in_q)]
    rows += [("OUT", s, q["question"]) for s, q in zip(out_s, out_q)]
    for g, s, text in sorted(rows, key=lambda r: r[1]):
        print(f"| {g} | {s:.3f} | {text[:70]} |")

    print(f"\nin-doc : min {min(in_s):.3f}  mean {sum(in_s)/len(in_s):.3f}")
    print(f"out-doc: max {max(out_s):.3f}  mean {sum(out_s)/len(out_s):.3f}")
    t, kept, blocked = best_threshold(in_s, out_s)
    print(f"\nSuggested RAG_MIN_SCORE = {t:.3f}  (keeps {kept:.0%} of in-doc, blocks {blocked:.0%} of out-of-doc)")
    if min(in_s) > max(out_s):
        print("The two groups do not overlap: a cutoff between them separates them perfectly.")
    else:
        print("The groups overlap, so no cutoff is perfect. Prefer a lower cutoff and let the LLM refuse the rest.")

    if args.llm:
        refused_out = sum(is_refusal(answer(q["question"], index, args.k)["answer"]) for q in out_q)
        refused_in = sum(is_refusal(answer(q["question"], index, args.k)["answer"]) for q in in_q)
        print(f"\nLLM refusals (no cutoff): out-of-doc {refused_out}/{len(out_q)} (higher is better), "
              f"in-doc {refused_in}/{len(in_q)} (lower is better)")


if __name__ == "__main__":
    main()
