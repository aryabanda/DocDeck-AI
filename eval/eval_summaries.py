"""Compare extractive vs LLM-written slides using ROUGE against reference summaries.

Setup: put pairs in a folder, e.g.
  eval/docs/paper1.pdf      eval/docs/paper1.ref.txt   (a human-written summary)
Usage:
  python -m eval.eval_summaries --docs eval/docs --ratio 0.2 --topics 5 [--no-llm] [--skip-pages 1]

Notes
  - Do not let the reference leak into the input. If the reference is a paper's
    abstract, use --skip-pages 1 or delete the abstract from the PDF.
  - ROUGE rewards word overlap, so a good paraphrase can score lower than copied
    sentences. Read a few outputs yourself as well and say so in your README.
"""
import argparse
import glob
import os

from rouge_score import rouge_scorer

from core.extractive import divide_into_topics, extractive_sentences
from core.pdf_utils import extract_pages, full_text
from core.slides import extractive_slide_specs, llm_slide_specs

METRICS = ["rouge1", "rouge2", "rougeL"]


def deck_text(specs: list[dict]) -> str:
    return " ".join(b for s in specs for b in s["bullets"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--ratio", type=float, default=0.2)
    ap.add_argument("--topics", type=int, default=5)
    ap.add_argument("--skip-pages", type=int, default=0)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)
    systems = ["extractive"] + ([] if args.no_llm else ["llm"])
    totals = {s: {m: 0.0 for m in METRICS} for s in systems}
    words = {s: 0 for s in systems}
    count = 0

    for pdf in sorted(glob.glob(os.path.join(args.docs, "*.pdf"))):
        ref_path = pdf[:-4] + ".ref.txt"
        if not os.path.exists(ref_path):
            print(f"skip {pdf}: no {os.path.basename(ref_path)}")
            continue
        reference = open(ref_path, encoding="utf-8").read()
        pages = [p for p in extract_pages(pdf) if p["page"] > args.skip_pages]
        sentences = extractive_sentences(full_text(pages), args.ratio)
        topics, keywords = divide_into_topics(sentences, args.topics)
        outputs = {"extractive": deck_text(extractive_slide_specs(topics, keywords))}
        if not args.no_llm:
            outputs["llm"] = deck_text(llm_slide_specs(topics, keywords))
        for name, text in outputs.items():
            scores = scorer.score(reference, text)
            for m in METRICS:
                totals[name][m] += scores[m].fmeasure
            words[name] += len(text.split())
        count += 1
        print(f"scored {os.path.basename(pdf)}")

    if not count:
        raise SystemExit("No PDF + .ref.txt pairs found.")
    print(f"\nAverages over {count} documents (F1)\n")
    print("| system | ROUGE-1 | ROUGE-2 | ROUGE-L | avg words |")
    print("|---|---|---|---|---|")
    for s in systems:
        print(f"| {s} | " + " | ".join(f"{totals[s][m] / count:.3f}" for m in METRICS) + f" | {words[s] // count} |")


if __name__ == "__main__":
    main()
