"""Evaluate retrieval and answers against eval/questions.csv.

Retrieval (answerable questions only). The ranking is the list of chunks the
model would be given; a chunk counts as relevant when its page is listed in
relevant_pages, and each relevant page is credited once:
    Hit@k     1 if a chunk from a relevant page is in the top k
    Recall@k  share of the relevant pages that appear in the top k chunks
    MRR       1 / rank of the first relevant chunk (0 if not in the top 10)
    nDCG@k    rank-weighted gain of the top k chunks

Answers (with --answers): every question is answered with RAG and closed
book. Abstention and citations are scored automatically; correctness,
faithfulness and citation accuracy are scored by hand in answers.csv using
the rubric below, then summarised with --summarise.

Run with:  uv run python -m wka.evaluate [--answers]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from wka.config import PROJECT_ROOT, load_config
from wka.generation import Generator, LLMError
from wka.retrieval import METHODS, Retriever

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.csv"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
MRR_DEPTH = 10

RUBRIC = {
    "correctness": "2 = matches expected_answer, 1 = partly right or incomplete, 0 = wrong or missing",
    "faithfulness": "2 = every claim is supported by the shown sources, 1 = minor unsupported detail, "
                    "0 = important claim not in the sources (closed book: leave blank)",
    "citation_accuracy": "2 = every citation points to a source that says it, 1 = some misattributed, "
                         "0 = mostly wrong or no citations (closed book: leave blank)",
}


def load_questions(path: Path = QUESTIONS_FILE) -> pd.DataFrame:
    questions = pd.read_csv(path, keep_default_na=False)
    missing = {"id", "category", "answerable", "question", "expected_answer", "relevant_pages"} - set(questions.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    if not questions["id"].is_unique:
        raise ValueError(f"{path} has duplicate ids.")
    return questions


def relevant_urls(row: pd.Series, base_url: str) -> set[str]:
    return {base_url + page.strip() for page in row["relevant_pages"].split("|") if page.strip()}


def retrieval_metrics(ranked_urls: list[str], relevant: set[str], k: int) -> dict:
    """Metrics for one ranked list of chunk page URLs (the same page may appear several times)."""
    found: list[str] = []
    dcg = 0.0
    for rank, url in enumerate(ranked_urls[:k], start=1):
        if url in relevant and url not in found:      # credit each relevant page once
            found.append(url)
            dcg += 1 / math.log2(rank + 1)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    first = next((rank for rank, url in enumerate(ranked_urls[:MRR_DEPTH], start=1) if url in relevant), None)
    return {
        f"hit@{k}": int(bool(found)),
        f"recall@{k}": len(found) / len(relevant),
        "mrr": 1 / first if first else 0.0,
        f"ndcg@{k}": dcg / ideal if ideal else 0.0,
        "first_rank": first,
    }


def evaluate_retrieval(retriever: Retriever, questions: pd.DataFrame, base_url: str, k: int) -> pd.DataFrame:
    rows = []
    for _, q in questions[questions["answerable"] == "yes"].iterrows():
        relevant = relevant_urls(q, base_url)
        results = retriever.search_all(q["question"], k=MRR_DEPTH)
        for method in METHODS:
            urls = [hit.chunk.url for hit in results[method]]
            rows.append({"id": q["id"], "category": q["category"], "method": method,
                         **retrieval_metrics(urls, relevant, k),
                         "top_pages": " | ".join(u.removeprefix(base_url) for u in urls[:k])})
    return pd.DataFrame(rows)


def summarise_retrieval(per_question: pd.DataFrame, k: int) -> pd.DataFrame:
    metrics = [f"hit@{k}", f"recall@{k}", "mrr", f"ndcg@{k}"]
    table = (pd.concat([per_question.assign(category="all"), per_question])
             .groupby(["method", "category"], sort=False)
             .agg(questions=("id", "count"), **{m: (m, "mean") for m in metrics})
             .reset_index())
    return table.round(3)


def evaluate_answers(generator: Generator, questions: pd.DataFrame, base_url: str, method: str) -> pd.DataFrame:
    rows = []
    for _, q in questions.iterrows():
        relevant = relevant_urls(q, base_url)
        for mode in ("rag", "closed_book"):
            print(f"  {q['id']:4} {mode:11}", end=" ", flush=True)
            answer = (generator.answer(q["question"], method=method) if mode == "rag"
                      else generator.answer_closed_book(q["question"]))
            cited_urls = [answer.sources[n - 1].chunk.url for n in answer.cited]
            should_abstain = q["answerable"] == "no"
            rows.append({
                "id": q["id"], "category": q["category"], "answerable": q["answerable"], "mode": mode,
                "question": q["question"], "expected_answer": q["expected_answer"], "answer": answer.text,
                "abstained": answer.abstained,
                "abstention_correct": answer.abstained == should_abstain,
                "n_sources": len(answer.sources), "n_cited": len(answer.cited),
                "invalid_citations": len(answer.invalid_citations),
                "uncited": answer.uncited,
                # Share of cited sources that come from a page listed as relevant (a rough precision proxy).
                "cited_relevant_share": (sum(u in relevant for u in cited_urls) / len(cited_urls)
                                         if cited_urls and relevant else None),
                "sources": "\n".join(f"[{n}] {h.chunk.url.removeprefix(base_url)} :: {h.chunk.heading}"
                                     for n, h in enumerate(answer.sources, start=1)),
                "seconds": answer.seconds, "prompt_tokens": answer.prompt_tokens,
                "answer_tokens": answer.answer_tokens,
                "correctness": "", "faithfulness": "", "citation_accuracy": "", "grader_notes": "",
            })
            print(f"{answer.seconds:5.1f}s {'abstained' if answer.abstained else ''}")
    return pd.DataFrame(rows)


def summarise_answers(answers: pd.DataFrame) -> pd.DataFrame:
    """Automatic answer metrics per mode, plus rubric means where grades are filled in."""
    rows = []
    for mode, group in answers.groupby("mode", sort=False):
        answerable = group[group["answerable"] == "yes"]
        unanswerable = group[group["answerable"] == "no"]
        row = {
            "mode": mode,
            "questions": len(group),
            "abstained_when_should (unanswerable)": unanswerable["abstained"].mean(),
            "false_abstention (answerable)": answerable["abstained"].mean(),
            "uncited_answers": group["uncited"].mean() if mode == "rag" else None,
            "invalid_citations": group["invalid_citations"].sum() if mode == "rag" else None,
            "cited_relevant_share": pd.to_numeric(group["cited_relevant_share"]).mean() if mode == "rag" else None,
            "mean_seconds": group["seconds"].mean(),
        }
        for criterion in RUBRIC:
            scores = pd.to_numeric(answerable[criterion], errors="coerce").dropna()
            row[f"{criterion} (0-2, n)"] = f"{scores.mean():.2f} (n={len(scores)})" if len(scores) else ""
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers.")
    parser.add_argument("--site", help="site id from config.yaml (default: active_site)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--answers", action="store_true", help="also generate RAG and closed-book answers (slow)")
    parser.add_argument("--method", choices=METHODS, default="hybrid", help="retrieval method for RAG answers")
    parser.add_argument("--summarise", type=Path, metavar="ANSWERS_CSV",
                        help="summarise a hand-graded answers.csv instead of running anything")
    args = parser.parse_args()

    if args.summarise:
        print(summarise_answers(pd.read_csv(args.summarise)).to_string(index=False))
        return

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = load_config(site_id=args.site)
    k = cfg.retrieval.top_k
    questions = load_questions(args.questions)
    out_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)

    generator = Generator.load(cfg)
    if generator.retriever.index.stale:
        print("Warning: the index is older than the stored pages. Run wka.indexing first.")
    # relevant_pages are paths below the site's prefix, e.g. https://scikit-learn.org/stable/
    base_url = f"{urlparse(cfg.site.base_url).scheme}://{cfg.site.allowed_domain}{cfg.site.allowed_path_prefix}"

    print(f"Retrieval: {int((questions['answerable'] == 'yes').sum())} answerable questions, k={k}")
    per_question = evaluate_retrieval(generator.retriever, questions, base_url, k)
    summary = summarise_retrieval(per_question, k)
    per_question.to_csv(out_dir / "retrieval_per_question.csv", index=False)
    summary.to_csv(out_dir / "retrieval_summary.csv", index=False)
    print(summary.to_string(index=False))

    if args.answers:
        print(f"\nAnswers: {len(questions)} questions x RAG ({args.method}) and closed book")
        try:
            answers = evaluate_answers(generator, questions, base_url, args.method)
        except LLMError as exc:
            raise SystemExit(f"Error: {exc}")
        answers.to_csv(out_dir / "answers.csv", index=False)
        print(summarise_answers(answers).to_string(index=False))
        print("\nGrade the empty rubric columns in answers.csv, then run:\n"
              f"  uv run python -m wka.evaluate --summarise {out_dir / 'answers.csv'}")

    (out_dir / "run_info.json").write_text(json.dumps({
        "site": cfg.site.site_id, "questions_file": str(args.questions), "k": k,
        "rrf_k": cfg.retrieval.rrf_k, "candidate_k": cfg.retrieval.candidate_k,
        "index": generator.retriever.index.manifest, "llm": cfg.llm.model,
        "temperature": cfg.llm.temperature, "rag_method": args.method if args.answers else None,
        "rubric": RUBRIC,
    }, indent=2), encoding="utf-8")
    print(f"\nResults saved in {out_dir}")


if __name__ == "__main__":
    main()