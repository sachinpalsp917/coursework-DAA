"""Grade the answers of an evaluation run by hand, one at a time.

Shows each answerable question with its expected answer, the model's answer
and (for RAG) the text of the sources it was given, then asks for 0-2 scores
using the rubric in wka.evaluate. Scores are written to answers.csv after
every answer, so you can stop at any time and continue later.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import pandas as pd

from wka.config import load_config
from wka.evaluate import RUBRIC
from wka.generation import parse_citations

RAG_CRITERIA = ("correctness", "faithfulness", "citation_accuracy")
CLOSED_BOOK_CRITERIA = ("correctness",)
UNCITED_PREVIEW = 200   # characters shown for sources the answer does not cite


class Quit(Exception):
    pass


def save(answers: pd.DataFrame, path: Path) -> None:
    # Write a temporary file and rename it, so an interrupted save cannot corrupt the CSV.
    tmp = path.with_suffix(".tmp")
    answers.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_sources(run_dir: Path, answers: pd.DataFrame) -> dict[str, list]:
    """Retrieve the RAG sources again to show their text, and check they match the saved run."""
    from wka.retrieval import Retriever

    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    retriever = Retriever.load(load_config(site_id=info["site"]))
    if retriever.index.manifest["built_at"] != info["index"]["built_at"]:
        print("Warning: the index was rebuilt after this run, so sources may differ.\n")
    sources = {}
    for _, row in answers[(answers["mode"] == "rag") & (answers["answerable"] == "yes")].iterrows():
        hits = retriever.search(row["question"], method=info["rag_method"], k=info["k"])
        saved = [line.split("] ", 1)[1].split(" :: ")[0] for line in row["sources"].splitlines()]
        if [h.chunk.url.endswith(path) for h, path in zip(hits, saved)] != [True] * len(saved):
            print(f"Warning: sources for {row['id']} differ from the saved run.")
        sources[row["id"]] = hits
    return sources


def show(row: pd.Series, hits: list | None, position: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{position}  {row['id']}  ({row['category']}, {row['mode'].replace('_', ' ')})\n{line}")
    print(f"QUESTION:  {row['question']}\n")
    print("EXPECTED:  " + textwrap.fill(row["expected_answer"], 78, subsequent_indent="           "))
    print(f"\nANSWER:\n{row['answer']}")
    if hits:
        cited, _ = parse_citations(row["answer"], len(hits))
        print("\nSOURCES (full text for cited sources):")
        for number, hit in enumerate(hits, start=1):
            mark = "CITED" if number in cited else "not cited"
            print(f"\n  [{number}] {mark} · {hit.chunk.heading}\n      {hit.chunk.url}")
            text = hit.chunk.text if number in cited else hit.chunk.text[:UNCITED_PREVIEW] + " ..."
            print(textwrap.indent(text, "      "))


def ask_score(criterion: str, current: str) -> str:
    hint = f" [now {current}]" if current else ""
    while True:
        reply = input(f"{criterion}{hint} (0/1/2, Enter = keep/skip, ? = rubric, q = quit): ").strip().lower()
        if reply == "q":
            raise Quit
        if reply == "?":
            print("  " + RUBRIC[criterion])
        elif reply == "":
            return current
        elif reply in ("0", "1", "2"):
            return reply
        else:
            print("  Please type 0, 1 or 2.")


def is_graded(row: pd.Series) -> bool:
    criteria = RAG_CRITERIA if row["mode"] == "rag" else CLOSED_BOOK_CRITERIA
    return all(str(row[c]).strip() != "" for c in criteria)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade evaluation answers by hand.")
    parser.add_argument("answers_csv", type=Path, help="answers.csv from an evaluation run")
    parser.add_argument("--regrade", action="store_true", help="also show answers that already have scores")
    parser.add_argument("--only", help="grade only these ids, comma separated (e.g. V5,P9)")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    path = args.answers_csv
    answers = pd.read_csv(path, keep_default_na=False, dtype=str)
    todo = answers[answers["answerable"] == "yes"]
    if args.only:
        todo = todo[todo["id"].isin([i.strip() for i in args.only.split(",")])]
    if not args.regrade:
        todo = todo[~todo.apply(is_graded, axis=1)]
    answerable = answers[answers["answerable"] == "yes"]
    done = int(answerable.apply(is_graded, axis=1).sum())
    print(f"{done} of {len(answerable)} answers graded; {len(todo)} to show now. Scores are saved after each answer.")
    if todo.empty:
        return

    print("Loading sources ...")
    sources = load_sources(path.parent, answers)

    graded = 0
    try:
        for n, (index, row) in enumerate(todo.iterrows(), start=1):
            show(row, sources.get(row["id"]) if row["mode"] == "rag" else None, f"[{n}/{len(todo)}]")
            print()
            criteria = RAG_CRITERIA if row["mode"] == "rag" else CLOSED_BOOK_CRITERIA
            for criterion in criteria:
                answers.at[index, criterion] = ask_score(criterion, answers.at[index, criterion])
            note = input(f"note [{answers.at[index, 'grader_notes'] or 'none'}] (Enter = keep): ").strip()
            if note:
                answers.at[index, "grader_notes"] = note
            save(answers, path)
            graded += is_graded(answers.loc[index])
    except (Quit, KeyboardInterrupt, EOFError):
        print("\nStopped.")
    save(answers, path)   # keeps any scores typed for an answer that was not finished
    remaining = int((~answers[answers["answerable"] == "yes"].apply(is_graded, axis=1)).sum())
    print(f"\nSaved {path}. Graded this session: {graded}. Still ungraded: {remaining}.")
    if remaining == 0:
        print(f"All done. Summarise with:\n  uv run python -m wka.evaluate --summarise {path}")


if __name__ == "__main__":
    main()