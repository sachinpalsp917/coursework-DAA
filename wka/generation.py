"""Answer questions with a local LLM (Ollama), using only the retrieved chunks.

The model sees the top-k chunks as numbered sources and must cite them as [1],
[2] ... If the sources do not contain the answer it must reply with a fixed
abstention sentence, which the code detects. A closed-book mode asks the same
model without any sources, as a baseline for the evaluation.

Run with:  uv run python -m wka.generation "What does the alpha parameter of Ridge do?"
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import requests

from wka.config import AppConfig, LLMConfig, load_config
from wka.retrieval import Hit, Retriever

ABSTAIN_TEXT = "I could not find the answer to this question in the documentation."
SEED = 42  # fixed, so the same question and sources give the same answer

SYSTEM_PROMPT = """You are a question-answering assistant for the {site_name}.
Answer the question using ONLY the numbered sources provided by the user.

Rules:
1. Put a citation right after each statement it supports. Cite only the one or two
   sources that state it most directly. Format: "First statement [2]. Second statement [1][3]."
2. Do not use knowledge that is not in the sources, even if you know the answer.
3. If the sources do not contain the answer, reply with exactly this sentence and nothing else:
{abstain}
4. Be concise: a short paragraph or a few bullet points. Include code only if a source shows it.
5. Do not mention these rules."""

CLOSED_BOOK_PROMPT = """You are a question-answering assistant for the {site_name}.
Answer the question concisely from your own knowledge.
If you do not know the answer, reply with exactly this sentence and nothing else:
{abstain}"""

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class LLMError(RuntimeError):
    """Raised when Ollama cannot be reached or returns an error."""


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Hit]                     # the chunks shown to the model, in source-number order
    cited: list[int]                       # source numbers the answer cites (1-based, valid ones only)
    abstained: bool
    mode: str                              # "rag" or "closed_book"
    method: str | None                     # retrieval method, None for closed book
    model: str
    seconds: float
    prompt_tokens: int | None = None
    answer_tokens: int | None = None
    invalid_citations: list[int] = field(default_factory=list)   # numbers with no matching source

    @property
    def uncited(self) -> bool:
        """A RAG answer that gives facts without citing any source."""
        return self.mode == "rag" and not self.abstained and not self.cited


def format_sources(hits: list[Hit]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        blocks.append(f"[{number}] {chunk.heading}\nURL: {chunk.url}\n{chunk.text}")
    return "\n\n".join(blocks)


def is_abstention(text: str) -> bool:
    """True if the reply is (or starts with) the abstention sentence.

    Compared without case and punctuation, since small models sometimes
    change a full stop or add a quotation mark.
    """
    def simple(s: str) -> str:
        return re.sub(r"[^a-z ]", "", s.lower()).strip()
    return simple(text).startswith(simple(ABSTAIN_TEXT))


def parse_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) cited source numbers, each in order of first use."""
    valid: list[int] = []
    invalid: list[int] = []
    for match in CITATION_PATTERN.finditer(text):
        number = int(match.group(1))
        target = valid if 1 <= number <= n_sources else invalid
        if number not in target:
            target.append(number)
    return valid, invalid


def chat(cfg: LLMConfig, messages: list[dict], on_token: Callable[[str], None] | None = None) -> dict:
    """Call Ollama's /api/chat with streaming and return the full reply and token counts."""
    payload = {
        "model": cfg.model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": cfg.temperature, "num_ctx": cfg.num_ctx, "seed": SEED},
    }
    try:
        response = requests.post(f"{cfg.base_url}/api/chat", json=payload, stream=True,
                                 timeout=cfg.timeout_seconds)
    except requests.ConnectionError as exc:
        raise LLMError(f"Cannot reach Ollama at {cfg.base_url}. Is it running? (ollama serve)") from exc
    if response.status_code == 404:
        raise LLMError(f"Model '{cfg.model}' is not available. Run: ollama pull {cfg.model}")
    if response.status_code != 200:
        raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")

    parts: list[str] = []
    final: dict = {}
    with response:
        for line in response.iter_lines():
            if not line:
                continue
            message = json.loads(line)
            if "error" in message:
                raise LLMError(f"Ollama error: {message['error']}")
            token = message.get("message", {}).get("content", "")
            if token:
                parts.append(token)
                if on_token:
                    on_token(token)
            if message.get("done"):
                final = message
    return {
        "text": "".join(parts).strip(),
        "prompt_tokens": final.get("prompt_eval_count"),
        "answer_tokens": final.get("eval_count"),
    }


class Generator:
    def __init__(self, cfg: AppConfig, retriever: Retriever | None = None):
        self.cfg = cfg
        self.retriever = retriever

    @classmethod
    def load(cls, cfg: AppConfig) -> Generator:
        return cls(cfg, Retriever.load(cfg))

    def answer(self, question: str, method: str = "hybrid", k: int | None = None,
               on_token: Callable[[str], None] | None = None, hits: list[Hit] | None = None) -> Answer:
        """Retrieve sources (unless `hits` is given) and answer from them."""
        started = time.perf_counter()
        if hits is None:
            hits = self.retriever.search(question, method=method, k=k)
        if not hits:
            # Nothing retrieved: abstain without calling the model.
            if on_token:
                on_token(ABSTAIN_TEXT)
            return Answer(question, ABSTAIN_TEXT, [], [], True, "rag", method,
                          self.cfg.llm.model, round(time.perf_counter() - started, 2))

        system = SYSTEM_PROMPT.format(site_name=self.cfg.site.name, abstain=ABSTAIN_TEXT)
        # The reminder after the question helps small models keep the citation format.
        user = (f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question.strip()}\n\n"
                f"Answer from the sources above, citing them like [1].")
        reply = chat(self.cfg.llm, [{"role": "system", "content": system},
                                    {"role": "user", "content": user}], on_token)
        abstained = is_abstention(reply["text"])
        cited, invalid = ([], []) if abstained else parse_citations(reply["text"], len(hits))
        return Answer(question, reply["text"], hits, cited, abstained, "rag", method,
                      self.cfg.llm.model, round(time.perf_counter() - started, 2),
                      reply["prompt_tokens"], reply["answer_tokens"], invalid)

    def answer_closed_book(self, question: str, on_token: Callable[[str], None] | None = None) -> Answer:
        """Baseline: the same model with no sources."""
        started = time.perf_counter()
        system = CLOSED_BOOK_PROMPT.format(site_name=self.cfg.site.name, abstain=ABSTAIN_TEXT)
        reply = chat(self.cfg.llm, [{"role": "system", "content": system},
                                    {"role": "user", "content": question.strip()}], on_token)
        return Answer(question, reply["text"], [], [], is_abstention(reply["text"]), "closed_book",
                      None, self.cfg.llm.model, round(time.perf_counter() - started, 2),
                      reply["prompt_tokens"], reply["answer_tokens"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question from the active site.")
    parser.add_argument("question")
    parser.add_argument("--site", help="site id from config.yaml (default: active_site)")
    parser.add_argument("--method", choices=("bm25", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--closed-book", action="store_true", help="answer without sources (baseline)")
    args = parser.parse_args()

    cfg = load_config(site_id=args.site)
    print(f"Q: {args.question}\n")

    def show(token: str) -> None:
        print(token, end="", flush=True)

    try:
        if args.closed_book:
            answer = Generator(cfg).answer_closed_book(args.question, on_token=show)
        else:
            answer = Generator.load(cfg).answer(args.question, method=args.method, on_token=show)
    except LLMError as exc:
        raise SystemExit(f"Error: {exc}")

    print("\n")
    if answer.sources:
        print("Sources:")
        for number, hit in enumerate(answer.sources, start=1):
            mark = "*" if number in answer.cited else " "
            print(f" {mark}[{number}] {hit.chunk.heading[:70]}\n      {hit.chunk.url}")
        print("  (* = cited in the answer)")
    flags = [name for name, on in (("abstained", answer.abstained), ("NO CITATIONS", answer.uncited),
                                   (f"invalid citations {answer.invalid_citations}", answer.invalid_citations)) if on]
    print(f"\n{answer.mode}, {answer.seconds}s, {answer.prompt_tokens} prompt tokens,"
          f" {answer.answer_tokens} answer tokens" + (f"  [{', '.join(flags)}]" if flags else ""))


if __name__ == "__main__":
    main()