"""Streamlit chat interface for the Website Knowledge Assistant.

Run with:  uv run streamlit run app.py
"""

from __future__ import annotations

import os

# The embedding model is downloaded when the index is built, so the app never
# needs the internet. Must be set before sentence-transformers is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import re
import sqlite3

import streamlit as st

from wka.config import AppConfig, ConfigError, load_config
from wka.generation import Answer, Generator, LLMError
from wka.indexing import INDEX_DIR, MANIFEST_FILE

METHOD_LABELS = {"hybrid": "Hybrid (BM25 + dense, RRF)", "bm25": "BM25 (keywords)", "dense": "Dense (embeddings)"}


@st.cache_resource(show_spinner="Loading the index and embedding model ...")
def load_generator(site_id: str, built_at: str) -> Generator:
    # built_at is part of the cache key, so a rebuilt index is loaded again.
    generator = Generator.load(load_config(site_id=site_id))
    generator.retriever.index.embed_query("warm up")   # load the embedding model now, not on the first question
    return generator


def read_manifest(cfg: AppConfig) -> dict | None:
    path = cfg.site_dir / INDEX_DIR / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def last_crawl(cfg: AppConfig) -> str | None:
    path = cfg.site_dir / "wka.sqlite3"
    if not path.exists():
        return None
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT finished_at FROM crawl_runs WHERE status = 'completed'"
                           " ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row else None


def link_citations(text: str, answer: Answer) -> str:
    """Turn [n] into a link to source n, outside code blocks only."""
    def link(match: re.Match) -> str:
        number = int(match.group(1))
        if 1 <= number <= len(answer.sources):
            return f"[[{number}]]({answer.sources[number - 1].chunk.url})"
        return match.group(0)
    parts = text.split("```")
    for i in range(0, len(parts), 2):          # even parts are outside code fences
        parts[i] = re.sub(r"\[(\d+)\]", link, parts[i])
    return "```".join(parts)


def show_answer(answer: Answer) -> None:
    if answer.mode == "closed_book":
        st.divider()
        st.markdown("**Without sources (closed book):** the model's own knowledge, not checked against the documentation.")
    if answer.mode == "rag" and not answer.abstained:
        st.markdown(link_citations(answer.text, answer))
    else:
        st.markdown(answer.text)

    notes = []
    if answer.abstained:
        notes.append("The assistant abstained: the retrieved pages did not contain the answer.")
    if answer.uncited:
        notes.append("This answer cites no sources, so it may not be supported by the documentation.")
    if answer.invalid_citations:
        notes.append(f"The answer cites sources that do not exist: {answer.invalid_citations}.")
    for note in notes:
        (st.info if answer.abstained else st.warning)(note)

    details = f"{answer.seconds}s"
    if answer.method:
        details = f"{METHOD_LABELS[answer.method]} · {details}"
    if answer.prompt_tokens:
        details += f" · {answer.prompt_tokens} prompt + {answer.answer_tokens} answer tokens"
    st.caption(("Closed book (no sources) · " if answer.mode == "closed_book" else "") + details)

    if answer.sources:
        with st.expander(f"Sources ({len(answer.cited)} of {len(answer.sources)} cited)"):
            for number, hit in enumerate(answer.sources, start=1):
                chunk = hit.chunk
                mark = "**cited**" if number in answer.cited else "not cited"
                ranks = (f"BM25 rank {hit.bm25_rank or '-'}, dense rank {hit.dense_rank or '-'}")
                st.markdown(f"**[{number}] [{chunk.title}]({chunk.url})** · {mark} · {ranks}  \n"
                            f"{chunk.heading}")
                preview = chunk.text if len(chunk.text) <= 400 else chunk.text[:400] + " ..."
                st.text(preview)


def main() -> None:
    st.set_page_config(page_title="Website Knowledge Assistant", page_icon="📚", layout="wide")

    try:
        base_cfg = load_config()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        # Only sites that have been indexed can be chosen.
        indexed = [s for s in base_cfg.available_sites
                   if read_manifest(load_config(site_id=s)) is not None]
        if not indexed:
            st.error("No site has been indexed yet. Run:\n\n"
                     "`uv run python -m wka.ingest`\n\n`uv run python -m wka.indexing`")
            st.stop()
        default = indexed.index(base_cfg.site.site_id) if base_cfg.site.site_id in indexed else 0
        site_id = st.selectbox("Website", indexed, index=default,
                               help="The assistant answers from one website at a time.")
        cfg = load_config(site_id=site_id)
        method = st.radio("Retrieval method", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
        top_k = st.slider("Sources per answer", 1, 10, cfg.retrieval.top_k)
        compare = st.toggle("Also answer without sources (closed book)",
                            help="Asks the same model with no documentation, to compare.")
        if st.button("Clear chat"):
            st.session_state.pop(f"messages_{site_id}", None)
            st.rerun()

        manifest = read_manifest(cfg)
        st.divider()
        st.subheader("Index")
        st.markdown(f"**{cfg.site.name}**  \n{cfg.site.base_url}")
        st.markdown(f"{manifest['pages']:,} pages · {manifest['chunks']:,} chunks  \n"
                    f"Model: `{manifest['model']}`  \n"
                    f"Last crawl: {(last_crawl(cfg) or 'unknown')[:16].replace('T', ' ')} UTC  \n"
                    f"Index built: {manifest['built_at'][:16].replace('T', ' ')} UTC")
        st.caption("To update the content, run  \n`uv run python -m wka.refresh`  \nthen reload this page.")

    st.title("Website Knowledge Assistant")
    st.caption(f"Answers come only from **{cfg.site.name}**, with numbered citations. "
               "Each question is answered on its own, without memory of earlier questions.")

    try:
        generator = load_generator(site_id, manifest["built_at"])
    except Exception as exc:
        st.error(f"Could not load the index: {exc}")
        st.stop()
    if generator.retriever.index.stale:
        st.warning("Pages changed after this index was built. Run `uv run python -m wka.indexing`.")

    key = f"messages_{site_id}"
    messages = st.session_state.setdefault(key, [])
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                for answer in message["answers"]:
                    show_answer(answer)

    question = st.chat_input(f"Ask about {cfg.site.name} ...")
    if not question:
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answers: list[Answer] = []
        try:
            for closed_book in ([False, True] if compare else [False]):
                placeholder = st.empty()
                streamed: list[str] = []

                def on_token(token: str) -> None:
                    streamed.append(token)
                    placeholder.markdown("".join(streamed) + " ▌")

                with st.spinner("Searching the documentation ..." if not closed_book else
                                "Answering without sources ..."):
                    if closed_book:
                        answer = generator.answer_closed_book(question, on_token=on_token)
                    else:
                        answer = generator.answer(question, method=method, k=top_k, on_token=on_token)
                placeholder.empty()
                show_answer(answer)
                answers.append(answer)
        except LLMError as exc:
            st.error(str(exc))
        if answers:
            messages.append({"role": "assistant", "answers": answers})


main()