# Website Knowledge Assistant

A retrieval-augmented (RAG) chatbot that answers questions from **one documentation website at a time**, with numbered citations to the pages it used, and refuses when the answer is not in the documentation.

The configured site is the [scikit-learn documentation](https://scikit-learn.org/stable/) (version 1.9.1). A second site (PyTorch) is defined in the config to show how the assistant is switched to another website.

Coursework for 7CS134 Data Mining and Analytics (Option B: RAG chatbot).

## How it works

```
crawl ─► extract ─► store ─► chunk ─► index ─► retrieve ─► generate ─► Streamlit UI
 (polite)  (by section) (SQLite)  (headings) (FAISS+BM25) (RRF hybrid) (Ollama, cited)
                                                                  ▲
                                   refresh: re-crawl, detect changes, re-index changed pages
```

| Stage | Module | What it does |
|---|---|---|
| Crawl | `wka/crawler.py` | Breadth-first crawl inside one domain and path prefix. Follows robots.txt and its Crawl-delay, waits between requests, re-checks every redirect against the scope, retries with backoff, limits response size, removes duplicate pages via canonical URLs. |
| Extract | `wka/extract.py` | Takes the main content area of each page and splits it into sections with their heading trail. Keeps code blocks as fenced code and drops page furniture (menus, "[source]" links, download boxes). |
| Store | `wka/store.py`, `wka/ingest.py` | SQLite database per site: pages (versioned by a SHA-256 hash of the extracted text), crawl runs and a fetch log. Raw HTML is kept as gzipped snapshots, so pages can be re-extracted offline (`--reextract`). |
| Chunk | `wka/chunking.py` | Heading-aware chunks of at most 1,200 characters with 150 characters of overlap. Small sections are merged; code blocks are never left unbalanced. Only pages whose version changed are re-chunked. |
| Index | `wka/indexing.py` | `BAAI/bge-small-en-v1.5` embeddings in a FAISS inner-product index, plus BM25. Embeddings are cached by text hash, so a refresh only embeds new or changed chunks. |
| Retrieve | `wka/retrieval.py` | BM25, dense, or hybrid search using Reciprocal Rank Fusion (k = 60). |
| Generate | `wka/generation.py` | `llama3.1:8b` via Ollama at temperature 0 with a fixed seed. Answers cite sources as `[1]`, `[2]`… and use a fixed sentence to abstain. A closed-book mode answers without sources, as a baseline. |
| Refresh | `wka/refresh.py` | Re-crawls the site, reports new/changed/unchanged pages, marks pages removed only when the server confirms it (404/410 or a redirect), then updates the index incrementally. |
| UI | `app.py` | Streamlit chat with clickable citations, a sources panel (with BM25 and dense ranks), a retrieval-method switch and an optional closed-book comparison. |
| Evaluate | `wka/evaluate.py`, `wka/grade.py` | Retrieval metrics (Hit@5, Recall@5, MRR, nDCG@5) per method; RAG vs closed-book answers scored for abstention, citations and a 0–2 rubric. |

## Requirements

- macOS or Linux, Python 3.11, and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) with the `llama3.1:8b` model (about 4.9 GB)
- About 6 GB of free disk space (LLM, embedding model, data) and 8 GB of RAM or more
- Internet access for the first crawl and to download the models; after that everything runs offline

## Setup

```bash
uv sync
ollama pull llama3.1:8b
```

`uv sync` creates `.venv` and installs the exact package versions in `uv.lock`.
Make sure Ollama is running (`ollama serve`, or the Ollama app) before generating answers.

## Building the knowledge base

The crawled data is not in the repository (`data/` is git-ignored). Rebuild it with:

```bash
uv run python -m wka.ingest --quiet      # crawl and store ~1,060 pages, about 18 minutes
uv run python -m wka.indexing            # chunk and embed, about 2 minutes on the first run
```

The first `wka.indexing` run downloads the embedding model (about 130 MB) from Hugging Face. Later commands can run offline by prefixing them with `HF_HUB_OFFLINE=1`; the Streamlit app does this itself.

Useful checks:

```bash
uv run python -m wka.config                       # show the active site and its scope
uv run python -m wka.store                        # pages stored and recent crawl runs
uv run python -m wka.chunking --show https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
uv run python -m wka.ingest --limit 20            # small test crawl
```

## Using the assistant

**Chat interface**

```bash
uv run streamlit run app.py
```

Open http://localhost:8501. The server listens on this computer only and usage statistics are turned off (`.streamlit/config.toml`).

**Command line**

```bash
uv run python -m wka.retrieval "max_iter default Ridge"                  # compare BM25, dense and hybrid
uv run python -m wka.generation "What does the alpha parameter of Ridge do?"
uv run python -m wka.generation "What does StandardScaler do?" --closed-book
```

## Keeping the content up to date

```bash
uv run python -m wka.refresh
```

Re-crawls the whole site (about 18 minutes), lists new, changed and removed pages, and updates the index. Only changed text is re-chunked and re-embedded, so a refresh with no changes embeds nothing. Reload the Streamlit page afterwards.

## Evaluation

The question set `eval/questions.csv` has 32 questions, each checked against the stored documentation:

| Category | Questions | Tests |
|---|---|---|
| version_specific | 8 | Features added in scikit-learn 1.5–1.9, after the LLM's training data |
| api_detail | 7 | Exact defaults and numbers |
| paraphrase | 10 | Questions that describe a need without naming the API |
| multi_hop | 2 | Answers that combine or compare two facts |
| unanswerable | 2 | Nothing in the documentation answers them; the assistant should abstain |
| false_premise | 3 | Based on something untrue; the assistant should abstain or correct it |

```bash
uv run python -m wka.evaluate                    # retrieval metrics only, about 10 seconds
uv run python -m wka.evaluate --answers          # also RAG and closed-book answers, about 8 minutes
uv run python -m wka.grade eval/results/<run>/answers.csv
uv run python -m wka.evaluate --summarise eval/results/<run>/answers.csv
```

Each run is saved in its own folder under `eval/results/` with the per-question CSVs, summaries and a `run_info.json` recording the configuration, index and model. Temperature 0 and a fixed seed make runs reproducible: two separate runs produced identical answers for all 64 question/mode pairs.

### Results (run `eval/results/20261104-211313`)

Retrieval, 27 answerable questions, top 5 chunks:

| Method | Hit@5 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| BM25 | 0.778 | 0.608 | 0.651 | 0.542 |
| Dense | 0.889 | 0.713 | 0.773 | 0.650 |
| Hybrid (RRF) | 0.889 | 0.719 | 0.725 | 0.644 |

Hybrid is best on version-specific and API-detail questions; dense is best on paraphrased ones.

Answers (rubric scores 0–2 on the 27 answerable questions):

| | RAG (hybrid) | Closed book |
|---|---|---|
| Correctness | 1.70 | 0.41 |
| Faithfulness to sources | 1.59 | — |
| Citation accuracy | 1.67 | — |
| Abstained on the 5 unanswerable / false-premise questions | 5 of 5 | 2 of 5 |
| Mean time per answer | 11.0 s | 3.9 s |

The answers were graded against the rubric by an AI assistant and reviewed by the author; the reason for each score is in the `grader_notes` column.

## Configuration

Everything is set in `config/config.yaml`: the active site and the site definitions (start URL, domain, path prefix, page limit, excluded URL patterns), crawler politeness, chunk sizes, embedding model, retrieval parameters and the LLM.

To answer from another website, add a site entry and either change `active_site` or pass `--site`:

```bash
uv run python -m wka.ingest --site pytorch --limit 50
uv run python -m wka.indexing --site pytorch
```

Each site keeps its own data in `data/<site_id>/`, and the Streamlit app lists every site that has an index.

## Project structure

```
project/
├── app.py                  Streamlit chat interface
├── config/config.yaml      sites and all settings
├── wka/                    pipeline modules (see "How it works")
├── eval/
│   ├── questions.csv       evaluation questions with expected answers and relevant pages
│   └── results/<run>/      evaluation outputs
├── data/<site_id>/         created by the crawl (git-ignored)
│   ├── wka.sqlite3         pages, chunks, embedding cache, crawl runs, fetch log
│   ├── raw/                gzipped HTML snapshots
│   └── index/              dense.faiss and manifest.json
├── pyproject.toml, uv.lock
└── .streamlit/config.toml  local-only server, no usage statistics
```

## Responsible crawling and privacy

- Only public documentation pages inside the configured domain and path are fetched; robots.txt is followed, including Crawl-delay.
- At least 1 second between requests, a descriptive User-Agent, limited retries, and a response size limit.
- Pages are only marked as removed after the server confirms it; a temporary outage never deletes content.
- All models run locally (Ollama and sentence-transformers). Questions and answers never leave the computer.

## Known limitations

- Each question is answered on its own; there is no conversation memory.
- When retrieval misses the right page, the answer is wrong (questions V6 and P9 in the evaluation).
- The 8B model sometimes cites more sources than it uses, and writes example code that goes beyond the sources.
- Refresh re-crawls the whole site. HTTP ETag / Last-Modified values are stored but not yet used for conditional requests.
- The PyTorch site entry is defined but has not been crawled or evaluated.

## Acknowledgements

Built with the help of an AI coding assistant (Claude, by Anthropic); see the AI-use declaration in the report.
Documentation content © the scikit-learn developers (BSD-3-Clause).
