"""Build and load the search indexes for one site.

Dense: each chunk is embedded with a sentence-transformer and stored in a FAISS
inner-product index. The vectors are normalised, so inner product is cosine
similarity. Sparse: a BM25 index over the same chunks, rebuilt in memory when
the index is loaded (it takes about a second).

Embeddings are cached in SQLite by a hash of the chunk text, so after a refresh
only new or changed chunks are embedded again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from wka.chunking import Chunk, load_chunks, sync_chunks
from wka.config import AppConfig, load_config
from wka.store import Store

EMBEDDING_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    model      TEXT NOT NULL,
    text_hash  TEXT NOT NULL,      -- SHA-256 of Chunk.index_text
    vector     BLOB NOT NULL,      -- float32 values, already normalised
    PRIMARY KEY (model, text_hash)
);
"""

INDEX_DIR = "index"
DENSE_FILE = "dense.faiss"
MANIFEST_FILE = "manifest.json"

# Words, numbers and snake_case identifiers such as max_iter or n_features_in_.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case tokens for BM25. An identifier like max_iter is kept whole and
    also split into its parts, so "max_iter" and "max iter" both match it."""
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        tokens.append(token)
        if "_" in token:
            tokens.extend(token.split("_"))
    return tokens


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_embedding_model(cfg: AppConfig):
    # Imported here because loading torch is slow and BM25-only use does not need it.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cfg.embedding.model_name)


def _embed(model, texts: list[str], batch_size: int, show_progress: bool = False) -> np.ndarray:
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=show_progress)
    return vectors.astype(np.float32)


def build_index(cfg: AppConfig, model=None, verbose: bool = True) -> dict:
    """Bring chunks up to date, embed what is not cached, and write the FAISS index."""
    started = time.perf_counter()
    model_name = cfg.embedding.model_name
    with Store.for_config(cfg) as store:
        chunk_counts = sync_chunks(store, cfg.chunking)
        chunks = load_chunks(store)
        if not chunks:
            raise RuntimeError("No chunks to index. Run the crawl (wka.ingest) first.")

        store.conn.executescript(EMBEDDING_SCHEMA)
        hashes = [_text_hash(c.index_text) for c in chunks]
        cached = {
            h: np.frombuffer(v, dtype=np.float32)
            for h, v in store.conn.execute(
                "SELECT text_hash, vector FROM embeddings WHERE model = ?", (model_name,))
        }
        reused = sum(1 for h in hashes if h in cached)
        # Keyed by hash, so chunks with identical text are embedded once.
        todo = {h: c.index_text for h, c in zip(hashes, chunks) if h not in cached}
        if todo:
            if verbose:
                print(f"Embedding {len(todo):,} new chunks with {model_name} ...")
            model = model or load_embedding_model(cfg)
            vectors = _embed(model, list(todo.values()), cfg.embedding.batch_size, show_progress=verbose)
            with store.conn:
                store.conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (model, text_hash, vector) VALUES (?, ?, ?)",
                    [(model_name, h, v.tobytes()) for h, v in zip(todo, vectors)],
                )
            cached.update(zip(todo, vectors))

        matrix = np.vstack([cached[h] for h in hashes])
        ids = np.array([c.chunk_id for c in chunks], dtype=np.int64)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(matrix.shape[1]))
        index.add_with_ids(matrix, ids)

        index_dir = store.site_dir / INDEX_DIR
        index_dir.mkdir(exist_ok=True)
        # Write to temporary files and rename, so a running chatbot never reads a half-written index.
        faiss.write_index(index, str(index_dir / (DENSE_FILE + ".tmp")))
        manifest = {
            "site_id": cfg.site.site_id,
            "model": model_name,
            "dimension": int(matrix.shape[1]),
            "chunks": len(chunks),
            "pages": len({c.page_id for c in chunks}),
            "built_at": _now(),
        }
        (index_dir / (MANIFEST_FILE + ".tmp")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(index_dir / (DENSE_FILE + ".tmp"), index_dir / DENSE_FILE)
        os.replace(index_dir / (MANIFEST_FILE + ".tmp"), index_dir / MANIFEST_FILE)

        with store.conn:
            # Drop cached vectors no current chunk uses (old versions of changed pages).
            store.conn.execute("CREATE TEMP TABLE live (text_hash TEXT PRIMARY KEY)")
            store.conn.executemany("INSERT OR IGNORE INTO live VALUES (?)", [(h,) for h in hashes])
            pruned = store.conn.execute(
                "DELETE FROM embeddings WHERE model != ? OR text_hash NOT IN (SELECT text_hash FROM live)",
                (model_name,)).rowcount
            store.conn.execute("DROP TABLE live")
            store.conn.execute("UPDATE pages SET last_indexed_at = ? WHERE is_active = 1", (manifest["built_at"],))

    return {
        **manifest,
        "pages_rechunked": chunk_counts["pages_chunked"],
        "embedded": len(todo),
        "reused": reused,
        "pruned": pruned,
        "seconds": round(time.perf_counter() - started, 1),
    }


class SearchIndex:
    """The dense and BM25 indexes of one site, ready to search."""

    def __init__(self, cfg: AppConfig, chunks: list[Chunk], dense: faiss.Index, manifest: dict, stale: bool):
        self.cfg = cfg
        self.manifest = manifest
        self.stale = stale                   # True when pages changed after the index was built
        self.chunks = {c.chunk_id: c for c in chunks}
        self._bm25_ids = [c.chunk_id for c in chunks]
        self._bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])
        self._dense = dense
        self._model = None

    @classmethod
    def load(cls, cfg: AppConfig) -> SearchIndex:
        index_dir = cfg.site_dir / INDEX_DIR
        if not (index_dir / DENSE_FILE).exists():
            raise FileNotFoundError(f"No index for '{cfg.site.site_id}'. Run: uv run python -m wka.indexing")
        manifest = json.loads((index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest["model"] != cfg.embedding.model_name:
            raise RuntimeError(f"Index was built with {manifest['model']}, config says "
                               f"{cfg.embedding.model_name}. Rebuild the index.")
        dense = faiss.read_index(str(index_dir / DENSE_FILE))
        indexed_ids = set(faiss.vector_to_array(dense.id_map).tolist())

        with Store.for_config(cfg) as store:
            current = load_chunks(store)
        # Search only chunks that are both indexed and still in the database.
        chunks = [c for c in current if c.chunk_id in indexed_ids]
        stale = len(chunks) != len(current) or len(chunks) != len(indexed_ids)
        return cls(cfg, chunks, dense, manifest, stale)

    def embed_query(self, query: str) -> np.ndarray:
        if self._model is None:
            self._model = load_embedding_model(self.cfg)
        return _embed(self._model, [self.cfg.embedding.query_prefix + query], 1)

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """(chunk_id, cosine similarity) pairs, best first."""
        # Ask for a few extra in case some indexed chunks were removed since the build.
        scores, ids = self._dense.search(self.embed_query(query), k + 10)
        hits = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if int(i) in self.chunks]
        return hits[:k]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """(chunk_id, BM25 score) pairs, best first. Chunks sharing no word with the query are left out."""
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(self._bm25_ids[i], float(scores[i])) for i in top if scores[i] > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the search indexes, or try a query against them.")
    parser.add_argument("--site", help="site id from config.yaml (default: active_site)")
    parser.add_argument("--query", help="search the existing index instead of building it")
    parser.add_argument("-k", type=int, default=5, help="results to show per method (default: 5)")
    args = parser.parse_args()

    cfg = load_config(site_id=args.site)
    if not args.query:
        stats = build_index(cfg)
        print(f"Indexed {stats['chunks']:,} chunks from {stats['pages']:,} pages in {stats['seconds']}s"
              f" (embedded {stats['embedded']:,}, reused {stats['reused']:,} cached,"
              f" pruned {stats['pruned']:,} old vectors, re-chunked {stats['pages_rechunked']} pages)")
        print(f"Model {stats['model']}, dimension {stats['dimension']}")
        return

    index = SearchIndex.load(cfg)
    if index.stale:
        print("Warning: pages changed since the index was built. Run wka.indexing to update it.\n")
    for name, search in (("BM25", index.bm25_search), ("Dense", index.dense_search)):
        print(f"{name}:")
        for rank, (chunk_id, score) in enumerate(search(args.query, args.k), start=1):
            chunk = index.chunks[chunk_id]
            print(f"  {rank}. {score:7.3f}  {chunk.heading[:70]}")
            print(f"     {chunk.url}")
        print()


if __name__ == "__main__":
    main()