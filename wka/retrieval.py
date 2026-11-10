"""Retrieve the chunks that best answer a question: BM25, dense, or both fused.

Hybrid retrieval uses Reciprocal Rank Fusion (Cormack et al., 2009). Each
method ranks its top candidate_k chunks and a chunk scores

    sum over methods of 1 / (rrf_k + rank)

RRF uses ranks only, so the two methods' scores (BM25 is unbounded, cosine is
between -1 and 1) never need to be put on the same scale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from wka.chunking import Chunk
from wka.config import AppConfig, RetrievalConfig, load_config
from wka.indexing import SearchIndex

METHODS = ("bm25", "dense", "hybrid")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float                  # BM25 score, cosine similarity, or RRF score, depending on the method
    bm25_rank: int | None         # 1-based rank in the BM25 candidates, None if not among them
    dense_rank: int | None        # 1-based rank in the dense candidates, None if not among them


def rrf_fuse(rankings: list[list[int]], rrf_k: int) -> list[tuple[int, float]]:
    """Fuse ranked lists of chunk ids. Returns (chunk_id, score) pairs, best first.

    Ties are broken by the best single rank, then by chunk id, so results are deterministic.
    """
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))


class Retriever:
    def __init__(self, index: SearchIndex, cfg: RetrievalConfig):
        self.index = index
        self.cfg = cfg

    @classmethod
    def load(cls, cfg: AppConfig) -> Retriever:
        return cls(SearchIndex.load(cfg), cfg.retrieval)

    def search_all(self, query: str, k: int | None = None) -> dict[str, list[Hit]]:
        """Top-k hits of every method, running each underlying search only once."""
        k = k or self.cfg.top_k
        query = query.strip()
        if not query:
            return {method: [] for method in METHODS}

        n = max(self.cfg.candidate_k, k)
        bm25 = self.index.bm25_search(query, n)
        dense = self.index.dense_search(query, n)
        bm25_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25, start=1)}
        dense_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense, start=1)}

        def hits(scored: list[tuple[int, float]]) -> list[Hit]:
            return [Hit(self.index.chunks[chunk_id], score,
                        bm25_ranks.get(chunk_id), dense_ranks.get(chunk_id))
                    for chunk_id, score in scored[:k]]

        fused = rrf_fuse([[c for c, _ in bm25], [c for c, _ in dense]], self.cfg.rrf_k)
        return {"bm25": hits(bm25), "dense": hits(dense), "hybrid": hits(fused)}

    def search(self, query: str, method: str = "hybrid", k: int | None = None) -> list[Hit]:
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {', '.join(METHODS)}.")
        return self.search_all(query, k)[method]


def _rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the active site's index.")
    parser.add_argument("query", help="the question to search for")
    parser.add_argument("--site", help="site id from config.yaml (default: active_site)")
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all",
                        help="retrieval method to show (default: all three)")
    parser.add_argument("-k", type=int, help="number of results (default: retrieval.top_k)")
    args = parser.parse_args()

    cfg = load_config(site_id=args.site)
    retriever = Retriever.load(cfg)
    if retriever.index.stale:
        print("Warning: pages changed since the index was built. Run wka.indexing to update it.\n")

    results = retriever.search_all(args.query, args.k)
    for method in METHODS if args.method == "all" else (args.method,):
        print(f"{method.upper()}  (rank in BM25 / dense candidates)")
        for rank, hit in enumerate(results[method], start=1):
            print(f"  {rank}. {hit.score:8.4f}  [{_rank(hit.bm25_rank):>2} / {_rank(hit.dense_rank):>2}]"
                  f"  {hit.chunk.heading[:60]}")
            print(f"     {hit.chunk.url}")
        if not results[method]:
            print("  (no results)")
        print()


if __name__ == "__main__":
    main()