"""Split stored pages into chunks for retrieval.

Chunks follow the section structure from extraction: small neighbouring
sections are merged, long ones are split on line boundaries with a small
overlap. Each chunk remembers the page version it was built from, so only
pages whose content changed are re-chunked.

Run with:  uv run python -m wka.chunking
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter
from dataclasses import dataclass

from wka.config import ChunkingConfig, load_config
from wka.extract import CODE_FENCE, Section
from wka.store import Store, sections_from_json

CHUNK_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,  -- never reused, so an old index entry cannot point at a new chunk
    page_id       INTEGER NOT NULL REFERENCES pages(id),
    page_version  INTEGER NOT NULL,                   -- pages.version the chunk was built from
    position      INTEGER NOT NULL,                   -- order within the page, from 0
    heading       TEXT NOT NULL,                      -- heading trail, e.g. "Ridge > Examples"
    text          TEXT NOT NULL,
    char_count    INTEGER NOT NULL,
    UNIQUE (page_id, position)
);

CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id);
"""

HEADING_SEPARATOR = " > "


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    page_id: int
    page_version: int
    position: int
    url: str
    title: str
    heading: str
    text: str

    @property
    def index_text(self) -> str:
        """Text given to the embedding model and BM25: heading trail first, then the body."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def _wrap(text: str, limit: int) -> list[str]:
    """Split text into lines, breaking any line longer than `limit` at a space."""
    lines: list[str] = []
    for line in text.split("\n"):
        while len(line) > limit:
            cut = line.rfind(" ", 0, limit)
            if cut < limit // 2:          # no sensible space: cut mid-word
                cut = limit
            lines.append(line[:cut].rstrip())
            line = line[cut:].lstrip()
        lines.append(line)
    return lines


def _overlap(lines: list[str], limit: int, in_code: bool) -> list[str]:
    """Last lines of a chunk to repeat at the start of the next, at most `limit` chars.

    Never crosses a code fence, so the repeated lines are all code or all prose.
    """
    tail: list[str] = []
    size = 0
    for line in reversed(lines):
        if line == CODE_FENCE:
            break
        if size + len(line) + 1 > limit:
            if not tail and not in_code:  # one long last line: keep its end, starting at a word
                cut = line.find(" ", len(line) - limit)
                if cut != -1:
                    tail.append(line[cut + 1:])
            break
        tail.append(line)
        size += len(line) + 1
    return tail[::-1]


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split text into pieces of at most `max_chars`, cutting only between lines.

    A cut inside a code block closes the fence and reopens it in the next piece,
    so every piece has balanced ``` fences.
    """
    if len(text) <= max_chars:
        return [text]
    fence_cost = len(CODE_FENCE) + 1
    room = max_chars - fence_cost     # keep space to close a code block at the cut
    # Aim for pieces of similar size, so a section just over the limit does not
    # leave a tiny last piece.
    n_pieces = math.ceil(len(text) / max(room - overlap_chars, 1))
    target = min(room, len(text) // n_pieces + overlap_chars)

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    new_lines = 0                     # lines in `current` that are not repeated overlap
    in_code = False
    for line in _wrap(text, room - fence_cost):
        if new_lines and size + len(line) + 1 > target:
            opening = in_code and current[-1] == CODE_FENCE and new_lines > 1
            if opening:
                current.pop()             # a block opens right at the cut: open it in the next piece instead
            cut_in_code = in_code and not opening
            pieces.append("\n".join(current + [CODE_FENCE] if cut_in_code else current))
            carry = _overlap(current, overlap_chars, cut_in_code)
            if opening:
                current = carry + [CODE_FENCE]
            else:
                current = ([CODE_FENCE] if cut_in_code else []) + carry
            if sum(len(x) + 1 for x in current) + len(line) + 1 > room:
                current = [CODE_FENCE] if in_code else []   # overlap leaves no room for this line
            size = sum(len(x) + 1 for x in current)
            new_lines = 0
        current.append(line)
        size += len(line) + 1
        new_lines += 1
        if line == CODE_FENCE:
            in_code = not in_code
    if new_lines:
        pieces.append("\n".join(current))
    return pieces


def _common_prefix(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    prefix = paths[0]
    for path in paths[1:]:
        n = 0
        while n < min(len(prefix), len(path)) and prefix[n] == path[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


def _group_sections(sections: list[Section], cfg: ChunkingConfig) -> list[list[Section]]:
    """Merge a section shorter than min_chars with its neighbour when both fit in one chunk."""
    groups: list[list[Section]] = []
    for section in sections:
        if groups:
            last_size = sum(len(s.text) for s in groups[-1])
            small = last_size < cfg.min_chars or len(section.text) < cfg.min_chars
            if small and last_size + len(section.text) <= cfg.max_chars:
                groups[-1].append(section)
                continue
        groups.append([section])
    return groups


def chunk_page(title: str, sections: list[Section], cfg: ChunkingConfig) -> list[tuple[str, str]]:
    """Return (heading, text) pairs for one page, in page order."""
    chunks: list[tuple[str, str]] = []
    for group in _group_sections(sections, cfg):
        heading_path = _common_prefix([s.path for s in group])
        parts: list[str] = []
        for section in group:
            # Merged sections keep their own heading as a line, so the text still says where it came from.
            extra = section.path[len(heading_path):]
            if len(group) > 1 and extra:
                parts.append(HEADING_SEPARATOR.join(extra))
            parts.append(section.text)
        heading = HEADING_SEPARATOR.join(heading_path) or title
        for piece in split_text("\n".join(parts), cfg.max_chars, cfg.overlap_chars):
            chunks.append((heading, piece))
    return chunks


def sync_chunks(store: Store, cfg: ChunkingConfig, rebuild: bool = False) -> Counter:
    """Bring the chunks table in line with the pages table.

    Only pages that are new or whose version changed are re-chunked; chunks of
    inactive pages are removed. `rebuild` re-chunks everything, which is needed
    after changing the chunking settings.
    """
    counts: Counter = Counter()
    conn = store.conn
    conn.executescript(CHUNK_SCHEMA)
    with conn:
        if rebuild:
            counts["chunks_deleted"] += conn.execute("DELETE FROM chunks").rowcount
        built = dict(conn.execute("SELECT page_id, MIN(page_version) FROM chunks GROUP BY page_id"))
        pages = conn.execute("SELECT id, title, version, sections_json, is_active FROM pages").fetchall()
        for page in pages:
            if not page["is_active"]:
                if page["id"] in built:
                    counts["chunks_deleted"] += conn.execute(
                        "DELETE FROM chunks WHERE page_id = ?", (page["id"],)).rowcount
                    counts["pages_removed"] += 1
                continue
            if built.get(page["id"]) == page["version"]:
                counts["pages_unchanged"] += 1
                continue
            counts["chunks_deleted"] += conn.execute(
                "DELETE FROM chunks WHERE page_id = ?", (page["id"],)).rowcount
            pieces = chunk_page(page["title"], sections_from_json(page["sections_json"]), cfg)
            conn.executemany(
                "INSERT INTO chunks (page_id, page_version, position, heading, text, char_count)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(page["id"], page["version"], i, heading, text, len(text))
                 for i, (heading, text) in enumerate(pieces)],
            )
            counts["pages_chunked"] += 1
            counts["chunks_added"] += len(pieces)
    return counts


def load_chunks(store: Store) -> list[Chunk]:
    """All chunks of active pages, in a stable order (used by the indexing step)."""
    store.conn.executescript(CHUNK_SCHEMA)
    rows = store.conn.execute(
        "SELECT c.id, c.page_id, c.page_version, c.position, p.url, p.title, c.heading, c.text"
        " FROM chunks c JOIN pages p ON p.id = c.page_id"
        " WHERE p.is_active = 1 ORDER BY c.id"
    ).fetchall()
    return [Chunk(*row) for row in rows]


def describe(store: Store) -> str:
    sizes = sorted(r[0] for r in store.conn.execute("SELECT char_count FROM chunks"))
    if not sizes:
        return "No chunks yet."
    pages = store.conn.execute("SELECT COUNT(DISTINCT page_id) FROM chunks").fetchone()[0]
    return "\n".join([
        f"Chunks   : {len(sizes):,} from {pages:,} pages ({len(sizes) / pages:.1f} per page)",
        f"Size     : min {sizes[0]}, median {int(statistics.median(sizes))},"
        f" mean {int(statistics.mean(sizes))}, p90 {sizes[int(len(sizes) * 0.9)]}, max {sizes[-1]} chars",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the stored pages into retrieval chunks.")
    parser.add_argument("--site", help="site id from config.yaml (default: active_site)")
    parser.add_argument("--rebuild", action="store_true", help="re-chunk every page (after changing chunk settings)")
    parser.add_argument("--show", metavar="URL", help="print the chunks of one page instead")
    args = parser.parse_args()

    cfg = load_config(site_id=args.site)
    with Store.for_config(cfg) as store:
        if args.show:
            for chunk in load_chunks(store):
                if chunk.url == args.show:
                    print(f"--- chunk {chunk.chunk_id} (position {chunk.position}, {len(chunk.text)} chars)")
                    print(chunk.index_text)
            return
        counts = sync_chunks(store, cfg.chunking, rebuild=args.rebuild)
        print("Chunking: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        print(describe(store))


if __name__ == "__main__":
    main()