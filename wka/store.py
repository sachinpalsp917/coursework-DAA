"""SQLite storage for one site's pages, crawl runs and fetch log.

Each site has its own database at data/<site_id>/wka.sqlite3, so switching the
active site never mixes content from two sites.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

from wka.config import AppConfig, load_config
from wka.extract import ExtractedPage, Section

SCHEMA = """
CREATE TABLE IF NOT EXISTS crawl_runs (
    id                  INTEGER PRIMARY KEY,
    site_id             TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL,          -- running | completed | failed | interrupted
    page_limit          INTEGER,
    config_json         TEXT NOT NULL,          -- full config used, for reproducibility
    pages_new           INTEGER NOT NULL DEFAULT 0,
    pages_changed       INTEGER NOT NULL DEFAULT 0,
    pages_unchanged     INTEGER NOT NULL DEFAULT 0,
    pages_failed        INTEGER NOT NULL DEFAULT 0,
    pages_duplicate     INTEGER NOT NULL DEFAULT 0,
    out_of_scope_links  INTEGER NOT NULL DEFAULT 0,
    error               TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    id               INTEGER PRIMARY KEY,
    url              TEXT NOT NULL UNIQUE,      -- canonical URL
    title            TEXT NOT NULL,
    version          INTEGER NOT NULL,          -- +1 every time the extracted content changes
    content_hash     TEXT NOT NULL,
    sections_json    TEXT NOT NULL,             
    char_count       INTEGER NOT NULL,
    raw_html_path    TEXT,                      
    etag             TEXT,
    last_modified    TEXT,
    first_seen_at    TEXT NOT NULL,
    last_fetched_at  TEXT NOT NULL,             -- last successful fetch
    last_changed_at  TEXT NOT NULL,
    last_indexed_at  TEXT,                      -- set by the indexing step 
    last_error       TEXT,                      -- most recent failed fetch, cleared on success
    is_active        INTEGER NOT NULL DEFAULT 1 -- 0 once a page is confirmed removed 
);

CREATE TABLE IF NOT EXISTS fetch_log (
    id           INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES crawl_runs(id),
    url          TEXT NOT NULL,
    fetch_status TEXT NOT NULL,                 -- crawler status: ok, http_error, failed, ...
    outcome      TEXT,                          -- new | changed | unchanged | extract_failed
    http_status  INTEGER,
    elapsed_ms   INTEGER,
    fetched_at   TEXT NOT NULL,
    error        TEXT
);

CREATE INDEX IF NOT EXISTS idx_fetch_log_run ON fetch_log(run_id);
"""

RUN_COUNTERS = (
    "pages_new", "pages_changed", "pages_unchanged",
    "pages_failed", "pages_duplicate", "out_of_scope_links",
)


def sections_to_json(sections: tuple[Section, ...]) -> str:
    return json.dumps([{"path": list(s.path), "text": s.text} for s in sections], ensure_ascii=False)


def sections_from_json(raw: str) -> list[Section]:
    return [Section(path=tuple(item["path"]), text=item["text"]) for item in json.loads(raw)]


class Store:
    """Thin wrapper around the site's SQLite database."""

    def __init__(self, site_dir: Path):
        self.site_dir = site_dir
        self.raw_dir = site_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.path = site_dir / "wka.sqlite3"
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)

    @classmethod
    def for_config(cls, cfg: AppConfig) -> Store:
        return cls(cfg.site_dir)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


    def start_run(self, site_id: str, started_at: str, page_limit: int, config_json: str) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO crawl_runs (site_id, started_at, status, page_limit, config_json)"
                " VALUES (?, ?, 'running', ?, ?)",
                (site_id, started_at, page_limit, config_json),
            )
        return cursor.lastrowid

    def finish_run(self, run_id: int, finished_at: str, status: str,
                   counts: dict[str, int], error: str | None = None) -> None:
        values = [counts.get(name, 0) for name in RUN_COUNTERS]
        assignments = ", ".join(f"{name} = ?" for name in RUN_COUNTERS)
        with self.conn:
            self.conn.execute(
                f"UPDATE crawl_runs SET finished_at = ?, status = ?, error = ?, {assignments} WHERE id = ?",
                (finished_at, status, error, *values, run_id),
            )

    def log_fetch(self, run_id: int, url: str, fetch_status: str, outcome: str | None,
                  http_status: int | None, elapsed_ms: int, fetched_at: str, error: str | None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO fetch_log (run_id, url, fetch_status, outcome, http_status, elapsed_ms, fetched_at, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, url, fetch_status, outcome, http_status, elapsed_ms, fetched_at, error),
            )


    def get_page(self, url: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM pages WHERE url = ?", (url,)).fetchone()

    def save_raw_html(self, html: str) -> str:
        """Store a gzipped snapshot named by its hash; identical HTML is stored once."""
        data = html.encode("utf-8")
        name = hashlib.sha256(data).hexdigest() + ".html.gz"
        target = self.raw_dir / name
        if not target.exists():
            target.write_bytes(gzip.compress(data))
        return str(target.relative_to(self.site_dir))

    def save_page(self, url: str, page: ExtractedPage, html: str, etag: str | None,
                  last_modified: str | None, fetched_at: str) -> str:
        """Insert or update a page and return 'new', 'changed' or 'unchanged'.

        The raw HTML snapshot is only written for new or changed content.
        """
        existing = self.get_page(url)
        raw_html_path = None
        if existing is None or existing["content_hash"] != page.content_hash:
            raw_html_path = self.save_raw_html(html)
        with self.conn:
            if existing is None:
                self.conn.execute(
                    "INSERT INTO pages (url, title, version, content_hash, sections_json, char_count,"
                    " raw_html_path, etag, last_modified, first_seen_at, last_fetched_at, last_changed_at)"
                    " VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (url, page.title, page.content_hash, sections_to_json(page.sections), page.char_count,
                     raw_html_path, etag, last_modified, fetched_at, fetched_at, fetched_at),
                )
                return "new"
            if existing["content_hash"] == page.content_hash:
                # Same text: refresh the bookkeeping only. A page seen again is active again.
                self.conn.execute(
                    "UPDATE pages SET etag = ?, last_modified = ?, last_fetched_at = ?,"
                    " last_error = NULL, is_active = 1 WHERE id = ?",
                    (etag, last_modified, fetched_at, existing["id"]),
                )
                return "unchanged"
            self.conn.execute(
                "UPDATE pages SET title = ?, version = version + 1, content_hash = ?, sections_json = ?,"
                " char_count = ?, raw_html_path = ?, etag = ?, last_modified = ?, last_fetched_at = ?,"
                " last_changed_at = ?, last_error = NULL, is_active = 1 WHERE id = ?",
                (page.title, page.content_hash, sections_to_json(page.sections), page.char_count,
                 raw_html_path, etag, last_modified, fetched_at, fetched_at, existing["id"]),
            )
            return "changed"

    def record_page_error(self, url: str, error: str) -> None:
        """Note a failed fetch on a known page. The page and its content are kept."""
        with self.conn:
            self.conn.execute("UPDATE pages SET last_error = ? WHERE url = ?", (error, url))


    def summary(self) -> str:
        pages = self.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(char_count), 0) AS chars,"
            " COALESCE(SUM(is_active), 0) AS active FROM pages"
        ).fetchone()
        lines = [
            f"Database : {self.path}",
            f"Pages    : {pages['n']} stored, {pages['active']} active, {pages['chars']:,} characters of text",
            "Recent crawl runs:",
        ]
        runs = self.conn.execute("SELECT * FROM crawl_runs ORDER BY id DESC LIMIT 5").fetchall()
        for run in runs:
            lines.append(
                f"  #{run['id']} {run['started_at']} {run['status']:<11} new={run['pages_new']}"
                f" changed={run['pages_changed']} unchanged={run['pages_unchanged']}"
                f" failed={run['pages_failed']} duplicate={run['pages_duplicate']}"
            )
        if not runs:
            lines.append("  (none yet)")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show what is stored for a site.")
    parser.add_argument("--site", help="site id from config.yaml (default: active_site)")
    args = parser.parse_args()
    cfg = load_config(site_id=args.site)
    with Store.for_config(cfg) as store:
        print(store.summary())


if __name__ == "__main__":
    main()