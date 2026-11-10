"""Crawl the configured site, extract each page and store it in SQLite."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

from wka.config import AppConfig, load_config
from wka.crawler import Crawler
from wka.extract import extract_page
from wka.store import Store



def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_crawl(cfg: AppConfig, limit: int | None = None, verbose: bool = True) -> int:
    """Crawl and store one site. Returns the crawl run id."""
    page_limit = min(limit, cfg.site.max_pages) if limit else cfg.site.max_pages
    counts: Counter = Counter()

    with Store.for_config(cfg) as store, Crawler(cfg) as crawler:
        config_json = json.dumps(asdict(cfg), default=str, indent=2)
        run_id = store.start_run(cfg.site.site_id, _now(), page_limit, config_json)
        try:
            for result in crawler.crawl(limit=page_limit):
                outcome = None
                error = result.error

                if result.status == "ok":
                    page = extract_page(result.html, result.canonical_url)
                    if page is None:
                        outcome = "extract_failed"
                        error = "No main content found on the page."
                        counts["pages_failed"] += 1
                    else:
                        outcome = store.save_page(
                            result.canonical_url, page, result.html,
                            result.etag, result.last_modified, result.fetched_at,
                        )
                        counts[f"pages_{outcome}"] += 1
                elif result.status == "duplicate":
                    counts["pages_duplicate"] += 1
                elif result.status in ("not_html", "redirect_blocked"):
                    pass  # expected for some links; logged below but not a failure
                else:
                    counts["pages_failed"] += 1
                    store.record_page_error(result.url, error or f"HTTP {result.http_status}")

                store.log_fetch(run_id, result.url, result.status, outcome, result.http_status,
                                result.elapsed_ms, result.fetched_at, error)
                if verbose:
                    label = outcome or result.status
                    print(f"[{label:>16}] {result.http_status or '-':>3} {result.url}")

            counts["out_of_scope_links"] = crawler.stats.out_of_scope_links
            store.finish_run(run_id, _now(), "completed", counts)
        except KeyboardInterrupt:
            store.finish_run(run_id, _now(), "interrupted", counts, "Stopped by user.")
            raise
        except Exception as exc:
            store.finish_run(run_id, _now(), "failed", counts, f"{type(exc).__name__}: {exc}")
            raise

        print()
        print(store.summary())
    return run_id

def reextract(cfg: AppConfig) -> Counter:
    """Re-run extraction on every stored HTML snapshot without fetching anything.

    Used after the extractor is improved. Pages whose text changes get a new
    version, so the chunks built from them are rebuilt by the next chunking run.
    """
    counts: Counter = Counter()
    with Store.for_config(cfg) as store:
        rows = store.conn.execute("SELECT * FROM pages WHERE raw_html_path IS NOT NULL").fetchall()
        for row in rows:
            html = gzip.decompress((store.site_dir / row["raw_html_path"]).read_bytes()).decode("utf-8")
            page = extract_page(html, row["url"])
            if page is None:
                counts["extract_failed"] += 1
                continue
            outcome = store.save_page(row["url"], page, html, row["etag"],
                                      row["last_modified"], row["last_fetched_at"])
            counts[outcome] += 1
    return counts

def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl the site and store the extracted pages.")
    parser.add_argument("--site", help="site id from config.yaml (default: active_site)")
    parser.add_argument("--limit", type=int, help="stop after this many unique pages (default: max_pages)")
    parser.add_argument("--quiet", action="store_true", help="only print the final summary")
    parser.add_argument("--reextract", action="store_true", help="re-extract the stored HTML snapshots instead of crawling")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    cfg = load_config(site_id=args.site)
    if args.reextract:
        counts = reextract(cfg)
        print("Re-extracted stored pages: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return
    print(f"Crawling {cfg.site.name} into {cfg.site_dir}")
    run_id = run_crawl(cfg, limit=args.limit, verbose=not args.quiet)
    print(f"Finished crawl run #{run_id}")


if __name__ == "__main__":
    main()