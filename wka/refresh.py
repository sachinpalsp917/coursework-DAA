"""Refresh a site: re-crawl it, detect what changed, and update the index.

Steps:
1. Crawl the whole site again. Each page's extracted text is hashed, so pages
   come back as new, changed or unchanged (see Store.save_page).
2. If the crawl covered the whole site (it did not stop at the page limit),
   re-check every stored page the crawl did not reach. A page is only marked
   removed when the server confirms it: 404/410, or a redirect to another page.
   Network errors and 5xx responses never remove anything.
3. Rebuild the index. Only chunks of new or changed pages are embedded again.

Run with:  uv run python -m wka.refresh
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from datetime import datetime, timezone

from wka.config import AppConfig, load_config
from wka.crawler import Crawler
from wka.extract import extract_page
from wka.indexing import build_index
from wka.ingest import run_crawl
from wka.store import Store

GONE_STATUSES = {404, 410}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_missing(store: Store, crawler: Crawler, run_id: int, urls: list[str]) -> Counter:
    """Fetch pages the crawl did not reach and decide whether each one still exists."""
    counts: Counter = Counter()
    for url in urls:
        reason = None
        if not crawler.robots_allows(url):
            reason = "Now disallowed by robots.txt."
            result = None
        else:
            result = crawler.fetch(url)
            if result.status == "ok" and result.canonical_url != url:
                reason = f"Now redirects to {result.canonical_url}."
            elif result.status == "redirect_blocked":
                reason = result.error
            elif result.status == "http_error" and result.http_status in GONE_STATUSES:
                reason = f"HTTP {result.http_status}."

        if reason:
            with store.conn:
                store.conn.execute("UPDATE pages SET is_active = 0, last_error = ? WHERE url = ?",
                                   (f"Removed: {reason}", url))
            outcome = "removed"
        elif result is not None and result.status == "ok":
            # Still online but no longer linked from the site: keep it and update its text.
            page = extract_page(result.html, url)
            if page is not None:
                store.save_page(url, page, result.html, result.etag, result.last_modified, result.fetched_at)
            outcome = "unlinked"
        else:
            # Temporary problem (timeout, 5xx ...): keep the page and its content.
            store.record_page_error(url, result.error or f"HTTP {result.http_status}")
            outcome = "check_failed"

        counts[outcome] += 1
        store.log_fetch(run_id, url, result.status if result else "robots_blocked", outcome,
                        result.http_status if result else None, result.elapsed_ms if result else 0,
                        result.fetched_at if result else _now(), reason)
    return counts


def refresh(cfg: AppConfig, limit: int | None = None, verbose: bool = False) -> dict:
    started = time.perf_counter()
    run_id = run_crawl(cfg, limit=limit, verbose=verbose)

    with Store.for_config(cfg) as store, Crawler(cfg) as crawler:
        run = store.conn.execute("SELECT * FROM crawl_runs WHERE id = ?", (run_id,)).fetchone()
        fetched_ok = store.conn.execute(
            "SELECT COUNT(*) FROM fetch_log WHERE run_id = ? AND fetch_status = 'ok'", (run_id,)).fetchone()[0]
        complete = fetched_ok < run["page_limit"]   # the crawl ran out of links before the limit

        missing_counts: Counter = Counter()
        missing: list[str] = []
        if complete:
            missing = [row[0] for row in store.conn.execute(
                "SELECT url FROM pages WHERE is_active = 1 AND last_fetched_at < ?", (run["started_at"],))]
            missing_counts = _check_missing(store, crawler, run_id, missing)

        changes = {
            outcome: [row[0] for row in store.conn.execute(
                "SELECT url FROM fetch_log WHERE run_id = ? AND outcome = ? ORDER BY id", (run_id, outcome))]
            for outcome in ("new", "changed", "removed")
        }

    index_stats = build_index(cfg, verbose=verbose)
    return {
        "run_id": run_id,
        "complete": complete,
        "page_limit": run["page_limit"],
        "pages_new": run["pages_new"],
        "pages_changed": run["pages_changed"],
        "pages_unchanged": run["pages_unchanged"],
        "pages_failed": run["pages_failed"],
        "not_reached": len(missing),
        "missing": dict(missing_counts),
        "changes": changes,
        "index": index_stats,
        "seconds": round(time.perf_counter() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-crawl the site, detect changes and update the index.")
    parser.add_argument("--site", help="site id from config.yaml (default: active_site)")
    parser.add_argument("--limit", type=int, help="stop after this many pages (skips removal checks)")
    parser.add_argument("--verbose", action="store_true", help="print every fetched URL")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    cfg = load_config(site_id=args.site)
    print(f"Refreshing {cfg.site.name} ...")
    stats = refresh(cfg, limit=args.limit, verbose=args.verbose)

    minutes, seconds = divmod(stats["seconds"], 60)
    print(f"\nRefresh (crawl run #{stats['run_id']}) finished in {minutes}m {seconds:02d}s")
    print(f"Pages  : {stats['pages_new']} new, {stats['pages_changed']} changed,"
          f" {stats['pages_unchanged']} unchanged, {stats['pages_failed']} failed")
    if stats["complete"]:
        missing = stats["missing"]
        print(f"Removal: {stats['not_reached']} stored pages not reached by the crawl -> "
              f"{missing.get('removed', 0)} removed, {missing.get('unlinked', 0)} still online,"
              f" {missing.get('check_failed', 0)} could not be checked (kept)")
    else:
        print(f"Removal: skipped, the crawl stopped at the page limit ({stats['page_limit']}),"
              " so unreached pages may still exist")
    for outcome in ("new", "changed", "removed"):
        urls = stats["changes"][outcome]
        if urls:
            print(f"{outcome.capitalize()} pages:")
            for url in urls[:20]:
                print(f"  {url}")
            if len(urls) > 20:
                print(f"  ... and {len(urls) - 20} more")
    index = stats["index"]
    print(f"Index  : {index['chunks']:,} chunks, embedded {index['embedded']:,} new,"
          f" reused {index['reused']:,}, pruned {index['pruned']:,} old vectors")


if __name__ == "__main__":
    main()