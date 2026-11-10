"""Polite, scope-limited crawler for a single documentation website.
The crawler only discovers and downloads pages. Text extraction and storage
happen in later parts, so nothing here writes to disk.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from wka.config import AppConfig, load_config

MAX_REDIRECTS = 5
RETRY_STATUSES = {429, 500, 502, 503, 504}
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
SKIPPED_LINK_PREFIXES = ("mailto:", "javascript:", "tel:", "#")

class CrawlError(RuntimeError):
    """Raised when a crawl cannot start safely, e.g. robots.txt is unreachable."""

@dataclass
class FetchResult:
    """Outcome of fetching one URL. Failures are recorded, never silently dropped."""

    url: str
    status: str
    fetched_at: str
    http_status: int | None = None
    final_url: str | None = None
    canonical_url: str | None = None
    content_type: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    html: str | None = None
    links: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    error: str | None = None


def _remove_dot_segments(path: str) -> str:
    """Resolve '.' and '..' so '/stable/../dev/' cannot slip past the scope check."""
    output: list[str] = []
    for segment in path.split("/"):
        if segment == "..":
            if len(output) > 1:
                output.pop()
        elif segment != ".":
            output.append(segment)
    result = "/".join(output)
    if path.endswith(("/.", "/..")):
        result += "/"
    return result if result.startswith("/") else "/" + result

def normalise_url(url: str) -> str:
    """Return one consistent spelling of a URL, used to detect duplicates.
    Lower-cases the scheme and host, drops the #fragment, any user:password@ part and default ports, and resolves dot segments. 
    The query string is kept because it can change the page content. Raises ValueError for malformed URLs."""
    parts = urlparse(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if port is None or (scheme, port) in (("http", 80), ("https", 443)):
        netloc = host
    else:
        netloc = f"{host}:{port}"
    path = _remove_dot_segments(parts.path or "/")
    return urlunparse((scheme, netloc, path, parts.params, parts.query, ""))

class Scope:
    """Decides whether a URL belongs to the configured site."""

    def __init__(self, domain: str, path_prefix: str, exclude_patterns: tuple[str, ...]):
        self.domain = domain
        self.path_prefix = path_prefix
        self.excludes = [re.compile(p) for p in exclude_patterns]

    def allows(self, url: str) -> bool:
        parts = urlparse(url)
        if parts.scheme not in ("http", "https"):
            return False
        if (parts.hostname or "").lower() != self.domain:
            return False
        if not parts.path.startswith(self.path_prefix):
            return False
        return not any(pattern.search(parts.path) for pattern in self.excludes)

def parse_links(html: str, page_url: str) -> tuple[str | None, list[str]]:
    """Return the page's canonical URL and its unique outgoing links."""
    soup = BeautifulSoup(html, "lxml")

    base_tag = soup.find("base", href=True)
    base_url = urljoin(page_url, base_tag["href"]) if base_tag else page_url

    canonical = None
    canonical_tag = soup.find("link", rel="canonical", href=True)
    if canonical_tag:
        try:
            canonical = normalise_url(urljoin(base_url, canonical_tag["href"]))
        except ValueError:
            canonical = None

    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(SKIPPED_LINK_PREFIXES):
            continue
        try:
            links.append(normalise_url(urljoin(base_url, href)))
        except ValueError:
            continue    
    return canonical, list(dict.fromkeys(links)) 

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _read_limited(response: requests.Response, limit: int) -> bytes | None:
    """Read the body or return None as soon as it exceeds `limit` bytes."""
    declared = response.headers.get("Content-Length", "")
    if declared.isdigit() and int(declared) > limit:
        return None
    body = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        body.extend(chunk)
        if len(body) > limit:
            return None
    return bytes(body)

def _decode(body: bytes, content_type: str) -> str:
    # requests falls back to ISO-8859-1 when no charset is sent, which garbles
    # UTF-8 pages, so we read the charset ourselves and default to UTF-8.
    match = re.search(r"charset=([\w.-]+)", content_type, re.IGNORECASE)
    encoding = match.group(1) if match else "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


@dataclass
class CrawlStats:
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    by_status: Counter = field(default_factory=Counter)
    out_of_scope_links: int = 0
    robots_blocked_links: int = 0

    def summary(self) -> str:
        statuses = ", ".join(f"{k}={v}" for k, v in sorted(self.by_status.items())) or "none"
        return (
            f"Crawl {self.started_at} -> {self.finished_at}\n"
            f"  results by status : {statuses}\n"
            f"  out-of-scope links: {self.out_of_scope_links}\n"
            f"  robots.txt blocked: {self.robots_blocked_links}"
        )


class Crawler:
    """Breadth-first crawler that stays inside one site's configured scope."""

    def __init__(self, cfg: AppConfig):
        self.site = cfg.site
        self.cfg = cfg.crawler
        self.scope = Scope(self.site.allowed_domain, self.site.allowed_path_prefix, self.site.exclude_patterns)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = self.cfg.user_agent
        self.stats = CrawlStats()
        self.robots = self._load_robots()
        # Use the slower of our own delay and any Crawl-delay the site asks for.
        robots_delay = self.robots.crawl_delay(self.cfg.user_agent)
        self.delay = max(self.cfg.request_delay_seconds, float(robots_delay or 0))
        self._last_request = float("-inf")
        self._seconds_waiting = 0.0  # polite-delay time, excluded from elapsed_ms

    def __enter__(self) -> Crawler:
        return self

    def __exit__(self, *exc_info) -> None:
        self.session.close()

    def _load_robots(self) -> RobotFileParser:
        """Fetch robots.txt following RFC 9309. 404 or other 4xx means no rules; 401/403 means disallow all; a server error or network failure stops the crawl, because we cannot know what the site permits."""
        parts = urlparse(self.site.base_url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        parser = RobotFileParser(robots_url)
        try:
            response = self.session.get(robots_url, timeout=self.cfg.timeout_seconds)
        except requests.RequestException as exc:
            raise CrawlError(f"Could not fetch {robots_url}: {exc}") from exc
        if response.status_code >= 500:
            raise CrawlError(f"{robots_url} returned {response.status_code}; not crawling.")
        if response.status_code in (401, 403):
            parser.disallow_all = True
        elif response.status_code >= 400:
            parser.allow_all = True
        else:
            parser.parse(response.text.splitlines())
        return parser

    def robots_allows(self, url: str) -> bool:
        return self.robots.can_fetch(self.cfg.user_agent, url)

    def _wait_politely(self) -> None:
        wait = self.delay - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
            self._seconds_waiting += wait
        self._last_request = time.monotonic()

    def _get(self, url: str) -> requests.Response:
        """GET one URL without following redirects, retrying transient failures."""
        for attempt in range(self.cfg.max_retries + 1):
            self._wait_politely()
            backoff = 2.0 ** attempt
            try:
                response = self.session.get(
                    url, timeout=self.cfg.timeout_seconds, allow_redirects=False, stream=True
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.cfg.max_retries:
                    raise
            else:
                if response.status_code not in RETRY_STATUSES or attempt == self.cfg.max_retries:
                    return response
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    backoff = min(float(retry_after), 60.0)
                response.close()
            time.sleep(backoff)
        raise AssertionError("unreachable")  # the loop always returns or raises

    def fetch(self, url: str) -> FetchResult:
        """Fetch one in-scope URL. Every redirect hop is re-checked against the scope."""
        result = FetchResult(url=url, status="failed", fetched_at=_now())
        started = time.monotonic()
        waited_before = self._seconds_waiting
        current = url
        try:
            for _ in range(MAX_REDIRECTS + 1):
                response = self._get(current)
                if not response.is_redirect:
                    break
                location = response.headers.get("Location", "")
                response.close()
                target = normalise_url(urljoin(current, location))
                if not self.scope.allows(target) or not self.robots_allows(target):
                    result.status = "redirect_blocked"
                    result.error = f"Redirect to {target} is outside the crawl scope or disallowed."
                    return result
                current = target
            else:
                result.error = f"More than {MAX_REDIRECTS} redirects."
                return result

            with response:
                result.http_status = response.status_code
                result.final_url = current
                result.content_type = response.headers.get("Content-Type", "")
                result.etag = response.headers.get("ETag")
                result.last_modified = response.headers.get("Last-Modified")

                if not 200 <= response.status_code < 300:
                    result.status = "http_error"
                    return result
                if not result.content_type.lower().startswith(HTML_CONTENT_TYPES):
                    result.status = "not_html"
                    return result
                body = _read_limited(response, self.cfg.max_response_bytes)
                if body is None:
                    result.status = "too_large"
                    return result

            result.html = _decode(body, result.content_type)
            canonical, result.links = parse_links(result.html, current)
            # Only trust a canonical URL that stays inside our scope.
            result.canonical_url = canonical if canonical and self.scope.allows(canonical) else current
            result.status = "ok"
            return result
        except (requests.RequestException, ValueError) as exc:
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        finally:
            waited = self._seconds_waiting - waited_before
            result.elapsed_ms = int((time.monotonic() - started - waited) * 1000)


    def crawl(self, limit: int | None = None) -> Iterator[FetchResult]:
        """Yield a FetchResult for every URL attempted, breadth-first from base_url.

        Stops after `limit` (or the site's max_pages) unique pages have been
        fetched successfully.
        """
        max_pages = min(limit, self.site.max_pages) if limit else self.site.max_pages
        start = normalise_url(self.site.base_url)
        if not self.robots_allows(start):
            raise CrawlError(f"robots.txt disallows the start URL {start}.")

        queue = deque([start])
        seen = {start}            # every URL ever queued or skipped, so nothing is fetched twice
        seen_canonical: set[str] = set()
        pages_ok = 0

        while queue and pages_ok < max_pages:
            result = self.fetch(queue.popleft())

            if result.status == "ok":
                if result.canonical_url in seen_canonical:
                    # Same page reached through a different URL, e.g. /stable/ and /stable/index.html.
                    result.status = "duplicate"
                    result.html = None
                else:
                    seen_canonical.add(result.canonical_url)
                    seen.update({result.final_url, result.canonical_url})
                    pages_ok += 1
                    for link in result.links:
                        if link in seen:
                            continue
                        seen.add(link)
                        if not self.scope.allows(link):
                            self.stats.out_of_scope_links += 1
                        elif not self.robots_allows(link):
                            self.stats.robots_blocked_links += 1
                        else:
                            queue.append(link)

            self.stats.by_status[result.status] += 1
            yield result

        self.stats.finished_at = _now()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test-crawl the configured site. Nothing is saved.")
    parser.add_argument("--site", help="site id from config.yaml (default: active_site)")
    parser.add_argument("--limit", type=int, default=20, help="stop after this many unique pages")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be a positive integer")

    cfg = load_config(site_id=args.site)
    print(f"Crawling {cfg.site.base_url} (scope {cfg.site.allowed_domain}{cfg.site.allowed_path_prefix})")
    with Crawler(cfg) as crawler:
        print(f"Delay between requests: {crawler.delay:.1f}s")
        for r in crawler.crawl(limit=args.limit):
            note = f"  ({r.error})" if r.error else ""
            print(f"[{r.status:>16}] {r.http_status or '-':>3} {r.elapsed_ms:>5} ms  {r.url}{note}")
        print(crawler.stats.summary())


if __name__ == "__main__":
    main()
