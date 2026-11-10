"""Load and validate the project configuration from config/config.yaml"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_SITE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

class ConfigError(ValueError):
    """Raised when the config file is missing a value or contains an invalid one."""

@dataclass(frozen=True)
class SiteConfig:
    site_id: str
    name: str
    base_url: str
    allowed_domain: str
    allowed_path_prefix: str
    max_pages: int
    exclude_patterns: tuple[str, ...]

@dataclass(frozen=True)
class CrawlerConfig:
    user_agent: str
    request_delay_seconds: float
    timeout_seconds: float
    max_retries: int
    max_response_bytes: int

@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int
    min_chars: int
    overlap_chars: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    query_prefix: str
    batch_size: int


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    candidate_k: int
    rrf_k: int


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    temperature: float
    num_ctx: int
    timeout_seconds: float

@dataclass(frozen=True)
class AppConfig:
    site: SiteConfig
    available_sites:tuple[str, ...]
    crawler: CrawlerConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    data_root: Path

    @property
    def site_dir(self) -> Path:
        """Folder holding everything stored for the selected site."""
        return self.data_root / self.site.site_id


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid section '{key}'.")
    return value

def _string(section: dict, key: str, where: str, allow_empty: bool = False) -> str:
    value = section.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"'{where}.{key}' must be a non-empty string, got {value!r}.")
    return value if allow_empty else value.strip()

def _positive_int(section: dict, key: str, where: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{where}.{key}' must be a positive integer, got {value!r}.")
    return value

def _non_negative_int(section: dict, key: str, where: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"'{where}.{key}' must be a non-negative integer, got {value!r}.")
    return value

def _non_negative_number(section: dict, key: str, where: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigError(f"'{where}.{key}' must be a non-negative number, got {value!r}.")
    return float(value)

def _parse_site(site_id: str, raw: dict) -> SiteConfig:
    where = f"sites.{site_id}"
    if not _SITE_ID_PATTERN.match(site_id):
        raise ConfigError(f"Site id '{site_id}' must use lowercase letters, digits, '-' or '_'.")
    if not isinstance(raw, dict):
        raise ConfigError(f"'{where}' must be a mapping.")

    base_url = _string(raw, "base_url", where)
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigError(f"'{where}.base_url' must be an http(s) URL, got {base_url!r}.")

    allowed_domain = _string(raw, "allowed_domain", where).lower()
    if parsed.hostname.lower() != allowed_domain:
        raise ConfigError(f"'{where}.base_url' host must equal allowed_domain '{allowed_domain}'.")

    prefix = _string(raw, "allowed_path_prefix", where)
    if not prefix.startswith("/"):
        raise ConfigError(f"'{where}.allowed_path_prefix' must start with '/'.")
    if not (parsed.path or "/").startswith(prefix):
        raise ConfigError(f"'{where}.base_url' must lie inside allowed_path_prefix '{prefix}'.")

    excludes = raw.get("exclude_patterns", [])
    if not isinstance(excludes, list) or not all(isinstance(p, str) for p in excludes):
        raise ConfigError(f"'{where}.exclude_patterns' must be a list of strings.")
    for pattern in excludes:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(f"Invalid regex in '{where}.exclude_patterns': {pattern!r} ({exc}).")

    return SiteConfig(
        site_id=site_id,
        name=_string(raw, "name", where),
        base_url=base_url,
        allowed_domain=allowed_domain,
        allowed_path_prefix=prefix,
        max_pages=_positive_int(raw, "max_pages", where),
        exclude_patterns=tuple(excludes),
    )

def _parse_chunking(raw: dict) -> ChunkingConfig:
    cfg = ChunkingConfig(
        max_chars=_positive_int(raw, "max_chars", "chunking"),
        min_chars=_positive_int(raw, "min_chars", "chunking"),
        overlap_chars=_non_negative_int(raw, "overlap_chars", "chunking"),
    )
    if cfg.min_chars > cfg.max_chars:
        raise ConfigError("'chunking.min_chars' cannot exceed 'chunking.max_chars'.")
    if cfg.overlap_chars >= cfg.max_chars:
        raise ConfigError("'chunking.overlap_chars' must be smaller than 'chunking.max_chars'.")
    return cfg

def _parse_retrieval(raw: dict) -> RetrievalConfig:
    cfg = RetrievalConfig(
        top_k=_positive_int(raw, "top_k", "retrieval"),
        candidate_k=_positive_int(raw, "candidate_k", "retrieval"),
        rrf_k=_positive_int(raw, "rrf_k", "retrieval"),
    )
    if cfg.candidate_k < cfg.top_k:
        raise ConfigError("'retrieval.candidate_k' must be at least 'retrieval.top_k'.")
    return cfg

def load_config(path: Path | str | None = None, site_id: str | None = None) -> AppConfig:
    """Load the config file.

    `site_id` overrides `active_site`, so a new site can be crawled and indexed
    without switching the chatbot away from the current one.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a YAML mapping at the top level.")

    sites_raw = _section(raw, "sites")
    selected = site_id or _string(raw, "active_site", "")
    if selected not in sites_raw:
        raise ConfigError(f"Site '{selected}' is not defined under 'sites'.")

    crawler_raw = _section(raw, "crawler")
    embedding_raw = _section(raw, "embedding")
    llm_raw = _section(raw, "llm")

    data_root = Path(raw.get("data_dir", "data"))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    return AppConfig(
        site=_parse_site(selected, sites_raw[selected]),
        available_sites=tuple(sites_raw.keys()),
        crawler=CrawlerConfig(
            user_agent=_string(crawler_raw, "user_agent", "crawler"),
            request_delay_seconds=_non_negative_number(crawler_raw, "request_delay_seconds", "crawler"),
            timeout_seconds=_non_negative_number(crawler_raw, "timeout_seconds", "crawler"),
            max_retries=_non_negative_int(crawler_raw, "max_retries", "crawler"),
            max_response_bytes=_positive_int(crawler_raw, "max_response_bytes", "crawler"),
        ),
        chunking=_parse_chunking(_section(raw, "chunking")),
        embedding=EmbeddingConfig(
            model_name=_string(embedding_raw, "model_name", "embedding"),
            query_prefix=_string(embedding_raw, "query_prefix", "embedding", allow_empty=True),
            batch_size=_positive_int(embedding_raw, "batch_size", "embedding"),
        ),
        retrieval=_parse_retrieval(_section(raw, "retrieval")),
        llm=LLMConfig(
            model=_string(llm_raw, "model", "llm"),
            base_url=_string(llm_raw, "base_url", "llm").rstrip("/"),
            temperature=_non_negative_number(llm_raw, "temperature", "llm"),
            num_ctx=_positive_int(llm_raw, "num_ctx", "llm"),
            timeout_seconds=_non_negative_number(llm_raw, "timeout_seconds", "llm"),
        ),
        data_root=data_root,
    )

if __name__ == "__main__":
    cfg = load_config()
    print(f"Active site : {cfg.site.site_id} ({cfg.site.name})")
    print(f"Start URL   : {cfg.site.base_url}")
    print(f"Scope       : {cfg.site.allowed_domain}{cfg.site.allowed_path_prefix}")
    print(f"Data folder : {cfg.site_dir}")
    print(f"All sites   : {', '.join(cfg.available_sites)}")