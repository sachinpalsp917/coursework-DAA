"""Turn a documentation pages HTML into clean text, split by section heading."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import PreformattedString

# Where the documentation text lives, most specific first. The first selector
# matches the pydata Sphinx theme used by scikit-learn, the second the PyTorch theme.
CONTENT_SELECTORS = (
    "article.bd-article",
    "div[itemprop='articleBody']",
    "div[role='main']",
    "main",
    "article",
)

# Page furniture inside the content area that is not documentation text.
NOISE_SELECTORS = (
    "script", "style", "noscript", "template", "img", "svg", "button", "form", "nav",
    "a.headerlink",                        # the '#' permalink after every heading
    ".viewcode-link",                      # "[source]" links on API pages
    ".sphx-glr-download",                  # "Download Jupyter notebook" boxes
    ".sphx-glr-download-link-note",        # "Go to the end to download the full example code"
    ".sphx-glr-footer", ".sphx-glr-signature", ".binder-badge",
    ".sphx-glr-timing",                    # "Total running time of the script"
    ".sphx-glr-script-out",                # printed output of example scripts (long, noisy logs)
    ".sphx-glr-thumbnails",                # gallery thumbnails listing other pages
    ".prev-next-area", ".bd-toc",
)

HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")

# Elements that start a new line. Everything else (a, code, span, em ...) is inline.
BLOCK_TAGS = {
    "address", "blockquote", "caption", "dd", "details", "div", "dl", "dt", "figcaption",
    "figure", "footer", "header", "hr", "li", "ol", "p", "summary", "table", "tbody",
    "thead", "tfoot", "tr", "ul", *HEADING_TAGS,
}

CODE_FENCE = "```"

@dataclass(frozen=True)
class Section:
    path: tuple[str, ...]
    text: str

@dataclass(frozen=True)
class ExtractedPage:
    title: str
    sections: tuple[Section, ...]
    content_hash: str
    char_count: int

def _inline_text(tag: Tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ")).strip()

def _render(node: Tag, parts: list[str], subsections: list[Tag], skip: Tag | None) -> None:
    """Append the text of `node` to `parts`, keeping code blocks verbatim.

    Nested <section> elements are not rendered; they are collected in
    `subsections` so each one becomes its own Section. `skip` is the heading
    already used as the section title.
    """
    for child in node.children:
        if isinstance(child, NavigableString):
            # Comments, doctypes and CDATA are PreformattedString subclasses.
            if not isinstance(child, PreformattedString):
                parts.append(re.sub(r"\s+", " ", str(child)))
            continue
        if not isinstance(child, Tag) or child is skip:
            continue
        if child.name == "section":
            subsections.append(child)
        elif child.name == "pre":
            code = child.get_text().strip("\n")
            parts.append(f"\n{CODE_FENCE}\n{code}\n{CODE_FENCE}\n")
        elif child.name == "br":
            parts.append("\n")
        else:
            is_block = child.name in BLOCK_TAGS
            if is_block:
                parts.append("\n")
            if "classifier" in child.get("class", []):
                parts.append(" : ")
            _render(child, parts, subsections, skip)
            if child.name in ("td", "th"):
                parts.append(" | ")
            if is_block:
                parts.append("\n")

def _clean(text: str) -> str:
    """Tidy whitespace line by line, leaving fenced code blocks untouched."""
    lines: list[str] = []
    in_code = False
    for line in text.split("\n"):
        if line.strip() == CODE_FENCE:
            in_code = not in_code
            lines.append(CODE_FENCE)
        elif in_code:
            lines.append(line.rstrip())
        else:
            line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
            if line.endswith(" |"):      # trailing separator after the last table cell
                line = line[:-2]
            if line:
                lines.append(line)
    return "\n".join(lines).strip()

def _collect(node: Tag, path: list[str], sections: list[Section], skip: Tag | None = None) -> None:
    parts: list[str] = []
    subsections: list[Tag] = []
    _render(node, parts, subsections, skip)
    text = _clean("".join(parts))
    if text:
        sections.append(Section(path=tuple(path), text=text))
    for sub in subsections:
        heading = sub.find(HEADING_TAGS, recursive=False)
        title = _inline_text(heading) if heading else ""
        _collect(sub, path + [title] if title else path, sections, skip=heading)

def _hash_sections(sections: list[Section]) -> str:
    digest = hashlib.sha256()
    for section in sections:
        digest.update(" > ".join(section.path).encode("utf-8"))
        digest.update(b"\n")
        digest.update(section.text.encode("utf-8"))
        digest.update(b"\n\n")
    return digest.hexdigest()

def extract_page(html: str, url: str) -> ExtractedPage | None:
    """Extract the main documentation text of a page.

    Returns None when no content area or no text is found, so the caller can
    log an extraction failure instead of storing an empty page.
    """
    soup = BeautifulSoup(html, "lxml")
    content = None
    for selector in CONTENT_SELECTORS:
        content = soup.select_one(selector)
        if content is not None:
            break
    if content is None:
        return None

    for element in content.select(", ".join(NOISE_SELECTORS)):
        element.decompose()

    sections: list[Section] = []
    _collect(content, [], sections)
    if not sections:
        return None

    h1 = content.find("h1")
    if h1 is not None and _inline_text(h1):
        title = _inline_text(h1)
    elif soup.title is not None:
        # "<page> — scikit-learn 1.9.1 documentation" -> "<page>"
        title = _inline_text(soup.title).split(" — ")[0]
    else:
        title = url

    return ExtractedPage(
        title=title,
        sections=tuple(sections),
        content_hash=_hash_sections(sections),
        char_count=sum(len(s.text) for s in sections),
    )