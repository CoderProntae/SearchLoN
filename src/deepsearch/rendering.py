"""Safe Markdown rendering with Pygments highlighting for code blocks."""

from __future__ import annotations

import bleach
from bleach.css_sanitizer import CSSSanitizer
from markdown import Markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension

_ALLOWED_TAGS = {
    "a", "blockquote", "br", "code", "del", "div", "em", "h1", "h2", "h3", "h4",
    "h5", "h6", "hr", "li", "ol", "p", "pre", "span", "strong", "table", "tbody",
    "td", "th", "thead", "tr", "ul",
}
_ALLOWED_ATTRIBUTES = {
    "a": ["href", "title"],
    "code": ["class"],
    "div": ["class"],
    "pre": ["class"],
    "span": ["class", "style"],
    "th": ["align"],
    "td": ["align"],
}
_CSS_SANITIZER = CSSSanitizer(
    allowed_css_properties={"color", "background-color", "font-weight", "font-style", "text-decoration"}
)


def render_markdown(text: str) -> str:
    """Convert Markdown to sanitized HTML suitable for QTextBrowser (no scripts/images)."""
    converter = Markdown(
        extensions=[
            FencedCodeExtension(),
            TableExtension(),
            CodeHiliteExtension(guess_lang=True, noclasses=True, use_pygments=True),
        ]
    )
    raw_html = converter.convert(text)
    safe_html = bleach.clean(
        raw_html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        protocols={"http", "https"},
        strip=True,
        strip_comments=True,
        css_sanitizer=_CSS_SANITIZER,
    )
    style = (
        "<style>body { color: #e5e7eb; font-family: Segoe UI, sans-serif; }"
        "pre { background: #10141b; padding: 8px; border-radius: 6px; }"
        "code { font-family: Consolas, 'Cascadia Code', monospace; }"
        "table { border-collapse: collapse; }"
        "th, td { border: 1px solid #394453; padding: 4px 7px; }</style>"
    )
    return f"<!doctype html><html><head>{style}</head><body>{safe_html}</body></html>"
