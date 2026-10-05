import pytest

from deepsearch.web import (
    WebSearchError,
    _DuckDuckGoParser,
    _ReadableHTMLParser,
    validate_public_url,
)


def test_duckduckgo_result_parser_extracts_title_url_and_snippet() -> None:
    html = """
    <div class="result results_links">
      <h2><a class="result__a" href="https://example.org/page">Example &amp; Docs</a></h2>
      <a class="result__snippet">A useful <b>snippet</b> about the page.</a>
    </div>
    """
    parser = _DuckDuckGoParser()
    parser.feed(html)
    parser.close()

    assert parser.results == [
        {
            "title": "Example & Docs",
            "url": "https://example.org/page",
            "snippet": "A useful snippet about the page.",
        }
    ]


def test_readable_html_omits_scripts_and_navigation() -> None:
    parser = _ReadableHTMLParser()
    parser.feed(
        "<html><head><title>Test page</title><script>hidden()</script></head>"
        "<body><nav>menu should disappear</nav><main><h1>Hello</h1>"
        "<p>Readable <strong>content</strong>.</p></main></body></html>"
    )
    parser.close()
    text = " ".join(" ".join(parser.parts).split())

    assert "Hello" in text
    assert "Readable content" in text
    assert "hidden" not in text
    assert "menu should disappear" not in text
    assert "Test page" in " ".join(parser.title_parts)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/admin",
        "http://10.0.0.12/",
        "http://user:password@example.com/",
        "http://localhost:11434/",
    ],
)
def test_public_url_guard_rejects_local_or_unsafe_urls(url: str) -> None:
    with pytest.raises(WebSearchError):
        validate_public_url(url)
