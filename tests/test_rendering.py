from deepsearch.rendering import render_markdown


def test_markdown_renders_tables_and_highlighted_fenced_code() -> None:
    html = render_markdown(
        "| Model | Version |\n|---|---|\n| Qwen | 3 |\n\n```python\nprint('hello')\n```"
    )

    assert "<table>" in html
    assert "<pre>" in html
    assert "print" in html
    assert "style=\"color:" in html


def test_markdown_sanitization_removes_active_html_and_unsafe_links() -> None:
    html = render_markdown(
        '<script>alert("x")</script> [bad](javascript:alert(1)) [safe](https://example.org)'
    )

    assert "<script>" not in html
    assert "href=\"javascript:" not in html
    assert "href=\"https://example.org\"" in html
