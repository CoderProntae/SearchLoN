"""Privacy-conscious DuckDuckGo HTML search and safe public-page extraction."""

from __future__ import annotations

import ipaddress
import re
import socket
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import ClassVar
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class WebSearchError(RuntimeError):
    """Search, network, parsing, or safe-fetch error."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class WebPage:
    title: str
    url: str
    text: str


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._div_depth = 0
        self._field: str | None = None
        self._field_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "div" and "result" in classes and self._current is None:
            self._current = {"title": "", "url": "", "snippet": ""}
            self._div_depth = 1
            return
        if self._current is None:
            return
        if tag == "div":
            self._div_depth += 1
        if tag == "a" and "result__a" in classes:
            self._current["url"] = attributes.get("href") or ""
            self._field, self._field_tag = "title", "a"
        elif "result__snippet" in classes:
            self._field, self._field_tag = "snippet", tag

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._field:
            self._current[self._field] += data

    def handle_endtag(self, tag: str) -> None:
        if self._field_tag == tag:
            self._field = None
            self._field_tag = None
        if self._current is not None and tag == "div":
            self._div_depth -= 1
            if self._div_depth <= 0:
                self._flush()

    def close(self) -> None:
        super().close()
        if self._current is not None:
            self._flush()

    def _flush(self) -> None:
        assert self._current is not None
        item = {key: " ".join(value.split()) for key, value in self._current.items()}
        if item["title"] and item["url"]:
            self.results.append(item)
        self._current = None
        self._div_depth = 0
        self._field = None
        self._field_tag = None


class _ReadableHTMLParser(HTMLParser):
    _DROP: ClassVar[frozenset[str]] = frozenset(
        {"head", "script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form"}
    )
    _BLOCK: ClassVar[frozenset[str]] = frozenset(
        {
            "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "h1", "h2",
            "h3", "h4", "h5", "h6", "li", "main", "ol", "p", "pre", "section", "table",
            "td", "th", "tr", "ul",
        }
    )
    _VOID: ClassVar[frozenset[str]] = frozenset(
        {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_stack: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if self._skip_stack:
            if tag not in self._VOID:
                self._skip_stack.append(tag)
            return
        if tag in self._DROP:
            self._skip_stack.append(tag)
            return
        if tag in self._BLOCK:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if self._skip_stack:
            # Tolerate minor malformed nesting without leaking hidden page content.
            if tag in self._skip_stack:
                index = len(self._skip_stack) - 1 - self._skip_stack[::-1].index(tag)
                del self._skip_stack[index:]
            return
        if tag in self._BLOCK:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if not self._skip_stack:
            self.parts.append(data)


def _normalize_extracted_text(text: str) -> str:
    normalized = " ".join(text.split())
    return re.sub(r"\s+([,.;:!?])", r"\1", normalized)


def validate_public_url(url: str) -> str:
    """Reject non-HTTP, credential-bearing and non-public destinations (basic SSRF guard)."""
    try:
        parsed = urlsplit(url.strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise WebSearchError("Yalnızca http:// veya https:// bağlantıları taranabilir.")
        if parsed.username or parsed.password:
            raise WebSearchError("Kullanıcı adı/şifre içeren bağlantılar güvenlik nedeniyle engellendi.")
        host = parsed.hostname.rstrip(".").lower()
        if host == "localhost" or host.endswith((".localhost", ".local")):
            raise WebSearchError("Yerel ağ adresleri güvenlik nedeniyle engellendi.")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            try:
                resolved = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
            except OSError as exc:
                raise WebSearchError(f"Web adresinin DNS adı çözümlenemedi: {host}") from exc
            addresses = {item[4][0].split("%", 1)[0] for item in resolved}
            if not addresses:
                raise WebSearchError("Web adresi için IP bulunamadı.")
            for resolved_ip in addresses:
                try:
                    ip = ipaddress.ip_address(resolved_ip)
                except ValueError as exc:
                    raise WebSearchError("Web adresi geçersiz IP döndürdü.") from exc
                if not ip.is_global:
                    raise WebSearchError("Özel veya yerel ağ adresleri güvenlik nedeniyle engellendi.")
        else:
            if not address.is_global:
                raise WebSearchError("Özel veya yerel ağ adresleri güvenlik nedeniyle engellendi.")
        return url
    except ValueError as exc:
        raise WebSearchError("Web adresi biçimi geçersiz.") from exc


class WebSearchClient:
    def __init__(
        self,
        endpoint: str = "https://html.duckduckgo.com/html/",
        timeout_seconds: int = 12,
        retry_count: int = 1,
        user_agent: str = "DeepSearchResearchAI/0.1 (desktop research client)",
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.user_agent = user_agent

    def search(self, query: str, result_count: int = 6) -> list[SearchResult]:
        query = " ".join(query.split())[:500]
        if not query:
            return []
        endpoint_parts = urlsplit(self.endpoint)
        parameters = parse_qsl(endpoint_parts.query, keep_blank_values=True)
        parameters = [(key, value) for key, value in parameters if key != "q"]
        parameters.append(("q", query))
        search_url = urlunsplit(
            (endpoint_parts.scheme, endpoint_parts.netloc, endpoint_parts.path, urlencode(parameters), "")
        )
        html = self._download(search_url, max_bytes=2_000_000)
        parser = _DuckDuckGoParser()
        parser.feed(html.decode("utf-8", errors="replace"))
        parser.close()

        results: list[SearchResult] = []
        seen: set[str] = set()
        for item in parser.results:
            url = self._normalize_result_url(item["url"])
            if not url or url in seen:
                continue
            try:
                validate_public_url(url)
            except WebSearchError:
                continue
            seen.add(url)
            results.append(SearchResult(item["title"], url, item["snippet"]))
            if len(results) >= max(1, min(int(result_count), 12)):
                break
        return results

    def fetch_page(self, url: str, max_chars: int = 12_000) -> WebPage:
        safe_url = validate_public_url(url)
        raw = self._download(safe_url, max_bytes=2_000_000)
        parser = _ReadableHTMLParser()
        parser.feed(raw.decode("utf-8", errors="replace"))
        parser.close()
        title = _normalize_extracted_text(" ".join(parser.title_parts))[:300]
        text = _normalize_extracted_text(" ".join(parser.parts))
        if not text:
            raise WebSearchError("Sayfadan okunabilir metin çıkarılamadı.")
        return WebPage(title=title or safe_url, url=safe_url, text=text[:max(500, max_chars)])

    @staticmethod
    def _normalize_result_url(url: str) -> str:
        url = url.strip()
        if url.startswith("//"):
            url = "https:" + url
        parsed = urlsplit(url)
        if parsed.hostname and parsed.hostname.lower().endswith("duckduckgo.com") and parsed.path == "/l/":
            for key, value in parse_qsl(parsed.query, keep_blank_values=True):
                if key == "uddg":
                    return value
        return url

    def _download(self, url: str, max_bytes: int) -> bytes:
        current_url = url
        for redirect_count in range(5):
            validate_public_url(current_url)
            request = Request(
                current_url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.2",
                    "Accept-Encoding": "identity",
                },
            )
            opener = build_opener(_NoRedirect(),)
            attempt = 0
            while True:
                try:
                    with opener.open(request, timeout=self.timeout_seconds) as response:
                        content_type = response.headers.get_content_type().lower()
                        if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                            raise WebSearchError(f"Desteklenmeyen sayfa türü: {content_type}")
                        return response.read(max_bytes + 1)[:max_bytes]
                except HTTPError as exc:
                    if exc.code in {301, 302, 303, 307, 308}:
                        location = exc.headers.get("Location")
                        exc.close()
                        if not location:
                            raise WebSearchError("Web sitesi geçersiz yönlendirme yanıtı verdi.") from exc
                        next_url = urljoin(current_url, location)
                        old_scheme = urlsplit(current_url).scheme
                        if old_scheme == "https" and urlsplit(next_url).scheme != "https":
                            raise WebSearchError("HTTPS bağlantısının güvensiz yönlendirmesi engellendi.")
                        current_url = next_url
                        break
                    if (exc.code >= 500 or exc.code in {408, 425, 429}) and attempt < self.retry_count:
                        exc.close()
                        attempt += 1
                        time.sleep(min(0.2 * attempt, 0.6))
                        continue
                    message = exc.read(1500).decode("utf-8", errors="replace").strip()
                    exc.close()
                    raise WebSearchError(f"Web sunucusu HTTP {exc.code}: {message or exc.reason}") from exc
                except (URLError, TimeoutError, OSError) as exc:
                    if attempt < self.retry_count:
                        attempt += 1
                        time.sleep(min(0.2 * attempt, 0.6))
                        continue
                    reason = getattr(exc, "reason", exc)
                    raise WebSearchError(f"Web sayfasına erişilemedi: {reason}") from exc
            else:
                continue
            # A validated redirect was followed; validate the destination on the next pass.
            continue
        raise WebSearchError("Web sitesi çok fazla yönlendirme yaptı.")
