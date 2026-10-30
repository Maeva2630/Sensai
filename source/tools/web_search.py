"""Web search tool: DuckDuckGo results parsed into title, URL and snippet.

No API key is needed. The HTML endpoint is not an official API, so every failure
(network, blocked request, changed page) is reported as a `ToolError` and the
assistant falls back to a normal answer. Reading a result page is left to the
`fetch` MCP server, which the model calls with one of the returned URLs.
"""

import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List
from urllib.parse import parse_qs, urlparse

import requests

from tools.base import Tool, ToolError
from tools.safety import personal_data_warning

SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (compatible; SensAI/1.0)"
MAX_RESULTS = 5
MAX_QUERY_CHARS = 200
MAX_SNIPPET_CHARS = 300
TIMEOUT_SECONDS = 10.0
MIN_SECONDS_BETWEEN_SEARCHES = 2.0

CITATION_NOTE = (
    "Answer using only these results and cite the URLs you used. "
    "If they do not answer the question, say so."
)
UNAVAILABLE_NOTE = "Answer without the web and tell the user the web could not be searched."


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class _ResultParser(HTMLParser):
    """Collects `result__a` (title and link) and `result__snippet` anchors in page order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[SearchResult] = []
        self._url = ""
        self._title: List[str] = []
        self._snippet: List[str] = []
        self._capture: str = ""

    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:
        if tag != "a":
            return
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if "result__a" in classes:
            self._flush()
            self._url = _clean_url(values.get("href") or "")
            self._capture = "title"
        elif "result__snippet" in classes:
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._capture = ""

    def handle_data(self, data: str) -> None:
        if self._capture == "title":
            self._title.append(data)
        elif self._capture == "snippet":
            self._snippet.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        title = _squash(self._title)
        if self._url and title:
            self.results.append(SearchResult(title, self._url, _squash(self._snippet)))
        self._url, self._title, self._snippet = "", [], []


def _squash(parts: List[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _clean_url(href: str) -> str:
    """Return the real target of a DuckDuckGo redirect link; drop ads and non-web links."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg")
        if not target:
            return ""
        href = target[0]
        parsed = urlparse(href)
    return href if parsed.scheme in ("http", "https") and parsed.netloc else ""


def parse_results(html: str, limit: int = MAX_RESULTS) -> List[SearchResult]:
    """Extract at most `limit` results from a DuckDuckGo HTML result page."""
    parser = _ResultParser()
    parser.feed(html)
    parser.close()
    return parser.results[:limit]


def format_results(query: str, results: List[SearchResult]) -> str:
    """Bounded, numbered result block followed by the citation instruction."""
    lines = [f'Web results for "{query}":']
    for number, result in enumerate(results, start=1):
        lines.append(f"{number}. {result.title}")
        lines.append(f"   URL: {result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet[:MAX_SNIPPET_CHARS]}")
    lines.append(CITATION_NOTE)
    return "\n".join(lines)


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web and return titles, URLs and snippets. Use it only for recent "
        "or external information the course documents cannot contain."
    )
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query."}},
        "required": ["query"],
    }
    untrusted = True

    _EXPLICIT_PATTERN = re.compile(
        r"(\b(sur|on|via)\s+(le\s+)?(web|internet)\b|\ben ligne\b|\bonline\b|\bgoogle\b|\bduckduckgo\b"
        r"|\b(search|look up|cherche\w*|recherche\w*)\b.{0,40}\b(web|internet|online|en ligne)\b)",
        re.IGNORECASE,
    )
    _RECENT_PATTERN = re.compile(
        r"\b(actualit\w*|news|latest|recent|r[ée]cent\w*|derni[èe]res?\s+(version|nouvelle|actu)\w*)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        timeout: float = TIMEOUT_SECONDS,
        max_results: int = MAX_RESULTS,
        min_interval: float = MIN_SECONDS_BETWEEN_SEARCHES,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._timeout = timeout
        self._max_results = max_results
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_search: float | None = None
        self._sources: List[str] = []

    def execute(self, arguments: Dict[str, Any]) -> str:
        query = str(arguments.get("query", "")).strip()[:MAX_QUERY_CHARS]
        if not query:
            raise ToolError("the search query is empty")
        html = self._download(query)
        results = parse_results(html, self._max_results)
        if not results:
            return f'No web results for "{query}". {UNAVAILABLE_NOTE}'
        for result in results:
            if result.url not in self._sources:
                self._sources.append(result.url)
        return format_results(query, results)

    def confirmation_prompt(self, arguments: Dict[str, Any]) -> str | None:
        query = str(arguments.get("query", "")).strip()[:MAX_QUERY_CHARS]
        warning = personal_data_warning(query)
        return f'Search the web for "{query}"? This text is sent to DuckDuckGo.{warning}'

    def take_sources(self) -> List[str]:
        """Return the URLs offered to the model since the last call, then forget them."""
        sources, self._sources = self._sources, []
        return sources

    def is_relevant_for(self, user_input: str) -> bool:
        """Offer the tool only when the request mentions the web or recent information."""
        return bool(self._EXPLICIT_PATTERN.search(user_input) or self._RECENT_PATTERN.search(user_input))

    def claims_request(self, user_input: str) -> bool:
        """An explicit web request skips document retrieval; a mere "recent" does not."""
        return bool(self._EXPLICIT_PATTERN.search(user_input))

    def _download(self, query: str) -> str:
        self._wait_for_turn()
        try:
            response = requests.get(
                SEARCH_URL,
                params={"q": query},
                headers={"User-Agent": USER_AGENT},
                timeout=self._timeout,
            )
        except requests.exceptions.RequestException as error:
            raise ToolError(f"web search is unavailable ({type(error).__name__}). {UNAVAILABLE_NOTE}") from error
        if response.status_code != 200:
            raise ToolError(
                f"web search is unavailable (HTTP {response.status_code}). {UNAVAILABLE_NOTE}"
            )
        return response.text

    def _wait_for_turn(self) -> None:
        """Basic rate limit: keep a minimum delay between two searches."""
        now = self._clock()
        if self._last_search is not None:
            remaining = self._min_interval - (now - self._last_search)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last_search = now
