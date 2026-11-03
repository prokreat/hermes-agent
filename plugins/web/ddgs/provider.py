"""DuckDuckGo search via the optional ``ddgs`` package (search only, no key). ``is_available()``
reflects package importability; the plugin registers either way so ``hermes tools`` can offer to
install it. Isolation: ``ddgs``/``primp`` can block inside native code while holding the GIL, so a
thread-pool ``future.result(timeout=…)`` cap can never fire and Ctrl+C/SIGTERM freeze the process —
each search runs in a disposable child process the parent can terminate/kill.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os
import subprocess
import sys
import time
from html.parser import HTMLParser
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx

from plugins.web._common import (
    BaseWebSearchProvider,
    document,
    page_error,
    run_extract,
    search_fail,
    search_ok,
    setup_schema,
    title_hit,
    with_origin_status,
)

logger = logging.getLogger(__name__)

# Hard wall-clock cap per search: ``DDGS(timeout=…)`` only bounds individual HTTP requests;
# ddgs's multi-engine retry loop has no overall cap, so a rate-limited response could
# otherwise hang the shared agent loop indefinitely.
# Enforce a hard cap here by killing a disposable worker process (#68096).
_SEARCH_TIMEOUT_SECS = 30
_POLL_INTERVAL_SECS = 0.1  # parent stdout / interrupt-flag poll cadence
_TERMINATE_GRACE_SECS = 1.0  # wait after terminate() before escalating to kill()
_test_hook: Optional[str] = None  # test-only hook forwarded to the child (see _search_worker.py)
_last_worker_proc: Optional[subprocess.Popen] = None  # last worker Popen (test reap checks)


class _SearchInterrupted(Exception):
    """Raised when tools.interrupt.is_interrupted() trips during a search wait."""


def _run_ddgs_search(query: str, safe_limit: int) -> list[dict[str, Any]]:
    """Blocking ddgs query → normalized hits (module-level: the child worker imports it,
    tests patch it for in-process runs).

    ``DDGS(timeout=…)`` bounds each individual HTTP request; the overall wall-clock cap is enforced by the
    parent via process timeout (#68096).
    """
    from ddgs import DDGS  # type: ignore
    results: list[dict[str, Any]] = []
    with DDGS(timeout=10) as client:
        for i, hit in enumerate(client.text(query, max_results=safe_limit)):
            if i >= safe_limit:
                break
            results.append(title_hit(str(hit.get("title", "")), str(hit.get("href") or hit.get("url") or ""), str(hit.get("body", "")), i + 1))
    return results


def _plugins_path_entry() -> str:
    """``sys.path`` entry that makes ``import plugins`` work in the child (live package
    location first; correct for source checkouts and site-packages)."""
    try:
        import plugins as plugins_pkg
        if pkg_file := getattr(plugins_pkg, "__file__", None):
            return os.path.dirname(os.path.dirname(os.path.abspath(pkg_file)))
    except Exception:
        pass
    return os.path.abspath(os.path.join(__file__, *([os.pardir] * 4)))


def _terminate_and_reap(proc: Optional[subprocess.Popen], *, grace: float = _TERMINATE_GRACE_SECS) -> None:
    """Terminate a worker, escalate to kill, and wait so no orphan remains. Does not close
    the parent's pipe ends — closing stdout while another thread is blocked in ``read()``
    deadlocks on some platforms; the caller drains first."""
    if proc is None:
        return
    alive = False

    def _wait_until_dead() -> bool:
        deadline = time.monotonic() + grace
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return proc.poll() is not None

    try:
        for escalate in (proc.terminate, proc.kill):
            if proc.poll() is None:
                escalate()
                alive = not _wait_until_dead()
        if alive:
            logger.warning("DDGS worker pid=%s did not exit after kill", proc.pid)
    except Exception as exc:
        logger.debug("DDGS worker reap error: %s", exc)


def _spawn_worker(env: dict[str, str]) -> subprocess.Popen:
    """Start ``_search_worker.py`` as a script with ``plugins`` importable. Running as a
    script puts ``plugins/web/ddgs/`` on ``sys.path[0]``, breaking ``import plugins...``,
    so the real package location is prepended to PYTHONPATH."""

    # pm store PATH: the worker runs under the STORE python, whose third-party
    # imports (ddgs/primp) arrive via the launcher-composed PYTHONPATH. The
    # sanitizer strips Hermes-owned entries (cross-version protection); this
    # worker is the SAME interpreter, so merge the ambient PYTHONPATH back in.
    _ambient_pp = os.environ.get("PYTHONPATH")
    if _ambient_pp:
        _current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            _ambient_pp + os.pathsep + _current if _current else _ambient_pp
        )
    child_pythonpath = env.get("PYTHONPATH", "")
    path_entry = _plugins_path_entry()
    if path_entry and path_entry not in child_pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = path_entry + os.pathsep + child_pythonpath if child_pythonpath else path_entry
    worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_search_worker.py")
    # Own process group/session. The 30 s cap terminates then kills the worker's pid only, not its group.
    # detached-spawn: ok — awaited under a 30 s cap that kills the worker; an owner killed mid-search leaves it running until that search returns (unbounded if primp hangs)
    extra_kwargs: dict[str, Any] = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
    # stdin/stdout/stderr stay explicit keyword args so scripts/check_subprocess_stdin.py sees them
    # (TUI gateway inherits stdin). stderr=DEVNULL: a chatty child would deadlock a stdout-only drain.
    return subprocess.Popen(
        [sys.executable, worker_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=env, text=True, encoding="utf-8", errors="replace", **extra_kwargs,
    )


def _parse_envelope(raw: str, proc: subprocess.Popen) -> list[dict[str, Any]]:
    """Decode the worker's stdout envelope; raise ``RuntimeError`` on any malformed shape."""
    raw = raw.strip()
    if not raw:
        raise RuntimeError(f"DDGS worker exited without a result (code={proc.poll()})")
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DDGS worker returned invalid JSON: {raw[:200]!r}") from exc
    if not isinstance(envelope, dict):
        raise RuntimeError(f"DDGS worker returned an invalid envelope: {envelope!r}")
    if not envelope.get("ok"):
        raise RuntimeError(str(envelope.get("error") or "DDGS worker failed"))
    results = envelope.get("results") or []
    if not isinstance(results, list):
        raise RuntimeError("DDGS worker returned non-list results")
    return results


def _run_ddgs_search_bounded(query: str, safe_limit: int) -> list[dict[str, Any]]:
    """Run ``_run_ddgs_search`` in a disposable process with a hard deadline. The parent
    never joins a child that may be in native code holding *its* GIL — it polls a
    communicator thread and, on timeout/interrupt, kills the OS process.
    Raises ``TimeoutError``, ``_SearchInterrupted``, or ``RuntimeError``."""
    from tools.interrupt import is_interrupted  # lazy: keep plugin import light
    from tools.environments.local import _sanitize_subprocess_env
    global _last_worker_proc
    request: dict[str, Any] = {"query": query, "safe_limit": safe_limit}
    env = _sanitize_subprocess_env(dict(os.environ))
    if _test_hook:
        request["test_hook"] = _test_hook
        env["HERMES_DDGS_ALLOW_TEST_HOOKS"] = "1"
    proc = _last_worker_proc = _spawn_worker(env)
    # ``communicate`` runs in a side thread so the parent can poll interrupt /
    # deadline without blocking; killing the child unblocks it.
    pool = cf.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(proc.communicate, json.dumps(request))
    interrupted, done, raw = False, False, ""
    try:
        deadline = time.monotonic() + _SEARCH_TIMEOUT_SECS
        while not done and not (interrupted := is_interrupted()) and (remaining := deadline - time.monotonic()) > 0:
            try:
                raw, done = fut.result(timeout=min(_POLL_INTERVAL_SECS, remaining))[0] or "", True
            except cf.TimeoutError:
                pass
    finally:
        _terminate_and_reap(proc)
        # After kill, communicate should return promptly; don't block forever.
        if not fut.done():
            try:
                raw = raw or fut.result(timeout=_TERMINATE_GRACE_SECS)[0] or ""
            except Exception:
                pass
        pool.shutdown(wait=False, cancel_futures=True)
    if interrupted:
        raise _SearchInterrupted("DuckDuckGo search interrupted")
    if not done:
        raise TimeoutError(f"DuckDuckGo search timed out after {_SEARCH_TIMEOUT_SECS}s")
    return _parse_envelope(raw, proc)


_LOCAL_EXTRACT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_LOCAL_EXTRACT_TIMEOUT_SECS = 20.0
_SKIP_HTML_TAGS = frozenset({"script", "style", "noscript", "svg", "template", "head"})
_BLOCK_HTML_TAGS = frozenset({
    "p", "div", "section", "article", "main", "header", "footer", "aside", "nav",
    "blockquote", "pre", "ul", "ol", "table", "tr", "hr", "br",
    "h1", "h2", "h3", "h4", "h5", "h6",
})


class _LocalHtmlExtractor(HTMLParser):
    """Lightweight stdlib HTML-to-Markdown/text extractor with zero cloud dependencies."""

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0
        self._in_pre = False
        self._link_href: Optional[str] = None
        self._link_text: list[str] = []
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t == "title":
            self._in_title = True
            return
        if t in _SKIP_HTML_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(t[1])
            self._chunks.append(f"\n\n{'#' * level} ")
        elif t == "li":
            self._chunks.append("\n- ")
        elif t == "pre":
            self._in_pre = True
            self._chunks.append("\n```\n")
        elif t == "code" and not self._in_pre:
            self._chunks.append("`")
        elif t == "a":
            href = dict(attrs).get("href") or ""
            if href and not href.startswith(("javascript:", "mailto:", "#")):
                self._link_href = urljoin(self.base_url, href.strip())
                self._link_text = []
        elif t in _BLOCK_HTML_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == "title":
            self._in_title = False
            return
        if t in _SKIP_HTML_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if t == "pre":
            self._in_pre = False
            self._chunks.append("\n```\n")
        elif t == "code" and not self._in_pre:
            self._chunks.append("`")
        elif t == "a" and self._link_href is not None:
            label = " ".join("".join(self._link_text).split())
            href = self._link_href
            self._link_href = None
            self._link_text = []
            if label and href:
                self._chunks.append(f"[{label}]({href})")
            elif label:
                self._chunks.append(label)
        elif t in _BLOCK_HTML_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._skip_depth > 0:
            return
        text = data if self._in_pre else " ".join(data.split())
        if not text:
            return
        if self._link_href is not None:
            self._link_text.append(text)
        else:
            if self._chunks and not self._in_pre and not self._chunks[-1].endswith((" ", "\n", "`", "#")):
                self._chunks.append(" ")
            self._chunks.append(text)

    def get_result(self) -> tuple[str, str]:
        title = " ".join("".join(self.title_parts).split()).strip()
        raw = "".join(self._chunks)
        lines = [line.rstrip() for line in raw.splitlines()]
        cleaned: list[str] = []
        blank_run = 0
        for line in lines:
            if not line.strip():
                blank_run += 1
                if blank_run <= 2:
                    cleaned.append("")
            else:
                blank_run = 0
                cleaned.append(line)
        return title, "\n".join(cleaned).strip()


def _http_fetch_url(url: str) -> Any:
    """Fetch *url* using ``primp`` Chrome TLS/HTTP2 impersonation when available (bypasses TLS
    fingerprint bot walls such as TechPowerUp), falling back to ``httpx.get``."""
    # If a unit test monkeypatched httpx.get, honor the monkeypatch directly.
    if getattr(httpx.get, "__module__", "").startswith("httpx"):
        try:
            import primp  # type: ignore

            client = primp.Client(
                impersonate="chrome",
                impersonate_os="linux",
                follow_redirects=True,
                timeout=_LOCAL_EXTRACT_TIMEOUT_SECS,
            )
            return client.get(url)
        except Exception as exc:
            logger.debug("primp fetch fell back to httpx for %s: %s", url, exc)
    return httpx.get(
        url,
        headers=_LOCAL_EXTRACT_HEADERS,
        follow_redirects=True,
        timeout=_LOCAL_EXTRACT_TIMEOUT_SECS,
    )


def _fetch_and_extract_url(url: str) -> dict[str, Any]:
    """Fetch a single URL locally (via primp Chrome TLS impersonation or httpx) and convert
    HTML/PDF/JSON/text into an extract document."""
    try:
        resp = _http_fetch_url(url)
    except Exception as exc:
        return page_error(url, f"Local HTTP extract failed: {exc}")

    final_url = str(getattr(resp, "url", None) or url)
    headers = getattr(resp, "headers", None) or {}
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "").lower()
    raw_bytes = getattr(resp, "content", b"") or b""
    raw_text = getattr(resp, "text", "") or ""
    title = final_url
    content = raw_text

    if (
        "application/pdf" in content_type
        or final_url.lower().split("?", 1)[0].endswith(".pdf")
        or raw_bytes.startswith(b"%PDF-")
    ):
        try:
            proc = subprocess.run(
                ["/usr/bin/pdftotext", "-", "-"],
                input=raw_bytes,
                capture_output=True,
                timeout=20,
            )
            pdf_text = proc.stdout.decode("utf-8", errors="replace").strip()
            if pdf_text:
                content = pdf_text
                first_line = next((ln.strip() for ln in pdf_text.splitlines() if ln.strip()), "")
                if first_line:
                    title = first_line[:160]
        except Exception as exc:
            logger.debug("Local pdftotext extraction failed for %s: %s", final_url, exc)
    elif "application/json" in content_type or final_url.lower().split("?", 1)[0].endswith(".json"):
        try:
            parsed = resp.json() if callable(getattr(resp, "json", None)) else json.loads(raw_text)
            content = json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            content = raw_text
    elif "html" in content_type or "<html" in raw_text[:1024].lower():
        parser = _LocalHtmlExtractor(base_url=final_url)
        try:
            parser.feed(raw_text)
            parser.close()
            extracted_title, extracted_body = parser.get_result()
            if extracted_title:
                title = extracted_title
            if extracted_body:
                content = extracted_body
        except Exception as exc:
            logger.debug("Local HTML parse fallback for %s: %s", final_url, exc)

    reason_phrase = getattr(resp, "reason_phrase", None)
    entry = document(final_url, title, content, source_url=url)
    return with_origin_status(entry, resp.status_code, reason_phrase)


class DDGSWebSearchProvider(BaseWebSearchProvider):
    """DuckDuckGo HTML-scrape search provider + local httpx URL extractor (no API key).
    ddgs errors surface as ``{"success": False, "error": ...}`` rather than raising."""

    NAME = "ddgs"
    DISPLAY_NAME = "DuckDuckGo (ddgs)"
    EXTRACT = True

    def is_available(self) -> bool:
        """True when ``ddgs`` is importable. Must NOT do network I/O — runs at
        tool-registration time and on every ``hermes tools`` paint."""
        try:
            import ddgs
            return True
        except ImportError:
            return False

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Run the search in a disposable child with a hard wall-clock timeout so a
        hung native ``primp`` call cannot freeze the Hermes process.

        See #36776, #68096.
        """
        if not self.is_available():
            return search_fail("ddgs package is not installed — run `pip install ddgs`")
        try:
            # max(1, …): defensive cap in case the package ignores its max_results hint.
            web_results = _run_ddgs_search_bounded(query, max(1, int(limit)))
        except TimeoutError:
            logger.warning("DDGS search timed out after %ds for query: %r", _SEARCH_TIMEOUT_SECS, query)
            return search_fail(
                f"DuckDuckGo search timed out after {_SEARCH_TIMEOUT_SECS}s — "
                "DuckDuckGo may be rate-limiting or slow. Try again later or switch to a different search provider."
            )
        except _SearchInterrupted:
            logger.info("DDGS search interrupted for query: %r", query)
            return search_fail("DuckDuckGo search interrupted")
        except Exception as exc:
            logger.warning("DDGS search error: %s", exc)
            return search_fail(f"DuckDuckGo search failed: {exc}")
        logger.info("DDGS search '%s': %d results (limit %d)", query, len(web_results), limit)
        return search_ok(web_results)

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """Extract URL content locally via httpx (HTML/JSON/text) with zero external cloud relays."""
        from tools.interrupt import is_interrupted

        def _body() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            for idx, url in enumerate(urls):
                if is_interrupted():
                    results.extend(page_error(u, "Interrupted") for u in urls[idx:])
                    break
                results.append(_fetch_and_extract_url(url))
            return results

        return run_extract("DuckDuckGo (ddgs)", logger, urls, _body)

    def get_setup_schema(self) -> dict[str, Any]:
        # post_setup triggers `_run_post_setup("ddgs")` so the package gets pip-installed on first pick.
        return setup_schema(
            "DuckDuckGo (ddgs)", "free · no key · local search + extract",
            "Search via ddgs and extract pages locally via httpx — no API key or cloud scraper", post_setup="ddgs",
        )
