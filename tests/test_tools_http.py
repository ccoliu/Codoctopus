# ---------------------------------------------------
# Codoctopus — HTTP tool tests
#
# The SSRF guard is tested directly against `_validate_public_url` (no
# network needed) plus one end-to-end request against a local httpx mock
# transport, so nothing here depends on real internet access.
# ---------------------------------------------------

from __future__ import annotations

import http.server
import socket
import threading

import httpcore
import httpx
import pytest

from codoctopus.tools.base import ToolContext
from codoctopus.tools.http import HttpRequestArgs, HttpRequestTool, _PublicOnlyBackend, _validate_public_url


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workspace=tmp_path)


# --- SSRF guard --------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://100.64.0.1/",  # shared/CGNAT range — not "private" to ipaddress, but not public
        "http://0.0.0.0/",  # reaches localhost on Linux
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
    ],
)
def test_validate_public_url_blocks_non_public_targets(url):
    with pytest.raises(ValueError, match="forbidden IP address"):
        _validate_public_url(url)


def test_validate_public_url_rejects_a_non_http_scheme():
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        _validate_public_url("file:///etc/passwd")


def test_validate_public_url_accepts_a_public_address():
    _validate_public_url("http://93.184.216.34/")  # a public IP literal - no DNS needed


# --- the tool itself, via a mock transport ------------------------------


def _mock_client_factory(handler):
    """
    Build a drop-in replacement for httpx.AsyncClient backed by a MockTransport.

    Captures the real AsyncClient class before it is patched — patching
    `codoctopus.tools.http.httpx.AsyncClient` replaces the attribute on the
    shared `httpx` module itself, so a lambda that calls `httpx.AsyncClient(...)`
    from inside would end up calling the very patch it's part of.
    """
    real_async_client = httpx.AsyncClient

    def factory(**kw):
        kw.pop("transport", None)  # replaced by the mock; the SSRF transport is tested separately below
        return real_async_client(transport=httpx.MockTransport(handler), **kw)

    return factory


async def test_http_request_returns_status_and_body(ctx, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-test"] == "1"
        return httpx.Response(200, text="ok")

    monkeypatch.setattr("codoctopus.tools.http.httpx.AsyncClient", _mock_client_factory(handler))
    # Bypass the SSRF check for this fake host, which does not resolve.
    monkeypatch.setattr("codoctopus.tools.http._validate_public_url", lambda url: None)

    tool = HttpRequestTool()
    result = await tool.run(HttpRequestArgs(url="http://example.test/", headers={"x-test": "1"}), ctx)

    assert "HTTP 200" in result
    assert "ok" in result


async def test_http_request_truncates_a_huge_response(ctx, monkeypatch):
    huge = "x" * 30_000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=huge)

    monkeypatch.setattr("codoctopus.tools.http.httpx.AsyncClient", _mock_client_factory(handler))
    monkeypatch.setattr("codoctopus.tools.http._validate_public_url", lambda url: None)

    tool = HttpRequestTool()
    result = await tool.run(HttpRequestArgs(url="http://example.test/"), ctx)

    assert "truncated" in result
    assert len(result) < len(huge)


async def test_http_request_refuses_a_private_target_before_any_network_call(ctx):
    tool = HttpRequestTool()
    with pytest.raises(ValueError, match="forbidden IP address"):
        await tool.run(HttpRequestArgs(url="http://127.0.0.1/admin"), ctx)


# --- DNS rebinding (issue #9) ------------------------------------------------


def _rebinding_getaddrinfo(monkeypatch, first: str, then: str) -> list[str]:
    """Answer the first lookup with `first` and every later one with `then`."""
    calls: list[str] = []

    def fake(host, port, *args, **kwargs):
        calls.append(host)
        ip = first if len(calls) == 1 else then
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    return calls


async def test_a_dns_rebind_between_check_and_connect_is_refused(ctx, monkeypatch):
    calls = _rebinding_getaddrinfo(monkeypatch, first="93.184.216.34", then="127.0.0.1")

    tool = HttpRequestTool()
    with pytest.raises(ValueError, match="forbidden IP address: 127.0.0.1"):
        await tool.run(HttpRequestArgs(url="http://rebind.test/"), ctx)

    # The up-front check passed; it was the connect-time lookup that caught it.
    assert len(calls) == 2


class _RecordingBackend(httpcore.AsyncNetworkBackend):
    def __init__(self) -> None:
        self.connected_to: list[str] = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connected_to.append(host)
        raise httpcore.ConnectError("stop here — only the target address matters")

    async def connect_unix_socket(self, *args, **kwargs):
        raise NotImplementedError

    async def sleep(self, seconds):
        pass


async def test_the_connection_goes_to_the_address_that_was_checked(monkeypatch):
    _rebinding_getaddrinfo(monkeypatch, first="93.184.216.34", then="93.184.216.34")
    inner = _RecordingBackend()

    with pytest.raises(httpcore.ConnectError):
        await _PublicOnlyBackend(inner).connect_tcp("example.test", 80)

    # The pinned IP, not the hostname — handing the hostname on would let the
    # inner backend resolve it a second time.
    assert inner.connected_to == ["93.184.216.34"]


@pytest.fixture
def local_server():
    """A real HTTP server on 127.0.0.1 — reachable if the guard ever stops working."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"internal secret")

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


async def test_the_real_transport_blocks_a_private_target_even_past_the_upfront_check(
    ctx, monkeypatch, local_server
):
    # Skip the up-front check entirely, as if it had been fooled; the only
    # thing left standing between the request and a live local server is the
    # connect-time guard on the real httpx transport.
    monkeypatch.setattr("codoctopus.tools.http._validate_public_url", lambda url: None)

    tool = HttpRequestTool()
    with pytest.raises(ValueError, match="forbidden IP address: 127.0.0.1"):
        await tool.run(HttpRequestArgs(url=f"http://127.0.0.1:{local_server}/"), ctx)
