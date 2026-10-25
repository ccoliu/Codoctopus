# ---------------------------------------------------
# Codoctopus — HTTP tool
#
# SSRF guard in two layers:
# - `_validate_public_url` checks a URL up front (clean early error).
# - `_PublicOnlyBackend` is what actually enforces it for http_request: it
#   resolves the hostname, checks every address, and connects to the address
#   it just checked. Checking first and letting httpx connect afterwards would
#   resolve the name twice, and a DNS-rebinding attacker can answer the second
#   lookup with 127.0.0.1 or 169.254.169.254.
# ---------------------------------------------------

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

import httpcore
import httpx
from pydantic import BaseModel, Field

from codoctopus.tools.base import Tool, ToolContext

#: Cap how much of a response body reaches the model's context.
_MAX_RESPONSE_CHARS = 20_000


def _forbidden(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # is_global rules out private, loopback, link-local, reserved, unspecified
    # (0.0.0.0) and shared/CGNAT (100.64.0.0/10) ranges in one go. An
    # IPv4-mapped IPv6 address (::ffff:127.0.0.1) is judged by its IPv4 form.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast


def _check_addresses(hostname: str, addresses: list[str]) -> None:
    if not addresses:
        raise ValueError(f"Failed to resolve hostname: {hostname}")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if _forbidden(ip):
            raise ValueError(f"Hostname resolved to a forbidden IP address: {ip}")


def _validate_public_url(url: str) -> None:
    """
    Refuse a URL whose host resolves to a non-public address — the standard
    SSRF guard against an agent being tricked into hitting internal services
    or a cloud metadata endpoint. On its own this is check-then-use (the
    caller resolves the name again), so http_request also pins the connection
    via _PublicOnlyBackend.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(f"Invalid URL: {url}")

    try:
        resolved = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Failed to resolve hostname: {hostname}") from exc

    _check_addresses(hostname, [info[4][0] for info in resolved])


class _ForbiddenTarget(httpcore.ConnectError):
    """Raised at connect time; kept distinct so run() can report it as an SSRF refusal."""


class _PublicOnlyBackend(httpcore.AsyncNetworkBackend):
    """
    Wraps httpcore's network backend so the TCP connection goes to an address
    that was checked a moment earlier, never to a fresh lookup. TLS still
    uses the original hostname for SNI and certificate verification — httpcore
    passes it to start_tls separately from the address it connected to.
    """

    def __init__(self, inner: httpcore.AsyncNetworkBackend) -> None:
        self._inner = inner

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise httpcore.ConnectError(f"Failed to resolve hostname: {host}") from exc

        addresses = [info[4][0] for info in infos]
        try:
            _check_addresses(host, addresses)
        except ValueError as exc:
            raise _ForbiddenTarget(str(exc)) from exc

        return await self._inner.connect_tcp(
            addresses[0], port, timeout=timeout, local_address=local_address, socket_options=socket_options
        )

    async def connect_unix_socket(self, *args: Any, **kwargs: Any) -> httpcore.AsyncNetworkStream:
        raise _ForbiddenTarget("Unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def _public_only_transport() -> httpx.AsyncHTTPTransport:
    transport = httpx.AsyncHTTPTransport()
    # httpx doesn't expose httpcore's network_backend option, so it's swapped
    # in on the pool directly. If an httpx upgrade renames these attributes,
    # test_tools_http's DNS-rebinding tests fail loudly rather than the guard
    # silently disappearing.
    pool = transport._pool
    pool._network_backend = _PublicOnlyBackend(pool._network_backend)
    return transport


class HttpRequestArgs(BaseModel):
    url: str = Field(description="URL to request. Must be a public http:// or https:// address.")
    method: str = Field(default="GET", description="HTTP method, e.g. GET, POST, PUT, DELETE")
    headers: dict[str, str] = Field(default_factory=dict, description="Request headers")
    body: str | None = Field(
        default=None,
        description="Raw request body. To send JSON, put the JSON text here and set "
        "a Content-Type: application/json header.",
    )
    timeout_seconds: float = Field(default=10.0, gt=0, le=30, description="Request timeout, capped at 30s")


class HttpRequestTool(Tool):
    """
    Make an outbound HTTP request.

    Redirects are never followed automatically — a redirect response is
    returned as-is so the caller can see where it points without this tool
    silently walking a chain of untrusted, unvalidated hops.
    """

    name = "http_request"
    description = (
        "Make an HTTP request to a public URL. Cannot reach private networks, "
        "localhost, or cloud metadata endpoints."
    )
    args_schema = HttpRequestArgs

    async def run(self, args: HttpRequestArgs, ctx: ToolContext) -> str:
        _validate_public_url(args.url)

        # trust_env=False: an HTTP(S)_PROXY from the environment would do its
        # own DNS lookup, out of reach of the connect-time check.
        async with httpx.AsyncClient(
            transport=_public_only_transport(),
            follow_redirects=False,
            timeout=args.timeout_seconds,
            trust_env=False,
        ) as client:
            try:
                response = await client.request(
                    args.method.upper(),
                    args.url,
                    headers=args.headers,
                    content=args.body,
                )
            except httpx.ConnectError as exc:
                if isinstance(exc.__cause__, _ForbiddenTarget):
                    raise ValueError(str(exc.__cause__)) from exc
                raise ValueError(f"Request failed: {exc}") from exc
            except httpx.HTTPError as exc:
                raise ValueError(f"Request failed: {exc}") from exc

        body = response.text
        truncated = len(body) > _MAX_RESPONSE_CHARS
        summary = f"HTTP {response.status_code}\n\n{body[:_MAX_RESPONSE_CHARS]}"
        if truncated:
            summary += f"\n\n[response truncated to {_MAX_RESPONSE_CHARS} characters]"
        return summary
