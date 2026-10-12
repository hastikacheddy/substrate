"""The only code in the package that makes outbound connections: an HTTP client that can reach a short list of named hosts and nothing else.

A data adapter is given a URL to fetch that is partly built from a user's query, and the response is untrusted data. So the client:

* connects only to **hosts on an allowlist**, over **HTTPS** (plain HTTP is accepted for loopback only, and only when asked for, for tests and local mirrors);
* refuses a URL with **credentials in it**, a **non-standard port**, or a **fragment**;
* resolves the host **once**, refuses any address that is **private, loopback, link-local (this includes the 169.254.169.254 cloud-metadata endpoint), multicast, reserved
  or unspecified**, and then **connects to that vetted address**, using the host name only for the TLS handshake, so a DNS answer that changes between the check and the
  connection (DNS rebinding) cannot redirect the request;
* follows **redirects only within the allowlist**, re-checking every hop, and at most three;
* accepts only **status 200** and a **listed content type**, and reads **at most a set number of bytes**;
* never puts a secret in an error message: query strings are redacted wherever a URL is reported.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ... import __version__
from ...errors import SubstrateError

ALLOWED_TYPES = ("application/json", "text/csv", "text/plain", "text/tab-separated-values")
MAX_REDIRECTS = 3


class FetchError(SubstrateError):
    """A fetch was refused by policy or did not succeed. The message is safe to show: it never contains a query string."""


@dataclass(frozen=True)
class Response:
    url: str                    # the URL fetched, with its query string removed
    status: int
    content_type: str
    body: bytes
    retrieved_at: float


def redact_url(url: str) -> str:
    """The URL without its query string, fragment or credentials: what may appear in a log or an error."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if not parts.scheme or not host:
        return "<invalid url>"                                                     # nothing of a malformed string is repeated: it might be a secret
    return urlunsplit((parts.scheme, host + (f":{parts.port}" if parts.port else ""), parts.path, "", ""))


def address_problem(address: str) -> str | None:
    """Why an address may not be connected to, or None if it may."""
    try:
        ip = ipaddress.ip_address(address.split("%")[0])
    except ValueError:
        return "not an IP address"
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    for name in ("is_loopback", "is_private", "is_link_local", "is_multicast", "is_reserved", "is_unspecified"):
        if getattr(ip, name):
            return name[3:].replace("_", " ")
    if ip.version == 4 and ip in ipaddress.ip_network("100.64.0.0/10"):           # carrier-grade NAT: not public either
        return "shared address space"
    return None


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connects to a given, already vetted IP address while verifying the certificate for the host name."""

    def __init__(self, host: str, address: str, port: int, timeout: float, context: ssl.SSLContext):
        super().__init__(host, port, timeout=timeout, context=context)
        self._address = address

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, address: str, port: int, timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class SafeHTTPClient:
    def __init__(self, allowed_hosts, *, timeout: float = 15.0, max_bytes: int = 5_000_000, allow_loopback: bool = False, min_interval_s: float = 0.0,
                 retries: int = 2, resolver=socket.getaddrinfo, sleep=time.sleep, clock=time.time, user_agent: str | None = None):
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)
        self.timeout, self.max_bytes, self.allow_loopback = timeout, max_bytes, allow_loopback
        self.min_interval_s, self.retries = min_interval_s, retries
        self.resolver, self.sleep, self.clock = resolver, sleep, clock
        self.user_agent = user_agent or f"substrate/{__version__} (+https://github.com/hastikacheddy/substrate)"
        self._last: dict[str, float] = {}
        self._context = ssl.create_default_context()                              # verifies certificates and host names

    # -- policy -----------------------------------------------------------------------------------------------------------------------
    def check_url(self, url: str) -> tuple[str, str, int, str]:
        """(scheme, host, port, path-and-query) for a URL the policy allows; raises FetchError for one it does not."""
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if parts.username or parts.password or "@" in parts.netloc:
            raise FetchError("a URL with credentials in it is refused")
        if parts.fragment:
            raise FetchError("a URL with a fragment is refused")
        if parts.scheme not in ("https", "http"):
            raise FetchError(f"only https is accepted (got '{parts.scheme}')")
        if host not in self.allowed_hosts:
            raise FetchError(f"host '{host}' is not on the allowlist")
        loopback_ok = self.allow_loopback and host in ("127.0.0.1", "localhost", "::1")
        if parts.scheme == "http" and not loopback_ok:
            raise FetchError("plain http is accepted only for loopback, and only when enabled")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            raise FetchError("the URL's port is not a number") from None
        if port != (443 if parts.scheme == "https" else 80) and not loopback_ok:
            raise FetchError(f"port {port} is not accepted")
        path = parts.path or "/"
        if any(c in path for c in ("\r", "\n", "\x00")) or "\\" in path:
            raise FetchError("the URL's path contains characters that are not accepted")
        return parts.scheme, host, port, path + (f"?{parts.query}" if parts.query else "")

    def vet_address(self, host: str, port: int) -> str:
        """Resolve `host` once and return an address that may be connected to."""
        try:
            infos = self.resolver(host, port, type=socket.SOCK_STREAM)
        except OSError:
            raise FetchError(f"host '{host}' could not be resolved") from None
        addresses = []
        for info in infos:
            address = info[4][0]
            problem = address_problem(address)
            if problem == "loopback" and self.allow_loopback and host in ("127.0.0.1", "localhost", "::1"):
                problem = None
            if problem:
                raise FetchError(f"host '{host}' resolves to an address that is not accepted ({problem})")
            addresses.append(address)
        if not addresses:
            raise FetchError(f"host '{host}' has no address")
        return addresses[0]

    # -- fetching ---------------------------------------------------------------------------------------------------------------------
    def fetch(self, url: str, *, headers: dict[str, str] | None = None) -> Response:
        current = url
        for hop in range(MAX_REDIRECTS + 1):
            for attempt in range(self.retries + 1):
                status, response_headers, body = self._once(current, headers or {})
                if status in (429, 503) and attempt < self.retries:
                    try:
                        wait = min(float(response_headers.get("retry-after", "1")), 10.0)
                    except ValueError:
                        wait = 1.0
                    self.sleep(wait)
                    continue
                break
            if status in (301, 302, 303, 307, 308):
                location = response_headers.get("location")
                if not location or hop == MAX_REDIRECTS:
                    raise FetchError(f"too many redirects fetching {redact_url(url)}" if location else f"a redirect from {redact_url(current)} has no location")
                current = urlsplit(current)._replace(path="", query="", fragment="").geturl() + location if location.startswith("/") else location
                continue
            if status != 200:
                raise FetchError(f"{redact_url(current)} answered {status}")
            content_type = response_headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in ALLOWED_TYPES:
                raise FetchError(f"{redact_url(current)} answered with a content type that is not accepted ('{content_type}')")
            return Response(redact_url(current), status, content_type, body, self.clock())
        raise FetchError(f"too many redirects fetching {redact_url(url)}")

    def _once(self, url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        for name, value in headers.items():
            if any(c in f"{name}{value}" for c in ("\r", "\n", "\x00")) or not name:
                raise FetchError("a request header with a line break in it is refused")
        scheme, host, port, target = self.check_url(url)
        address = self.vet_address(host, port)
        if self.min_interval_s:
            wait = self._last.get(host, 0.0) + self.min_interval_s - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last[host] = self.clock()
        connection = (_PinnedHTTPSConnection(host, address, port, self.timeout, self._context) if scheme == "https"
                      else _PinnedHTTPConnection(host, address, port, self.timeout))
        try:
            connection.request("GET", target, headers={"Host": host if port in (80, 443) else f"{host}:{port}", "User-Agent": self.user_agent,
                                                       "Accept": ", ".join(ALLOWED_TYPES), "Accept-Encoding": "identity", "Connection": "close", **headers})
            response = connection.getresponse()
            declared = response.getheader("content-length")
            if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
                raise FetchError(f"{redact_url(url)} is larger than {self.max_bytes:,} bytes")
            body = response.read(self.max_bytes + 1)
            if len(body) > self.max_bytes:
                raise FetchError(f"{redact_url(url)} is larger than {self.max_bytes:,} bytes")
            return response.status, {k.lower(): v for k, v in response.getheaders()}, body
        except FetchError:
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError) as error:
            raise FetchError(f"could not fetch {redact_url(url)} ({type(error).__name__})") from None
        finally:
            connection.close()


def with_query(url: str, **params) -> str:
    """`url` with query parameters added (and encoded), for adapters that need a key in the query string."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query)) | {k: str(v) for k, v in params.items()}
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))
