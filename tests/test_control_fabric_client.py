"""The outbound client: it reaches the hosts it was told to and nothing else, never an internal address, never more bytes than allowed, never a secret in a message."""
import http.server
import socket
import threading

import pytest

from substrate.control.fabric.client import FetchError, SafeHTTPClient, address_problem, redact_url, with_query

PUBLIC = "93.184.216.34"


class Handler(http.server.BaseHTTPRequestHandler):
    attempts = {"retry": 0}

    def log_message(self, *args):
        pass

    def reply(self, status=200, body=b"{}", content_type="application/json", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        port = self.server.server_address[1]
        if path == "/json":
            self.reply(body=b'{"ok": true}')
        elif path == "/csv":
            self.reply(body=b"a,b\n1,2\n", content_type="text/csv; charset=utf-8")
        elif path == "/html":
            self.reply(body=b"<html></html>", content_type="text/html")
        elif path == "/big":
            self.reply(body=b"x" * 5000, content_type="text/plain")
        elif path == "/nocontent-type":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
        elif path.startswith("/status/"):
            self.reply(status=int(path.rsplit("/", 1)[1]), body=b"{}")
        elif path == "/redirect-same":
            self.reply(status=302, headers={"Location": "/json"})
        elif path == "/redirect-absolute":
            self.reply(status=302, headers={"Location": f"http://127.0.0.1:{port}/json"})
        elif path == "/redirect-away":
            self.reply(status=302, headers={"Location": "http://evil.example/steal"})
        elif path == "/redirect-metadata":
            self.reply(status=302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})
        elif path == "/redirect-loop":
            self.reply(status=302, headers={"Location": "/redirect-loop"})
        elif path == "/redirect-nowhere":
            self.reply(status=302)
        elif path == "/retry":
            Handler.attempts["retry"] += 1
            if Handler.attempts["retry"] < 3:
                self.reply(status=429, headers={"Retry-After": "2"})
            else:
                self.reply(body=b'{"after": "retries"}')
        elif path == "/always-429":
            self.reply(status=429, headers={"Retry-After": "999"})
        elif path == "/echo-headers":
            import json
            self.reply(body=json.dumps({k.lower(): v for k, v in self.headers.items()}).encode())
        else:
            self.reply(status=404)


@pytest.fixture(scope="module")
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def client(server):
    Handler.attempts["retry"] = 0
    slept = []
    c = SafeHTTPClient({"127.0.0.1"}, allow_loopback=True, sleep=slept.append, max_bytes=1000)
    c.slept = slept
    return c


# -- fetching ----------------------------------------------------------------------------------------------------------------------------
def test_a_good_fetch_returns_the_bytes_the_type_and_the_url_without_its_query(client, server):
    response = client.fetch(f"{server}/json?api_key=SECRET&q=1")
    assert (response.status, response.content_type, response.body) == (200, "application/json", b'{"ok": true}')
    assert response.url == f"{server}/json" and "SECRET" not in response.url and response.retrieved_at > 1_700_000_000


def test_a_charset_on_the_content_type_is_fine(client, server):
    assert client.fetch(f"{server}/csv").content_type == "text/csv"


def test_the_request_identifies_itself_and_asks_for_no_compression_and_sends_no_cookies(client, server):
    import json
    seen = json.loads(client.fetch(f"{server}/echo-headers").body)
    assert seen["user-agent"].startswith("substrate/") and seen["accept-encoding"] == "identity" and "cookie" not in seen and "authorization" not in seen
    assert seen["connection"] == "close"


def test_an_adapter_can_add_a_header_but_never_one_with_a_line_break(client, server):
    import json
    assert json.loads(client.fetch(f"{server}/echo-headers", headers={"X-Api-Key": "k"}).body)["x-api-key"] == "k"
    for bad in ({"X-A": "v\r\nInjected: yes"}, {"X-A\nB": "v"}, {"": "v"}, {"X-A": "v\x00"}):
        with pytest.raises(FetchError, match="line break"):
            client.fetch(f"{server}/json", headers=bad)


# -- what is refused ---------------------------------------------------------------------------------------------------------------------
def test_loopback_and_plain_http_are_refused_unless_explicitly_enabled(server):
    strict = SafeHTTPClient({"127.0.0.1"})
    with pytest.raises(FetchError, match="plain http"):
        strict.fetch(f"{server}/json")
    with pytest.raises(FetchError, match="loopback"):
        strict.vet_address("127.0.0.1", 443)


@pytest.mark.parametrize("url, message", [
    ("https://other.example/x", "not on the allowlist"), ("https://data.example.org.evil.example/x", "not on the allowlist"),
    ("https://evil.example/https://data.example.org/", "not on the allowlist"),
    ("https://user:pw@data.example.org/x", "credentials"), ("https://data.example.org@evil.example/x", "credentials"),
    ("https://data.example.org/x#frag", "fragment"), ("ftp://data.example.org/x", "only https"), ("file:///etc/passwd", "only https"),
    ("javascript:alert(1)", "only https"), ("//data.example.org/x", "only https"), ("http://data.example.org/x", "plain http"),
    ("https://data.example.org:8443/x", "port 8443"), ("https://data.example.org:abc/x", "port is not a number"),
    ("https://data.example.org/a\\b", "characters that are not accepted"),
])
def test_urls_the_policy_does_not_allow_are_refused_before_anything_is_looked_up(url, message):
    calls = []
    client = SafeHTTPClient({"data.example.org"}, resolver=lambda *a, **k: calls.append(a) or [])
    with pytest.raises(FetchError, match=message):
        client.fetch(url)
    assert calls == []


def test_the_host_comparison_ignores_case_but_nothing_else():
    client = SafeHTTPClient({"Data.Example.ORG"})
    assert client.check_url("https://DATA.example.org/x?y=1") == ("https", "data.example.org", 443, "/x?y=1")


@pytest.mark.parametrize("address", ["10.0.0.5", "10.255.255.255", "172.16.0.1", "172.31.255.255", "192.168.1.1", "127.0.0.1", "127.1.2.3", "169.254.169.254", "169.254.0.1",
                                     "0.0.0.0", "224.0.0.1", "240.0.0.1", "100.64.0.1", "::1", "fe80::1", "fc00::1", "fd12:3456::1", "::", "ff02::1",
                                     "::ffff:10.0.0.1", "::ffff:127.0.0.1", "::ffff:169.254.169.254"])
def test_no_internal_address_is_ever_connected_to(address):
    assert address_problem(address) is not None
    client = SafeHTTPClient({"data.example.org"}, resolver=lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))])
    with pytest.raises(FetchError, match="not accepted"):
        client.vet_address("data.example.org", 443)


@pytest.mark.parametrize("address", [PUBLIC, "8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "172.32.0.1", "11.0.0.1", "100.128.0.1"])
def test_public_addresses_are_fine(address):
    assert address_problem(address) is None
    client = SafeHTTPClient({"data.example.org"}, resolver=lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))])
    assert client.vet_address("data.example.org", 443) == address


def test_one_internal_address_among_public_ones_is_enough_to_refuse():
    infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 443)) for a in (PUBLIC, "10.1.2.3")]
    client = SafeHTTPClient({"data.example.org"}, resolver=lambda *a, **k: infos)
    with pytest.raises(FetchError, match="private"):
        client.vet_address("data.example.org", 443)


def test_a_name_that_does_not_resolve_and_one_with_no_address_are_errors():
    def fail(*a, **k):
        raise socket.gaierror("no such host")

    with pytest.raises(FetchError, match="could not be resolved"):
        SafeHTTPClient({"data.example.org"}, resolver=fail).vet_address("data.example.org", 443)
    with pytest.raises(FetchError, match="no address"):
        SafeHTTPClient({"data.example.org"}, resolver=lambda *a, **k: []).vet_address("data.example.org", 443)


def test_the_connection_goes_to_the_address_that_was_checked_not_to_whatever_the_name_resolves_to_later(monkeypatch):
    """DNS rebinding: the name answers with a public address for the check and an internal one afterwards. The client resolves once and connects to the first."""
    answers = iter([PUBLIC, "169.254.169.254", "169.254.169.254"])
    resolutions, connected = [], []

    def resolver(host, port, **kw):
        resolutions.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), port))]

    def create_connection(address, timeout=None, *a, **k):
        connected.append(address)
        raise OSError("stop here: no network in tests")

    monkeypatch.setattr(socket, "create_connection", create_connection)
    client = SafeHTTPClient({"data.example.org"}, resolver=resolver, retries=0)
    with pytest.raises(FetchError, match="could not fetch"):
        client.fetch("https://data.example.org/x")
    assert resolutions == ["data.example.org"] and connected == [(PUBLIC, 443)]


# -- redirects -----------------------------------------------------------------------------------------------------------------------------
def test_a_redirect_within_the_allowlist_is_followed(client, server):
    assert client.fetch(f"{server}/redirect-same").body == b'{"ok": true}'
    assert client.fetch(f"{server}/redirect-absolute").body == b'{"ok": true}'


@pytest.mark.parametrize("path, message", [("/redirect-away", "not on the allowlist"), ("/redirect-metadata", "not on the allowlist"),
                                           ("/redirect-loop", "too many redirects"), ("/redirect-nowhere", "no location")])
def test_a_redirect_is_checked_like_any_other_request_and_loops_end(client, server, path, message):
    with pytest.raises(FetchError, match=message):
        client.fetch(server + path)


# -- bounds ---------------------------------------------------------------------------------------------------------------------------------
def test_a_response_larger_than_the_limit_is_refused_by_its_declared_length_and_by_its_actual_length(client, server):
    with pytest.raises(FetchError, match="larger than 1,000 bytes"):
        client.fetch(f"{server}/big")                                                      # Content-Length says 5000
    # a server that lies about the length: read stops at the limit
    liar = SafeHTTPClient({"127.0.0.1"}, allow_loopback=True, max_bytes=100)
    with pytest.raises(FetchError, match="larger than 100 bytes"):
        liar.fetch(f"{server}/big")


@pytest.mark.parametrize("path, message", [("/html", "content type"), ("/nocontent-type", "content type"), ("/status/404", "answered 404"), ("/status/500", "answered 500"),
                                           ("/status/204", "answered 204"), ("/missing", "answered 404")])
def test_only_status_200_with_a_listed_content_type_is_accepted(client, server, path, message):
    with pytest.raises(FetchError, match=message):
        client.fetch(server + path)


def test_busy_servers_are_retried_politely_and_given_up_on(client, server):
    assert client.fetch(f"{server}/retry").body == b'{"after": "retries"}'
    assert client.slept == [2.0, 2.0]                                                      # the Retry-After it asked for
    client.slept.clear()
    with pytest.raises(FetchError, match="answered 429"):
        client.fetch(f"{server}/always-429")
    assert client.slept == [10.0, 10.0]                                                    # capped, so a hostile server cannot make a worker sleep for hours


def test_requests_to_one_host_are_spaced_by_the_minimum_interval(server):
    now, slept = [1000.0], []
    client = SafeHTTPClient({"127.0.0.1"}, allow_loopback=True, min_interval_s=1.0, sleep=slept.append, clock=lambda: now[0])
    client.fetch(f"{server}/json")
    now[0] += 0.25
    client.fetch(f"{server}/json")
    now[0] += 5
    client.fetch(f"{server}/json")
    assert slept == [0.75]


def test_no_error_message_ever_contains_a_query_string(client, server):
    secret = "api_key=SECRET-VALUE-123"
    for path in ("/status/500", "/html", "/redirect-away", "/nowhere"):
        with pytest.raises(FetchError) as caught:
            client.fetch(f"{server}{path}?{secret}")
        assert "SECRET-VALUE" not in str(caught.value) and secret not in str(caught.value)
    with pytest.raises(FetchError) as caught:
        SafeHTTPClient({"data.example.org"}, retries=0, resolver=lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))]).fetch(
            f"https://data.example.org/p?{secret}")
    assert "SECRET-VALUE" not in str(caught.value)


def test_redaction_and_query_building():
    assert redact_url("https://user:pw@host.example:8443/a/b?k=v#f") == "https://host.example:8443/a/b"
    assert redact_url("not a url") == "<invalid url>" and redact_url("secret=abc") == "<invalid url>"
    assert with_query("https://host.example/p?a=1", key="x y", n=3) == "https://host.example/p?a=1&key=x+y&n=3"
