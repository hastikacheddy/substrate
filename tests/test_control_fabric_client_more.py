"""Cases of the outbound client that the first tests did not pin, found by mutation testing: redirect and retry counts, and the corners of the URL and address policy."""
import http.server
import socket
import threading

import pytest

from substrate.control.fabric.client import FetchError, SafeHTTPClient, address_problem


class Handler(http.server.BaseHTTPRequestHandler):
    hits: list = []
    busy = {"served": 0}

    def log_message(self, *args):
        pass

    def reply(self, status=200, body=b'{"ok": true}', headers=None):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        Handler.hits.append(path)
        if path.startswith("/hop/"):                                    # /hop/N redirects to /hop/N-1; /hop/0 answers
            n = int(path.rsplit("/", 1)[1])
            self.reply() if n == 0 else self.reply(status=302, body=b"", headers={"Location": f"/hop/{n - 1}"})
        elif path.startswith("/via/"):                                  # /via/CODE redirects with that status to /ok
            self.reply(status=int(path.rsplit("/", 1)[1]), body=b"", headers={"Location": "/ok"})
        elif path == "/ok":
            self.reply()
        elif path == "/busy":
            Handler.busy["served"] += 1
            self.reply(status=503, body=b"", headers={"Retry-After": "1"}) if Handler.busy["served"] == 1 else self.reply()
        elif path.startswith("/stream-"):                               # no Content-Length: the body ends when the connection closes, so only counting bytes can bound it
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"x" * int(path.rsplit("-", 1)[1]))
        elif path == "/declares-a-gigabyte":                            # the headers promise far more than the limit and then nothing follows
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000000000")
            self.end_headers()
        else:
            self.reply(status=404, body=b"")


@pytest.fixture(scope="module")
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture
def client(server):
    Handler.hits.clear()
    Handler.busy["served"] = 0
    slept = []
    c = SafeHTTPClient({"127.0.0.1"}, allow_loopback=True, sleep=slept.append, max_bytes=1000)
    c.slept = slept
    return c


# -- redirects and retries, counted ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("hops", [1, 2, 3])
def test_up_to_three_redirects_are_followed(client, server, hops):
    assert client.fetch(f"{server}/hop/{hops}").body == b'{"ok": true}'
    assert Handler.hits == [f"/hop/{n}" for n in range(hops, -1, -1)]


def test_a_fourth_redirect_is_not_followed(client, server):
    with pytest.raises(FetchError, match="too many redirects"):
        client.fetch(f"{server}/hop/4")
    assert Handler.hits == ["/hop/4", "/hop/3", "/hop/2", "/hop/1"]                      # the request the fourth redirect pointed at is never made


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_every_ordinary_redirect_status_is_followed(client, server, status):
    assert client.fetch(f"{server}/via/{status}").body == b'{"ok": true}'


def test_a_503_is_retried_after_the_wait_the_server_asked_for(client, server):
    assert client.fetch(f"{server}/busy").body == b'{"ok": true}'
    assert client.slept == [1.0] and Handler.hits == ["/busy", "/busy"]


def test_a_length_declared_over_the_limit_is_refused_without_waiting_for_the_body(client, server):
    with pytest.raises(FetchError, match="larger than 1,000 bytes"):
        client.fetch(f"{server}/declares-a-gigabyte")


def test_a_body_of_unknown_length_is_cut_off_at_one_byte_over_the_limit_and_refused(client, server):
    assert len(client.fetch(f"{server}/stream-1000").body) == 1000                      # exactly the limit is fine
    with pytest.raises(FetchError, match="larger than 1,000 bytes"):
        client.fetch(f"{server}/stream-1001")                                           # one byte more is refused, not quietly truncated
    with pytest.raises(FetchError, match="larger than 1,000 bytes"):
        client.fetch(f"{server}/stream-50000")


# -- the corners of the URL policy -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("url, message", [("https://@data.example.org/x", "credentials"), ("https://:@data.example.org/x", "credentials"),
                                          ("https://data.example.org/a\x00b", "characters that are not accepted")])
def test_urls_with_an_empty_user_or_a_nul_byte_are_refused(url, message):
    with pytest.raises(FetchError, match=message):
        SafeHTTPClient({"data.example.org"}).check_url(url)


def test_enabling_loopback_does_not_relax_the_rules_for_any_other_host():
    client = SafeHTTPClient({"data.example.org", "127.0.0.1"}, allow_loopback=True)
    with pytest.raises(FetchError, match="plain http"):
        client.check_url("http://data.example.org/x")
    with pytest.raises(FetchError, match="port 8443"):
        client.check_url("https://data.example.org:8443/x")
    assert client.check_url("http://127.0.0.1:8080/x") == ("http", "127.0.0.1", 8080, "/x")             # loopback itself may use plain http on any port


def test_a_named_host_that_resolves_to_loopback_is_refused_even_when_loopback_is_enabled():
    def resolver(host, port, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    client = SafeHTTPClient({"data.example.org", "127.0.0.1"}, allow_loopback=True, resolver=resolver)
    with pytest.raises(FetchError, match="loopback"):
        client.vet_address("data.example.org", 443)                                          # a name on the allowlist that points inside is the attack
    assert client.vet_address("127.0.0.1", 443) == "127.0.0.1"                              # the literal loopback address the operator enabled is fine


def test_an_ipv4_address_written_as_ipv6_is_judged_as_the_ipv4_address():
    assert address_problem("::ffff:8.8.8.8") is None
    assert address_problem("::ffff:10.0.0.1") is not None and address_problem("::ffff:169.254.169.254") is not None


def test_the_certificate_is_verified_for_the_host_name_while_the_connection_goes_to_the_vetted_address(monkeypatch):
    import ssl
    connected, wrapped = [], []

    def wrap_socket(sock, server_hostname=None):
        wrapped.append(server_hostname)
        raise ssl.SSLError("stop here: no TLS in tests")

    monkeypatch.setattr(socket, "create_connection", lambda address, timeout=None, *a, **k: connected.append(address) or object())
    public = "93.184.216.34"
    client = SafeHTTPClient({"data.example.org"}, resolver=lambda host, port, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (public, port))], retries=0)
    client._context.wrap_socket = wrap_socket                                            # the real context, with the handshake replaced
    with pytest.raises(FetchError, match="could not fetch"):
        client.fetch("https://data.example.org/x")
    assert connected == [(public, 443)] and wrapped == ["data.example.org"]


def test_the_default_tls_context_verifies_certificates_and_host_names():
    import ssl
    context = SafeHTTPClient({"data.example.org"})._context
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True
