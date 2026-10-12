"""A wake-up signal for workers, so that an idle worker does not have to ask the database for work twice a second.

The task queue is the database. Nothing here decides what runs, for whom or in what order, and a notification carries no data: it says "look at the queue". So a lost,
duplicated, delayed or forged notification costs at most one poll interval (or one needless query), and when Redis is down the workers are plain pollers again. That is also
why the connection may be as plain as it is: whoever can publish on the channel can make a worker query the database sooner, and can do nothing else.

Redis must never slow down the things that matter, so every call to it is made on a background thread: `publish()` only sets a flag (the API's request and a worker's loop go
on at once, even when Redis is down and its name no longer resolves, which can take seconds that no socket timeout covers), and `wait()` only waits on a local event that
the listening thread sets. After a failure the threads stop trying for a few seconds.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from urllib.parse import urlsplit

log = logging.getLogger("substrate.control.notify")

CHANNEL = "substrate:wake"
SLICE_S = 0.5                       # how long one blocking read lasts, so that a stop request is noticed within half a second


class Notifier:
    """No notifier: workers poll. The base class is the one used when no Redis is configured."""

    @property
    def active(self) -> bool:
        return False

    def start(self) -> None:
        """Begin listening (a worker calls this once, before its first idle moment)."""

    def publish(self) -> None:
        """Tell the idle workers that there may be something to claim. Returns at once."""

    def wait(self, stop: threading.Event, poll_s: float, idle_s: float) -> None:
        """Return when there may be work: after `poll_s` with no notifier, after a notification or `idle_s` with one."""
        stop.wait(poll_s)

    def close(self) -> None: ...


class RedisNotifier(Notifier):
    """Publish and subscribe on one Redis channel. `client` is a redis-py client (or anything with `publish` and `pubsub`)."""

    def __init__(self, client, *, channel: str = CHANNEL, retry_s: float = 5.0, clock=time.monotonic):
        self.client, self.channel, self.retry_s, self.clock = client, channel, retry_s, clock
        self._closing = threading.Event()
        self._lock = threading.Lock()
        self._listener = self._sender = None
        self._connected = False             # subscribed right now
        self._woken = threading.Event()     # a notification arrived since the last wait
        self._to_send = threading.Event()   # a publish was asked for and not yet made
        self._sending = False
        self._publish_failed = False
        self._publish_after = 0.0           # no publishing before this time (after a failure)

    @property
    def active(self) -> bool:
        return self._connected

    # -- publishing (API and workers) -------------------------------------------------------------------------------------------------
    def publish(self) -> None:
        self._start_thread("_sender", self._send_loop, "substrate-wake-sender")
        self._to_send.set()                                           # asking twice before the sender looks is one publish: a wake-up has no content

    def _send_loop(self) -> None:
        while not self._closing.is_set():
            self._to_send.wait()
            if self._closing.is_set():
                return
            self._sending = True
            self._to_send.clear()
            try:
                if self.clock() < self._publish_after:
                    continue
                try:
                    self.client.publish(self.channel, b"1")
                except Exception as error:                            # noqa: BLE001 (redis-py raises its own hierarchy)
                    self._publish_after = self.clock() + self.retry_s
                    if not self._publish_failed:                      # once per outage, and never the URL: it holds the password
                        log.warning("cannot publish the wake-up on Redis (%s); workers will find the work when they poll", type(error).__name__)
                    self._publish_failed = True
                else:
                    if self._publish_failed:
                        log.info("publishing on Redis works again")
                    self._publish_failed = False
            finally:
                self._sending = False

    def drain(self, timeout: float = 5.0) -> bool:
        """Wait until every publish asked for so far has been tried. For tests and for a clean exit."""
        deadline = time.monotonic() + timeout
        while (self._to_send.is_set() or self._sending) and time.monotonic() < deadline:
            time.sleep(0.005)
        return not (self._to_send.is_set() or self._sending)

    # -- listening (workers) ----------------------------------------------------------------------------------------------------------
    def start(self) -> None:
        self._start_thread("_listener", self._listen_loop, "substrate-wake-listener")

    def _listen_loop(self) -> None:
        outage_logged = False
        while not self._closing.is_set():
            pubsub = None
            try:
                pubsub = self.client.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(self.channel)
                self._connected, outage_logged = True, False
                while not self._closing.is_set():
                    if pubsub.get_message(timeout=SLICE_S) is not None:
                        while pubsub.get_message(timeout=0) is not None:          # a burst of notifications is one wake-up
                            pass
                        self._woken.set()
            except Exception as error:                                # noqa: BLE001
                if not outage_logged:
                    log.warning("no subscription to the wake-up channel on Redis (%s); polling until it comes back", type(error).__name__)
                    outage_logged = True
            finally:
                self._connected = False
                try:
                    if pubsub is not None:
                        pubsub.close()
                except Exception:                                     # noqa: BLE001
                    pass
            self._closing.wait(self.retry_s)

    def wait(self, stop: threading.Event, poll_s: float, idle_s: float) -> None:
        """Return on a notification; failing that, after `idle_s` while subscribed and after `poll_s` while not (the limit is looked at again every slice, so a
        subscription that comes up or is lost during the wait changes it, and a notification is heard whichever state the connection is in)."""
        self.start()
        started = time.monotonic()
        while not stop.is_set():
            remaining = started + (idle_s if self._connected else poll_s) - time.monotonic()
            if remaining <= 0:
                return
            if self._woken.wait(min(remaining, SLICE_S)):
                self._woken.clear()
                return

    # -- housekeeping -------------------------------------------------------------------------------------------------------------------
    def _start_thread(self, attribute: str, target, name: str) -> None:
        if getattr(self, attribute) is None and not self._closing.is_set():
            with self._lock:
                if getattr(self, attribute) is None:
                    thread = threading.Thread(target=target, name=name, daemon=True)
                    setattr(self, attribute, thread)
                    thread.start()

    def close(self) -> None:
        self._closing.set()
        self._to_send.set()
        for thread in (self._listener, self._sender):
            if thread is not None:
                thread.join(timeout=2.0)                              # a thread stuck in a DNS lookup is a daemon: it does not hold the process up
        try:
            closer = getattr(self.client, "close", None)
            if closer is not None:
                closer()
        except Exception:                                             # noqa: BLE001
            pass


def redis_url(environ=None) -> str:
    """The Redis URL: $SUBSTRATE_REDIS_URL, else the contents of the file named by $SUBSTRATE_REDIS_URL_FILE (how a container receives it from a secret, since the URL
    holds the password), else empty. Raises ValueError, never with the URL in the message."""
    environ = os.environ if environ is None else environ
    url, path = (environ.get("SUBSTRATE_REDIS_URL") or "").strip(), (environ.get("SUBSTRATE_REDIS_URL_FILE") or "").strip()
    if url and path:
        raise ValueError("set SUBSTRATE_REDIS_URL or SUBSTRATE_REDIS_URL_FILE, not both")
    if path:
        try:
            with open(path, encoding="utf-8") as handle:
                url = handle.read().strip()
        except OSError as error:
            raise ValueError(f"cannot read the Redis URL from {path} ({type(error).__name__})") from None
        if not url:
            raise ValueError(f"{path} is empty: it should hold the Redis URL")
    return url


def notifier_from_environment(environ=None, client=None) -> Notifier:
    """A RedisNotifier when a Redis URL is configured, else the polling Notifier. `client` is for tests."""
    url = redis_url(environ)
    if not url:
        return Notifier()
    try:
        parts = urlsplit(url)
        hostname, _ = parts.hostname, parts.port                  # the port is checked when it is read
    except ValueError:
        raise ValueError("the Redis URL is not a valid URL") from None
    if parts.scheme not in ("redis", "rediss") or not hostname or parts.fragment:
        raise ValueError("the Redis URL must look like redis://[:password@]host[:port][/db] (or rediss:// for TLS)")
    if client is None:
        try:
            import redis
        except ImportError:
            raise ImportError("a Redis URL is configured but the redis package is not installed (pip install 'substrate[redis]')") from None
        client = redis.Redis.from_url(url, socket_connect_timeout=1.0, socket_timeout=2.0, health_check_interval=30)
    return RedisNotifier(client)
