"""The wake-up signal: what it does when Redis is there, and (more to the point) that nothing breaks or slows down when it is not.

The fake client below stands in for redis-py in the unit tests. The tests marked "real Redis" run against a server named by SUBSTRATE_TEST_REDIS_URL and skip without one.
"""
import logging
import os
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from substrate.control import notify  # noqa: E402
from substrate.control.artifacts import Artifacts, LocalArtifactStore  # noqa: E402
from substrate.control.notify import Notifier, RedisNotifier, notifier_from_environment, redis_url  # noqa: E402
from substrate.control.orchestrator import Orchestrator  # noqa: E402
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.security import Role  # noqa: E402
from substrate.control.worker import Worker  # noqa: E402
from substrate.experiment import read_spec  # noqa: E402

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
PASSWORD = "s3cr3t-pa55word-xyz"


class FakePubSub:
    def __init__(self, hub, fail_on_read=False):
        self.hub, self.messages, self.fail_on_read, self.closed = hub, [], fail_on_read, False
        self.cv = threading.Condition()

    def subscribe(self, channel):
        if self.hub.subscribe_delay:
            time.sleep(self.hub.subscribe_delay)
        if self.hub.fail_subscribe:
            raise ConnectionError("refused")
        self.hub.subscribers.setdefault(channel, []).append(self)

    def deliver(self, data):
        with self.cv:
            self.messages.append(data)
            self.cv.notify_all()

    def get_message(self, timeout=0.0):
        if self.fail_on_read:
            raise ConnectionError("reset by peer")
        with self.cv:
            if not self.messages and timeout:
                self.cv.wait(timeout)
            return {"type": "message", "data": self.messages.pop(0)} if self.messages else None

    def close(self):
        self.closed = True


class FakeRedis:
    def __init__(self):
        self.subscribers, self.published = {}, 0
        self.fail_publish = self.fail_subscribe = self.next_pubsub_fails = False
        self.publish_delay = self.subscribe_delay = 0.0

    def publish(self, channel, data):
        if self.publish_delay:
            time.sleep(self.publish_delay)
        if self.fail_publish:
            raise ConnectionError("refused")
        self.published += 1
        for pubsub in self.subscribers.get(channel, []):
            pubsub.deliver(data)
        return len(self.subscribers.get(channel, []))

    def pubsub(self, **options):
        assert options.get("ignore_subscribe_messages") is True
        fails, self.next_pubsub_fails = self.next_pubsub_fails, False
        return FakePubSub(self, fail_on_read=fails)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def timed(call, *args):
    start = time.perf_counter()
    call(*args)
    return time.perf_counter() - start


def eventually(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def listening(redis, **options):
    notifier = RedisNotifier(redis, **options)
    notifier.start()
    assert eventually(lambda: notifier.active), "the listener did not subscribe"
    return notifier


@pytest.fixture
def notifiers():
    made = []
    yield made
    for notifier in made:
        notifier.close()


# -- waiting and waking -------------------------------------------------------------------------------------------------------------------
def test_a_notification_wakes_a_waiting_worker_long_before_its_idle_interval_ends(notifiers):
    redis = FakeRedis()
    waiter, publisher = listening(redis), RedisNotifier(redis)
    notifiers += [waiter, publisher]
    threading.Timer(0.2, publisher.publish).start()
    assert timed(waiter.wait, threading.Event(), 30.0, 30.0) < 3.0


def test_with_no_notification_the_wait_lasts_the_idle_interval_and_no_longer(notifiers):
    waiter = listening(FakeRedis())
    notifiers.append(waiter)
    elapsed = timed(waiter.wait, threading.Event(), 30.0, 0.6)
    assert 0.5 <= elapsed < 3.0


def test_a_notification_that_arrived_while_the_worker_was_busy_is_still_there_when_it_waits(notifiers):
    redis = FakeRedis()
    waiter, publisher = listening(redis), RedisNotifier(redis)
    notifiers += [waiter, publisher]
    publisher.publish()                                               # while the "worker" is busy elsewhere
    assert publisher.drain() and eventually(lambda: waiter._woken.is_set())
    assert timed(waiter.wait, threading.Event(), 30.0, 30.0) < 1.0


def test_a_burst_of_notifications_is_one_wake_up(notifiers):
    redis = FakeRedis()
    waiter, publisher = listening(redis), RedisNotifier(redis)
    notifiers += [waiter, publisher]
    for _ in range(7):
        publisher.publish()
        publisher.drain()
    time.sleep(0.3)                                                   # all of them have been delivered and read
    assert timed(waiter.wait, threading.Event(), 30.0, 30.0) < 1.0
    assert timed(waiter.wait, threading.Event(), 30.0, 0.4) >= 0.3    # and one wake-up used them all up


def test_a_stop_request_ends_the_wait_within_a_slice(notifiers):
    waiter = listening(FakeRedis())
    notifiers.append(waiter)
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    assert timed(waiter.wait, stop, 30.0, 30.0) < notify.SLICE_S + 1.5


def test_the_base_notifier_is_a_plain_poll():
    stop = threading.Event()
    assert Notifier().active is False
    assert 0.15 <= timed(Notifier().wait, stop, 0.2, 30.0) < 2.0
    Notifier().start()
    Notifier().publish()                                              # nothing to do, nothing raised
    Notifier().close()


# -- Redis must not slow anything down -------------------------------------------------------------------------------------------------
def test_publishing_returns_at_once_even_when_the_redis_call_takes_seconds(notifiers):
    """Measured on a stack with Redis stopped: its name stopped resolving, and every API submit blocked for 4-8 s in the lookup, which no socket timeout covers."""
    redis = FakeRedis()
    redis.publish_delay = 2.0
    notifier = RedisNotifier(redis)
    notifiers.append(notifier)
    assert timed(notifier.publish) < 0.3
    assert timed(notifier.publish) < 0.3                              # nor does the second, while the first is still in flight
    assert notifier.drain(timeout=10) and redis.published >= 1        # and it was made, in the background


def test_waiting_while_a_subscription_attempt_hangs_still_polls_at_the_poll_interval(notifiers):
    redis = FakeRedis()
    redis.subscribe_delay = 3.0
    notifier = RedisNotifier(redis)
    notifiers.append(notifier)
    for _ in range(3):
        assert 0.15 <= timed(notifier.wait, threading.Event(), 0.2, 30.0) < 1.0
    assert notifier.active is False


# -- when Redis is not there -----------------------------------------------------------------------------------------------------------------
def test_publishing_never_raises_logs_once_per_outage_and_stops_trying_for_a_while(caplog, notifiers):
    clock, redis = Clock(), FakeRedis()
    notifier = RedisNotifier(redis, retry_s=5.0, clock=clock)
    notifiers.append(notifier)
    redis.fail_publish = True
    with caplog.at_level(logging.INFO, logger="substrate.control.notify"):
        notifier.publish()
        assert notifier.drain()
        redis.fail_publish = False
        notifier.publish()                                            # inside the back-off: not even attempted
        assert notifier.drain() and redis.published == 0
        clock.t += 5.1
        notifier.publish()                                            # tried again, worked
        assert notifier.drain() and redis.published == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "ConnectionError" in warnings[0].getMessage()
    assert any("works again" in r.getMessage() for r in caplog.records)


def test_a_failed_subscription_falls_back_to_polling_and_is_retried_later(caplog, notifiers):
    redis = FakeRedis()
    redis.fail_subscribe = True
    notifier = RedisNotifier(redis, retry_s=0.2)
    notifiers.append(notifier)
    with caplog.at_level(logging.WARNING, logger="substrate.control.notify"):
        assert 0.15 <= timed(notifier.wait, threading.Event(), 0.2, 30.0) < 2.0          # the poll interval, not the idle one
        time.sleep(0.7)                                               # several retries
    assert notifier.active is False
    assert len([r for r in caplog.records if "no subscription" in r.getMessage()]) == 1  # logged once for the outage, not once per retry
    redis.fail_subscribe = False
    assert eventually(lambda: notifier.active)
    assert timed(notifier.wait, threading.Event(), 0.2, 0.4) >= 0.3                       # subscribed now: it waits for the signal


def test_a_broken_connection_in_the_middle_of_listening_falls_back_to_polling_then_reconnects(notifiers):
    redis = FakeRedis()
    redis.next_pubsub_fails = True
    notifier = RedisNotifier(redis, retry_s=0.2)
    notifiers.append(notifier)
    notifier.start()
    assert eventually(lambda: notifier.active is False and redis.next_pubsub_fails is False)   # the first connection broke on its first read
    assert eventually(lambda: notifier.active)                                                 # and the second one is fine
    publisher = RedisNotifier(redis)
    notifiers.append(publisher)
    threading.Timer(0.2, publisher.publish).start()
    assert timed(notifier.wait, threading.Event(), 30.0, 30.0) < 3.0


def test_close_is_safe_after_anything_and_stops_the_threads():
    notifier = listening(FakeRedis())
    notifier.publish()
    notifier.close()
    notifier.close()
    assert not notifier._listener.is_alive() and not notifier._sender.is_alive()
    assert 0.15 <= timed(notifier.wait, threading.Event(), 0.2, 30.0) < 2.0               # after close it just polls
    RedisNotifier(FakeRedis()).close()                                                     # never started


# -- configuration --------------------------------------------------------------------------------------------------------------------
def test_no_url_means_no_notifier(monkeypatch):
    for name in ("SUBSTRATE_REDIS_URL", "SUBSTRATE_REDIS_URL_FILE"):
        monkeypatch.delenv(name, raising=False)
    assert type(notifier_from_environment()) is Notifier


def test_the_url_comes_from_the_environment_or_from_a_file_and_not_from_both(tmp_path):
    secret = tmp_path / "redis_url"
    secret.write_text(f"redis://:{PASSWORD}@redis:6379/0\n", encoding="utf-8")
    assert redis_url({"SUBSTRATE_REDIS_URL": "redis://h:1/0"}) == "redis://h:1/0"
    assert redis_url({"SUBSTRATE_REDIS_URL_FILE": str(secret)}) == f"redis://:{PASSWORD}@redis:6379/0"
    with pytest.raises(ValueError, match="not both"):
        redis_url({"SUBSTRATE_REDIS_URL": "redis://h:1/0", "SUBSTRATE_REDIS_URL_FILE": str(secret)})
    empty = tmp_path / "empty"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        redis_url({"SUBSTRATE_REDIS_URL_FILE": str(empty)})
    with pytest.raises(ValueError, match="cannot read") as error:
        redis_url({"SUBSTRATE_REDIS_URL_FILE": str(tmp_path / "missing")})
    assert PASSWORD not in str(error.value)


@pytest.mark.parametrize("url", ["http://redis:6379", "redis://", "redis:///0", "redis://h:6379/0#frag", "unix:///run/redis.sock", "redis://h:notaport/0", "redis://[::1/0"])
def test_a_url_that_is_not_a_redis_url_is_refused_without_echoing_it(url):
    url = url.replace("redis://h", f"redis://:{PASSWORD}@h")
    with pytest.raises(ValueError) as error:
        notifier_from_environment({"SUBSTRATE_REDIS_URL": url}, client=FakeRedis())
    assert PASSWORD not in str(error.value)


def test_a_good_url_gives_a_redis_notifier_with_short_timeouts(caplog):
    pytest.importorskip("redis")
    notifier = notifier_from_environment({"SUBSTRATE_REDIS_URL": f"redis://:{PASSWORD}@127.0.0.1:1/0"})
    assert isinstance(notifier, RedisNotifier)
    options = notifier.client.connection_pool.connection_kwargs
    assert options["socket_connect_timeout"] == 1.0 and options["socket_timeout"] == 2.0
    with caplog.at_level(logging.WARNING, logger="substrate.control.notify"):
        notifier.publish()                                            # nothing listens on port 1: logged, not raised
        assert notifier.drain(timeout=10)
    assert caplog.records and all(PASSWORD not in r.getMessage() for r in caplog.records)
    notifier.close()


def test_without_the_redis_package_the_message_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)                  # `import redis` now raises ImportError
    with pytest.raises(ImportError, match=r"substrate\[redis\]"):
        notifier_from_environment({"SUBSTRATE_REDIS_URL": "redis://127.0.0.1:6379/0"})


def test_the_cli_turns_a_bad_url_into_its_own_error(monkeypatch):
    from substrate.control import cli
    from substrate.errors import SubstrateError
    monkeypatch.setenv("SUBSTRATE_REDIS_URL", "http://nope")
    with pytest.raises(SubstrateError, match="must look like redis://"):
        cli._notifier()


# -- the orchestrator and the worker use it -------------------------------------------------------------------------------------------------
class Recording(Notifier):
    def __init__(self):
        self.published, self.waits, self.started = 0, [], 0

    def start(self):
        self.started += 1

    def publish(self):
        self.published += 1

    def wait(self, stop, poll_s, idle_s):
        self.waits.append((poll_s, idle_s))
        stop.set()                                                    # one idle round is enough


@pytest.fixture
def world(tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab")
    registry.set_quota(admin, "concurrent_runs", 50)
    notifier = Recording()
    orchestrator = Orchestrator(registry, Artifacts(registry, LocalArtifactStore(tmp_path / "store")), notifier=notifier)
    ada = registry.create_principal(tenant, "ada", Role.RESEARCHER)
    return {"registry": registry, "orch": orchestrator, "notifier": notifier, "ada": ada, "tmp": tmp_path}


def submit(world):
    spec = read_spec(EXPERIMENTS / "proton_transfer.yaml")
    experiment = world["registry"].add_experiment(world["ada"], "proton_transfer", spec)["id"]
    return world["orch"].submit(world["ada"], experiment, n_samples=0)["id"]


def test_submitting_a_run_publishes_once_and_a_refused_submission_publishes_nothing(world):
    submit(world)
    assert world["notifier"].published == 1
    with pytest.raises(LookupError):
        world["orch"].submit(world["ada"], "EXP-999999")
    assert world["notifier"].published == 1


def test_a_worker_publishes_after_each_task_because_that_task_may_have_unblocked_others(world):
    submit(world)
    before = world["notifier"].published
    assert Worker(world["orch"], queues=("cpu",)).run_until_idle() == 7
    assert world["notifier"].published - before == 7


def test_an_idle_worker_starts_the_listener_and_waits_through_the_notifier_with_both_intervals(world):
    stop = threading.Event()
    Worker(world["orch"], queues=("cpu",)).serve(stop, poll_s=0.25, idle_s=12.0)
    assert world["notifier"].started == 1 and world["notifier"].waits == [(0.25, 12.0)] and stop.is_set()


def test_the_default_orchestrator_has_no_notifier_and_workers_behave_as_before(tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    orchestrator = Orchestrator(registry, Artifacts(registry, LocalArtifactStore(tmp_path / "store")))
    assert type(orchestrator.notifier) is Notifier
    stop = threading.Event()
    threading.Timer(0.4, stop.set).start()
    assert timed(Worker(orchestrator, queues=("cpu",)).serve, stop, 0.2, 30.0) < 3.0           # polled every 0.2 s; the 30 s idle interval is for a connected notifier only


# -- real Redis ---------------------------------------------------------------------------------------------------------------------------
REAL = os.environ.get("SUBSTRATE_TEST_REDIS_URL")
needs_redis = pytest.mark.skipif(not REAL, reason="SUBSTRATE_TEST_REDIS_URL is not set (a throwaway Redis, for example: docker run -p 127.0.0.1:56379:6379 redis:7-alpine)")


@needs_redis
def test_real_redis_one_notifier_wakes_another(notifiers):
    pytest.importorskip("redis")
    waiter = notifier_from_environment({"SUBSTRATE_REDIS_URL": REAL})
    publisher = notifier_from_environment({"SUBSTRATE_REDIS_URL": REAL})
    notifiers += [waiter, publisher]
    assert waiter.active is False
    waiter.start()
    assert eventually(lambda: waiter.active)
    threading.Timer(0.3, publisher.publish).start()
    elapsed = timed(waiter.wait, threading.Event(), 30.0, 30.0)
    assert 0.2 <= elapsed < 5.0
    for _ in range(5):
        publisher.publish()
        publisher.drain()
    time.sleep(0.3)
    assert timed(waiter.wait, threading.Event(), 30.0, 30.0) < 2.0                  # the burst is one wake-up ...
    assert timed(waiter.wait, threading.Event(), 30.0, 0.8) >= 0.7                  # ... and nothing is left over


@needs_redis
def test_real_redis_a_wrong_password_degrades_to_polling_without_raising(caplog, notifiers):
    pytest.importorskip("redis")
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(REAL)
    wrong = urlunsplit((parts.scheme, f":not-the-password@{parts.hostname}:{parts.port}", parts.path, "", ""))
    notifier = notifier_from_environment({"SUBSTRATE_REDIS_URL": wrong})
    notifiers.append(notifier)
    with caplog.at_level(logging.WARNING, logger="substrate.control.notify"):
        notifier.publish()
        assert notifier.drain(timeout=10)
        assert 0.15 <= timed(notifier.wait, threading.Event(), 0.2, 30.0) < 3.0
        assert eventually(lambda: any("no subscription" in r.getMessage() for r in caplog.records))
    assert notifier.active is False
    assert all("not-the-password" not in r.getMessage() for r in caplog.records)


@needs_redis
def test_real_redis_a_run_is_picked_up_by_a_sleeping_worker_because_of_the_signal_and_not_the_poll(tmp_path):
    """The worker polls every 60 s and waits at most 60 s for a signal; the run is submitted while it sleeps. If the run is done within the test's 40 s, the signal woke
    the worker (after that first task it carries on by itself)."""
    pytest.importorskip("redis")
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab")
    ada = registry.create_principal(tenant, "ada", Role.RESEARCHER)
    artifacts = Artifacts(registry, LocalArtifactStore(tmp_path / "store"))
    worker_side = Orchestrator(registry, artifacts, notifier=notifier_from_environment({"SUBSTRATE_REDIS_URL": REAL}))
    api_side = Orchestrator(registry, artifacts, notifier=notifier_from_environment({"SUBSTRATE_REDIS_URL": REAL}))
    stop = threading.Event()
    worker = Worker(worker_side, queues=("cpu",))
    thread = threading.Thread(target=worker.serve, args=(stop, 60.0, 60.0), daemon=True)
    thread.start()
    try:
        assert eventually(lambda: worker_side.notifier.active, timeout=10)         # the worker has found nothing and is subscribed
        time.sleep(0.5)                                                              # ... and is inside its wait
        spec = read_spec(EXPERIMENTS / "proton_transfer.yaml")
        experiment = registry.add_experiment(ada, "proton_transfer", spec)["id"]
        run = api_side.submit(ada, experiment, n_samples=0)["id"]
        deadline = time.time() + 40
        while time.time() < deadline and api_side.describe(ada, run)["state"] not in ("succeeded", "failed"):
            time.sleep(0.25)
        assert api_side.describe(ada, run)["state"] == "succeeded"
    finally:
        stop.set()
        thread.join(timeout=5)
        worker_side.notifier.close()
        api_side.notifier.close()
