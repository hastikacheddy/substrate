"""The artifact store: content addressing, checksum on every read, narrow accepted types, no pickles, tenant separation, size and quota limits."""
import io
import json

import numpy as np
import pytest

pytest.importorskip("sqlalchemy")

from substrate import ResourceLimitError, ValidationError  # noqa: E402
from substrate.control.artifacts import (Artifacts, ArtifactError, LocalArtifactStore, MEDIA_TYPES, S3ArtifactStore,  # noqa: E402
                                         check_content, sha256_bytes)
from substrate.control.registry import Registry  # noqa: E402
from substrate.control.security import AuthorizationError, QuotaExceededError, Role  # noqa: E402


@pytest.fixture
def world(tmp_path):
    registry = Registry(f"sqlite:///{tmp_path / 'r.sqlite3'}")
    tenant, admin, _ = registry.bootstrap("lab-a")
    other, boris, _ = registry.bootstrap("lab-b", "boris")
    store = LocalArtifactStore(tmp_path / "store")
    return {"registry": registry, "tenant": tenant, "admin": admin, "other": other, "boris": boris, "store": store, "tmp": tmp_path,
            "artifacts": Artifacts(registry, store), "viewer": registry.create_principal(tenant, "vera", Role.VIEWER),
            "operator": registry.create_principal(tenant, "olga", Role.OPERATOR)}


# -- the local store ----------------------------------------------------------------------------------------------------------------
def test_bytes_go_in_under_their_hash_and_come_out_unchanged(world):
    store = world["store"]
    sha = store.put("T-0001", b"hello")
    assert sha == sha256_bytes(b"hello") and store.exists("T-0001", sha) and store.get("T-0001", sha) == b"hello"
    assert not store.exists("T-0001", sha256_bytes(b"other"))


def test_the_same_bytes_are_stored_once_and_leave_no_temporary_files(world):
    store = world["store"]
    assert store.put("T-0001", b"same") == store.put("T-0001", b"same")
    files = [p for p in world["store"].root.rglob("*") if p.is_file()]
    assert len(files) == 1 and not any(p.name.endswith(".tmp") for p in files)


def test_many_threads_storing_the_same_bytes_at_once_all_succeed_and_leave_one_file(world):
    import threading
    errors, shas = [], []

    def store():
        try:
            for _ in range(10):
                shas.append(world["store"].put("T-0001", b"identical content " * 1000))
        except Exception as error:                                                     # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=store) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    files = [p for p in world["store"].root.rglob("*") if p.is_file()]
    assert not errors and len(set(shas)) == 1 and len(files) == 1 and not any(f.name.endswith(".tmp") for f in files)
    assert world["store"].get("T-0001", shas[0]) == b"identical content " * 1000


def test_tenants_have_separate_namespaces(world):
    sha = world["store"].put("T-0001", b"private")
    assert not world["store"].exists("T-0002", sha)
    with pytest.raises(ArtifactError, match="no artifact"):
        world["store"].get("T-0002", sha)


def test_a_file_changed_on_disk_is_detected_on_read_and_on_rewrite(world):
    store = world["store"]
    sha = store.put("T-0001", b"original")
    path = store._path("T-0001", sha)
    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="does not match its checksum"):
        store.get("T-0001", sha)
    with pytest.raises(ArtifactError, match="has been altered"):
        store.put("T-0001", b"original")                                      # would otherwise trust the damaged copy and skip the write


@pytest.mark.parametrize("tenant, sha", [("../x", "a" * 64), ("T-0001/../..", "a" * 64), ("T-0001", "../../etc/passwd"), ("T-0001", "A" * 64),
                                         ("T-0001", "a" * 63), ("T-0001", "a" * 64 + "/.."), ("", "a" * 64), ("T 0001", "a" * 64)])
def test_an_address_that_could_leave_the_store_is_refused(world, tenant, sha):
    for call in (world["store"].get, world["store"].exists):
        with pytest.raises(ArtifactError, match="malformed"):
            call(tenant, sha)


def test_reading_something_that_is_not_there_is_an_artifact_error(world):
    with pytest.raises(ArtifactError, match="no artifact"):
        world["store"].get("T-0001", "b" * 64)


# -- what is accepted ----------------------------------------------------------------------------------------------------------------
def npy(array, allow_pickle=False):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=allow_pickle)
    return buffer.getvalue()


def test_the_accepted_types_are_exactly_these():
    assert MEDIA_TYPES == {"application/json", "text/plain", "text/markdown", "text/csv", "application/x-npy"}


@pytest.mark.parametrize("media_type", ["application/octet-stream", "application/x-python-pickle", "application/zip", "text/html", "image/png", "", "application/json; charset=x"])
def test_other_types_are_refused(media_type):
    with pytest.raises(ValidationError, match="not accepted"):
        check_content(b"{}", media_type)


def test_good_content_of_each_accepted_type_passes():
    check_content(b'{"a": [1, 2.5, null]}', "application/json")
    check_content("héllo\nworld".encode(), "text/plain")
    check_content(b"a,b\n1,2\n", "text/csv")
    check_content(b"# title", "text/markdown")
    check_content(npy(np.arange(6.0).reshape(2, 3)), "application/x-npy")


@pytest.mark.parametrize("data", [b"{not json", b"", b'{"a": NaN}', b'{"a": Infinity}', b"\xff\xfe", pytest.param(b"[" * 100_000, id="deeply-nested")])
def test_json_that_is_not_json_or_holds_nan_is_refused(data):
    with pytest.raises(ValidationError, match="not valid JSON"):
        check_content(data, "application/json")


def test_text_must_be_utf8_without_nul_bytes():
    with pytest.raises(ValidationError, match="UTF-8"):
        check_content(b"\xff\xfe\x00", "text/plain")
    with pytest.raises(ValidationError, match="NUL"):
        check_content(b"abc\x00def", "text/csv")


def test_an_array_with_pickled_objects_or_random_bytes_is_refused_not_loaded():
    objects = npy(np.array([{"a": 1}, None], dtype=object), allow_pickle=True)
    with pytest.raises(ValidationError, match="NumPy array without pickled objects"):
        check_content(objects, "application/x-npy")
    with pytest.raises(ValidationError, match="NumPy array"):
        check_content(b"this is not an array", "application/x-npy")
    import pickle
    with pytest.raises(ValidationError):
        check_content(pickle.dumps({"a": 1}), "application/x-npy")


# -- the service: limits, quota, authorisation ------------------------------------------------------------------------------------------
def test_write_records_the_artifact_and_returns_a_reference(world):
    ref = world["artifacts"].write_json(world["tenant"], {"x": 1}, name="result.json", run_id="RUN-000001")
    assert (ref.tenant_id, ref.media_type, ref.name) == (world["tenant"], "application/json", "result.json") and ref.size == len(b'{"x":1}')
    meta = world["registry"].artifact_meta(world["admin"], ref.sha256)
    assert meta["run_id"] == "RUN-000001" and meta["size"] == ref.size
    assert world["artifacts"].read_json(world["tenant"], ref.sha256) == {"x": 1}


def test_arrays_round_trip_with_dtype_and_shape(world):
    array = np.linspace(0, 1, 12).reshape(3, 4).astype(np.float32)
    ref = world["artifacts"].write_array(world["tenant"], array, name="a.npy")
    back = world["artifacts"].read_array(world["tenant"], ref.sha256)
    assert back.dtype == np.float32 and back.shape == (3, 4) and np.array_equal(back, array)


def test_nan_and_infinity_survive_a_round_trip_as_floats_while_the_stored_json_stays_strict(world):
    value = {"a": [1.0, float("nan"), float("inf"), float("-inf")], "b": {"c": float("nan")}, "d": 2}
    ref = world["artifacts"].write_json(world["tenant"], value, name="odd.json")
    stored = json.loads(world["store"].get(world["tenant"], ref.sha256).decode())                  # strict JSON: no bare NaN
    assert stored["a"][1] == {"__float__": "nan"}
    back = world["artifacts"].read_json(world["tenant"], ref.sha256)
    assert back["a"][0] == 1.0 and np.isnan(back["a"][1]) and back["a"][2] == float("inf") and back["a"][3] == float("-inf")
    assert np.isnan(back["b"]["c"]) and back["d"] == 2


def test_a_dict_that_merely_looks_like_the_tag_is_left_alone():
    from substrate.control.artifacts import decode_nonfinite, encode_nonfinite
    for odd in ({"__float__": "banana"}, {"__float__": "nan", "extra": 1}):
        assert decode_nonfinite(encode_nonfinite(odd)) == odd


def test_an_artifact_over_the_size_limit_is_refused_before_it_is_checked_or_stored(world, monkeypatch):
    monkeypatch.setenv("SUBSTRATE_MAX_ARTIFACT_BYTES", "100")
    world["artifacts"].write(world["tenant"], b"x" * 100, name="ok.txt", media_type="text/plain")
    with pytest.raises(ResourceLimitError, match="size in bytes: 101"):
        world["artifacts"].write(world["tenant"], b"x" * 101, name="big.txt", media_type="text/plain")
    assert len([p for p in world["store"].root.rglob("*") if p.is_file()]) == 1


def test_the_tenants_storage_quota_blocks_a_write_and_is_audited_but_a_duplicate_costs_nothing(world):
    registry = world["registry"]
    registry.set_quota(world["admin"], "artifact_bytes_total", 30)
    world["artifacts"].write(world["tenant"], b"a" * 20, name="one.txt", media_type="text/plain")
    world["artifacts"].write(world["tenant"], b"a" * 20, name="again.txt", media_type="text/plain")      # same bytes: already paid for
    with pytest.raises(QuotaExceededError, match="would take the tenant to 40 of 30"):
        world["artifacts"].write(world["tenant"], b"b" * 20, name="two.txt", media_type="text/plain")
    assert registry.artifact_bytes_total(world["tenant"]) == 20
    blocked = [e for e in registry.events(f"audit:{world['tenant']}", world["tenant"]) if e["kind"] == "quota.blocked"]
    assert blocked[-1]["payload"]["metric"] == "artifact_bytes_total" and blocked[-1]["payload"]["result"] == "BLOCKED"
    # another tenant's storage is its own
    world["artifacts"].write(world["other"], b"c" * 25, name="elsewhere.txt", media_type="text/plain")


def test_a_scanner_can_refuse_a_write_and_sees_the_bytes_and_type(world):
    seen = []

    def scanner(data, media_type):
        seen.append((data, media_type))
        if b"forbidden" in data:
            raise ValidationError("content policy")

    artifacts = Artifacts(world["registry"], world["store"], scanner)
    artifacts.write(world["tenant"], b"fine", name="a", media_type="text/plain")
    with pytest.raises(ValidationError, match="content policy"):
        artifacts.write(world["tenant"], b"forbidden words", name="b", media_type="text/plain")
    assert seen == [(b"fine", "text/plain"), (b"forbidden words", "text/plain")]
    assert world["registry"].artifact_bytes_total(world["tenant"]) == 4


def test_reading_needs_the_role_and_the_tenant(world):
    ref = world["artifacts"].write(world["tenant"], b"data", name="d.txt", media_type="text/plain")
    data, meta = world["artifacts"].read(world["viewer"], ref.sha256)
    assert data == b"data" and meta["name"] == "d.txt"
    with pytest.raises(AuthorizationError):
        world["artifacts"].read(world["operator"], ref.sha256)                              # operators see runs, not data
    with pytest.raises(ArtifactError, match="no artifact"):
        world["artifacts"].read(world["boris"], ref.sha256)                                  # another tenant: indistinguishable from absent
    with pytest.raises(ArtifactError, match="no artifact"):
        world["artifacts"].read_for(world["other"], ref.sha256)


def test_an_artifact_known_to_the_registry_but_altered_in_the_store_is_not_served(world):
    ref = world["artifacts"].write(world["tenant"], b"data", name="d.txt", media_type="text/plain")
    world["store"]._path(world["tenant"], ref.sha256).write_bytes(b"evil")
    with pytest.raises(ArtifactError, match="checksum"):
        world["artifacts"].read(world["viewer"], ref.sha256)


# -- the S3-compatible store (against a minimal fake of the client API) ------------------------------------------------------
class NoSuchKey(Exception):
    pass


class FakeS3:
    def __init__(self):
        self.objects, self.puts = {}, 0

    def put_object(self, Bucket, Key, Body):                                                  # noqa: N803 (the boto3 signature)
        self.puts += 1
        self.objects[(Bucket, Key)] = bytes(Body)

    def get_object(self, Bucket, Key):                                                        # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):                                                       # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise NoSuchKey(Key)
        return {}


def test_the_s3_store_keeps_the_same_contract():
    client = FakeS3()
    store = S3ArtifactStore(client, "bucket", "prefix/")
    sha = store.put("T-0001", b"payload")
    assert store.put("T-0001", b"payload") == sha and client.puts == 1                         # an existing object is not written again
    assert store.exists("T-0001", sha) and not store.exists("T-0002", sha)
    assert store.get("T-0001", sha) == b"payload"
    assert list(client.objects) == [("bucket", f"prefix/T-0001/{sha[:2]}/{sha}")]
    with pytest.raises(ArtifactError, match="no artifact"):
        store.get("T-0001", "c" * 64)


def test_the_s3_store_detects_altered_objects_and_refuses_bad_addresses():
    client = FakeS3()
    store = S3ArtifactStore(client, "bucket")
    sha = store.put("T-0001", b"payload")
    client.objects[("bucket", store._key("T-0001", sha))] = b"changed"
    with pytest.raises(ArtifactError, match="checksum"):
        store.get("T-0001", sha)
    for tenant, address in (("../x", sha), ("T-0001", "../../x")):
        with pytest.raises(ArtifactError, match="malformed"):
            store.get(tenant, address)


def test_the_service_works_the_same_over_the_s3_store(world):
    artifacts = Artifacts(world["registry"], S3ArtifactStore(FakeS3(), "bucket"))
    ref = artifacts.write_json(world["tenant"], {"k": [1, 2]}, name="x.json")
    assert artifacts.read_json(world["tenant"], ref.sha256) == {"k": [1, 2]}
