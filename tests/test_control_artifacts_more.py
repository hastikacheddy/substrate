"""Cases of the artifact service that the first tests did not pin, found by mutation testing: what is checked at the boundary, the exact byte at which the tenant's allowance runs out,
equal values getting equal names, and bytes that sit in the store without the registry's say-so never being served."""
import io

import numpy as np
import pytest

pytest.importorskip("sqlalchemy")

from substrate import ValidationError  # noqa: E402
from substrate.control.artifacts import ArtifactError, sha256_bytes  # noqa: E402
from substrate.control.security import AuthorizationError, QuotaExceededError  # noqa: E402

from test_control_artifacts import world  # noqa: E402,F401


def test_what_does_not_match_its_declared_type_is_refused_by_the_service_and_nothing_is_stored(world):
    artifacts, store = world["artifacts"], world["store"]
    for data, media_type in ((b"not json", "application/json"), (b'{"x": NaN}', "application/json"), (b"\xff\xfe", "text/plain"), (b"a\x00b", "text/csv"),
                             (b"garbage", "application/x-npy"), (b"x", "application/x-python-pickle")):
        with pytest.raises(ValidationError):
            artifacts.write(world["tenant"], data, name="n", media_type=media_type)
        assert not store.exists(world["tenant"], sha256_bytes(data))
    assert world["registry"].artifact_bytes_total(world["tenant"]) == 0


def test_the_last_byte_of_the_tenants_allowance_can_be_used_and_the_next_cannot(world):
    registry, artifacts = world["registry"], world["artifacts"]
    registry.set_quota(world["admin"], "artifact_bytes_total", 10)
    artifacts.write(world["tenant"], b"x" * 6, name="a", media_type="text/plain")
    artifacts.write(world["tenant"], b"y" * 4, name="b", media_type="text/plain")                # exactly 10 of 10
    with pytest.raises(QuotaExceededError, match="11 of 10"):
        artifacts.write(world["tenant"], b"z", name="c", media_type="text/plain")
    artifacts.write(world["tenant"], b"y" * 4, name="again", media_type="text/plain")            # bytes already held cost nothing, even when the tenant is full


def test_equal_json_values_get_the_same_checksum_whatever_order_their_keys_were_written_in(world):
    a = world["artifacts"].write_json(world["tenant"], {"b": 1, "a": {"y": 2, "x": 3}}, name="one")
    b = world["artifacts"].write_json(world["tenant"], {"a": {"x": 3, "y": 2}, "b": 1}, name="two")
    assert a.sha256 == b.sha256


def test_bytes_the_registry_does_not_know_the_tenant_owns_are_never_served(world):
    store, artifacts, tenant = world["store"], world["artifacts"], world["tenant"]
    sha = store.put(tenant, b'{"smuggled": true}')                                              # in the store, but nobody recorded it
    with pytest.raises(ArtifactError):
        artifacts.read_for(tenant, sha)
    with pytest.raises(ArtifactError):
        artifacts.read(world["admin"], sha)
    with pytest.raises(ArtifactError):
        artifacts.read_json(tenant, sha)
    world["registry"].record_artifact(tenant, sha, run_id=None, name="now recorded", media_type="application/json", size=18)
    assert artifacts.read_json(tenant, sha) == {"smuggled": True} and artifacts.read(world["admin"], sha)[0] == b'{"smuggled": true}'


def test_reading_through_the_service_needs_the_permission_to_read_artifacts(world):
    ref = world["artifacts"].write_json(world["tenant"], {"x": 1}, name="x")
    assert world["artifacts"].read(world["viewer"], ref.sha256)[1]["name"] == "x"                  # a viewer reads
    with pytest.raises(AuthorizationError):
        world["artifacts"].read(world["operator"], ref.sha256)                                  # an operator never reads the data itself


def test_a_pickled_array_that_got_into_the_store_is_not_unpickled_when_read(world):
    buffer = io.BytesIO()
    np.save(buffer, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    data = buffer.getvalue()
    sha = world["store"].put(world["tenant"], data)
    world["registry"].record_artifact(world["tenant"], sha, run_id=None, name="evil", media_type="application/x-npy", size=len(data))
    with pytest.raises(ValueError, match="pickle"):
        world["artifacts"].read_array(world["tenant"], sha)
