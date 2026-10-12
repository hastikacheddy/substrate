"""Cases of the orchestrator that the first tests did not pin, found by mutation testing: what a transfer study may be asked, who may start one, and that starting a run and reproducing one are
separate permissions even though the shipped roles always hold them together."""
import pytest

pytest.importorskip("sqlalchemy")

from substrate import ValidationError  # noqa: E402
from substrate.control import security  # noqa: E402
from substrate.control.security import Action, AuthorizationError, PERMISSIONS, Role  # noqa: E402

from test_control_orchestrator import experiment, run_to_end, spec_of, work, world  # noqa: E402,F401


def a_transfer_study(world):
    first = experiment(world, "proton_transfer_electronic")
    second = experiment(world, "proton_transfer_electronic")
    return world["registry"].add_experiment(world["researcher"], "study", {"transfer": {"references": [first, second], "window_ev": None, "sigma_ev": 0.05}})["id"]


def test_a_transfer_study_cannot_be_given_an_ensemble_size(world):
    study = a_transfer_study(world)
    for n in (1, 5, 100):
        with pytest.raises(ValidationError, match="a transfer study has no ensemble"):
            world["orch"].submit(world["researcher"], study, n_samples=n)
    assert world["registry"].list_runs(world["researcher"]) == []                       # nothing was created, and nothing was charged
    assert world["registry"].used_today(world["researcher"], "runs_per_day") == 0


def test_only_those_who_may_run_transfer_studies_can_start_one(world):
    first = experiment(world, "proton_transfer_electronic")
    second = experiment(world, "proton_transfer_electronic")
    for person in (world["viewer"], world["operator"]):
        with pytest.raises(AuthorizationError, match="transfer.run"):
            world["orch"].submit_transfer(person, "study", [first, second])
    assert world["orch"].submit_transfer(world["researcher"], "study", [first, second])["state"] == "queued"
    assert len(world["registry"].list_experiments(world["researcher"])) == 3         # the refused attempts stored no study


def test_starting_a_run_and_reproducing_one_are_separate_permissions(world, monkeypatch):
    run_id, _ = run_to_end(world, "proton_transfer_electronic")
    exp = world["orch"].registry.get_run(world["researcher"], run_id)["experiment_id"]
    only_create = frozenset(PERMISSIONS[Role.RESEARCHER] - {Action.RUN_REPRODUCE})
    only_reproduce = frozenset(PERMISSIONS[Role.RESEARCHER] - {Action.RUN_CREATE})
    monkeypatch.setitem(security.PERMISSIONS, Role.RESEARCHER, only_create)
    assert world["orch"].submit(world["researcher"], exp, n_samples=0)["state"] == "queued"
    with pytest.raises(AuthorizationError, match="run.reproduce"):
        world["orch"].reproduce(world["researcher"], run_id)
    with pytest.raises(AuthorizationError, match="run.reproduce"):
        world["orch"].submit(world["researcher"], exp, n_samples=0, reproduces=run_id)
    monkeypatch.setitem(security.PERMISSIONS, Role.RESEARCHER, only_reproduce)
    with pytest.raises(AuthorizationError, match="run.create"):
        world["orch"].submit(world["researcher"], exp, n_samples=0)
    assert world["orch"].reproduce(world["researcher"], run_id)["reproduces"] == run_id
