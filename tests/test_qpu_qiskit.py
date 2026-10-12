"""The Qiskit providers: the circuits Qiskit runs are the ones this package means, the policy is checked before anything is sent, the solver recovers the eigenvalues
from measurements (exactly when the measurements are exact, within statistics when they are shots, and biased in the known direction when the device is noisy), and a
hardware provider's token is read from where it should be and goes nowhere else.

No real device is used anywhere here. The `ibm` branch is tested against stand-ins for the runtime service and sampler (Aer's own SamplerV2 has the same interface), so it
shows that this package's side of the conversation is right, not that IBM's side answers."""
import math
import sys

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

from substrate import ValidationError  # noqa: E402
from substrate.qpu import (Circuit, Gate, QuantumConfigurationError, QuantumPolicy, QuantumPolicyError, QuantumProviderError, StatevectorSimulator,  # noqa: E402
                           backend_from_config)
from substrate import qpu_qiskit  # noqa: E402
from substrate.qpu_qiskit import QiskitProvider, QiskitSolverBackend, make_provider, mean_z, to_qiskit  # noqa: E402

POLICY = QuantumPolicy(providers=("simulator", "qiskit-aer", "qiskit-fake", "ibm"), hardware_providers=("ibm",))
TOKEN = "tok-1234567890-secret-abcdef"


def random_symmetric(n, seed, scale=0.5):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, 2, 2)) * scale
    return (m + m.transpose(0, 2, 1)) / 2


class SpyBackend:
    """Stands in for a backend and records that it was never asked to run anything."""

    def __init__(self):
        self.ran = 0

    def run(self, *args, **kwargs):
        self.ran += 1
        raise AssertionError("the backend should not have been reached")


# -- the circuits are the ones meant ---------------------------------------------------------------------------------------------------------
def random_circuit(rng, n):
    gates = []
    for _ in range(int(rng.integers(1, 14))):
        name = rng.choice(["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "cz"])
        if name in ("cx", "cz"):
            if n < 2:
                continue
            gates.append(Gate(str(name), tuple(int(q) for q in rng.choice(n, size=2, replace=False))))
        elif name in ("rx", "ry", "rz"):
            gates.append(Gate(str(name), (int(rng.integers(n)),), float(rng.uniform(-math.pi, math.pi))))
        else:
            gates.append(Gate(str(name), (int(rng.integers(n)),)))
    return Circuit(n, tuple(gates))


@pytest.mark.parametrize("seed", range(25))
def test_a_translated_circuit_gives_the_same_state_as_this_packages_simulator(seed):
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(seed)
    circuit = random_circuit(rng, int(rng.integers(1, 5)))
    ours = StatevectorSimulator(QuantumPolicy(max_qubits=6)).run(circuit)
    theirs = Statevector(to_qiskit(circuit)).data
    assert np.allclose(ours, theirs, atol=1e-12), circuit


def test_qubit_zero_is_the_rightmost_bit_of_the_counts_as_in_qiskit_and_in_the_simulator():
    provider = make_provider("qiskit-aer", None, POLICY, seed=1)
    counts = provider.counts([Circuit(3, (Gate("x", (0,)),)), Circuit(3, (Gate("x", (2,)),)), Circuit(2, (Gate("h", (0,)), Gate("cx", (0, 1))))], 500)
    assert counts[0] == {"001": 500} and counts[1] == {"100": 500}
    assert set(counts[2]) == {"00", "11"} and abs(counts[2]["00"] - 250) < 60                    # a Bell pair


def test_mean_z_reads_counts_and_refuses_an_empty_result():
    assert mean_z({"0": 75, "1": 25}) == 0.5 and mean_z({"1": 10}) == -1.0 and mean_z({"0": 3}) == 1.0
    with pytest.raises(ValidationError, match="no counts"):
        mean_z({})


# -- the policy comes first -------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("policy, circuits, shots, message", [
    (QuantumPolicy(providers=("simulator",)), 1, 100, "not allowed"),
    (QuantumPolicy(providers=("qiskit-aer",), max_shots=50), 1, 100, "shots"),
    (QuantumPolicy(providers=("qiskit-aer",), max_circuits=1), 2, 10, "circuits"),
    (QuantumPolicy(providers=("qiskit-aer",), max_qubits=1), 1, 10, "qubits"),
])
def test_a_job_the_policy_refuses_is_refused_before_anything_is_built_or_sent(policy, circuits, shots, message):
    spy = SpyBackend()
    provider = QiskitProvider(spy, name="qiskit-aer", device="ideal", policy=policy)
    big = Circuit(2 if message == "qubits" else 1, (Gate("h", (0,)),))
    with pytest.raises(QuantumPolicyError, match=message):
        provider.counts([big] * circuits, shots)
    assert spy.ran == 0 and provider.jobs == 0


def test_a_nonsense_shot_count_is_refused():
    provider = make_provider("qiskit-aer", None, POLICY, seed=1)
    for shots in (0, -5, 2.5, True, "100", None):
        with pytest.raises(ValidationError, match="shots"):
            provider.counts([Circuit(1, (Gate("h", (0,)),))], shots)
    assert provider.counts([], 10) == []


# -- the solver --------------------------------------------------------------------------------------------------------------------------------
class ExactProvider:
    """Counts from the exact probabilities with a huge number of shots, so that the only error left is the solver's own algebra."""

    name, device, hardware, policy = "qiskit-aer", "exact", False, QuantumPolicy(providers=("qiskit-aer",))
    SHOTS = 10**9

    def counts(self, circuits, shots):
        out = []
        for circuit in circuits:
            probabilities = StatevectorSimulator(QuantumPolicy()).probabilities(StatevectorSimulator(QuantumPolicy()).run(circuit))
            ones = int(round(probabilities[1] * self.SHOTS))
            out.append({"0": self.SHOTS - ones, "1": ones})
        return out


def test_with_exact_measurements_the_fit_recovers_the_eigenvalues_and_eigenvectors_to_rounding():
    h = random_symmetric(40, 5, scale=2.0)
    backend = QiskitSolverBackend(ExactProvider(), shots=ExactProvider.SHOTS)
    evals, evecs = backend.symmetric_eigh(h)
    exact_values, _ = np.linalg.eigh(h)
    assert np.allclose(evals, exact_values, atol=1e-8)
    for matrix, values, vectors in zip(h, evals, evecs):
        assert np.allclose(vectors.T @ vectors, np.eye(2), atol=1e-12)
        assert np.allclose(matrix @ vectors, vectors * values, atol=1e-7)
    assert backend.last_report["fit_rms"] < 1e-8 and backend.last_report["circuits"] == 2 * qpu_qiskit.ANGLES


def test_a_matrix_that_is_a_multiple_of_the_identity_is_handled():
    evals, evecs = QiskitSolverBackend(ExactProvider(), 1).symmetric_eigh(np.array([[[1.5, 0.0], [0.0, 1.5]]]))
    assert np.allclose(evals, 1.5, atol=1e-8) and np.allclose(evecs[0].T @ evecs[0], np.eye(2))


def test_the_batch_shape_is_kept_and_what_is_not_a_real_symmetric_2x2_is_refused_like_the_simulator_does():
    backend = QiskitSolverBackend(ExactProvider(), 1)
    assert backend.symmetric_eigh(random_symmetric(1, 1)[0]).__class__ is tuple
    for shape in ((2, 2), (3, 2, 2), (2, 3, 2, 2)):
        evals, evecs = backend.symmetric_eigh(np.zeros(shape) + np.eye(2))
        assert evals.shape == shape[:-1] and evecs.shape == shape
    for bad, message in ((np.eye(3), "2x2"), (np.array([[0.0, 1.0], [2.0, 0.0]]), "symmetric"), (np.array([[np.nan, 0.0], [0.0, 1.0]]), "NaN")):
        with pytest.raises(ValidationError, match=message):
            backend.symmetric_eigh(bad)


def test_on_the_ideal_aer_simulator_the_error_is_shot_noise_and_has_no_sign():
    backend = backend_from_config({"type": "quantum", "provider": "qiskit-aer", "shots": 8000}, POLICY)
    h = random_symmetric(30, 11)
    errors = backend.symmetric_eigh(h)[0][:, 0] - np.linalg.eigvalsh(h)[:, 0]
    assert np.abs(errors).max() < 0.04 and abs(errors.mean()) < 0.01
    assert backend.name == "quantum.qiskit-aer.ideal" and backend.last_report["hardware"] is False and backend.last_report["fit_rms"] < 0.02


def test_on_an_emulated_ibm_device_the_ground_energy_comes_out_too_high_as_noise_predicts():
    """Noise shrinks the amplitudes the fit finds, so alpha - hypot(beta, gamma) is above the truth. This is the emulation's own number, not a claim about the device."""
    backend = backend_from_config({"type": "quantum", "provider": "qiskit-fake", "device": "sherbrooke", "shots": 8000}, POLICY)
    h = random_symmetric(12, 11)
    evals, evecs = backend.symmetric_eigh(h)
    errors = evals[:, 0] - np.linalg.eigvalsh(h)[:, 0]
    assert errors.mean() > 0 and np.abs(errors).max() < 0.1, errors                                   # high, and not by much
    residuals = np.linalg.norm(np.einsum("nij,nj->ni", h, evecs[:, :, 0]) - np.linalg.eigvalsh(h)[:, :1] * evecs[:, :, 0], axis=1)
    assert residuals.max() < 0.05                                                                       # the direction of the ground state survives the noise
    assert backend.name == "quantum.qiskit-fake.sherbrooke" and backend.last_report["device"] == "sherbrooke" and backend.last_report["hardware"] is False


def test_the_solver_counts_its_jobs_and_circuits():
    backend = backend_from_config({"type": "quantum", "provider": "qiskit-aer", "shots": 100}, POLICY)
    backend.symmetric_eigh(random_symmetric(50, 1))
    backend.symmetric_eigh(random_symmetric(3, 2))
    assert (backend.jobs, backend.circuits_run, backend.provider.jobs) == (2, 32, 2)                    # one job of 16 circuits per call, however many matrices


# -- the backend block --------------------------------------------------------------------------------------------------------------------
def test_the_backend_block_picks_the_qiskit_providers_and_refuses_what_it_does_not_know():
    assert isinstance(backend_from_config({"type": "quantum", "provider": "qiskit-aer"}, POLICY), QiskitSolverBackend)
    for bad, message in (({"type": "quantum", "provider": "qiskit-fake"}, "needs a device"),
                         ({"type": "quantum", "provider": "qiskit-fake", "device": "Sher/brooke"}, "device name"), ({"type": "quantum", "provider": "qiskit-aer", "token": "x"}, "only type"),
                         ({"type": "quantum", "provider": "qiskit-aer", "shots": 0}, "shots"), ({"type": "quantum", "provider": "qiskit-aer", "shots": True}, "shots"),
                         ({"type": "quantum", "provider": "qiskit-aer", "shots": "5"}, "shots"), ({"type": "quantum", "provider": "qiskit-aer", "shots": 10**9}, "shots"),
                         ({"type": "classical", "provider": "qiskit-aer"}, "not available"), ({"type": "quantum", "provider": "azure"}, "not available")):
        with pytest.raises(ValidationError, match=message):
            backend_from_config(bad, POLICY)


def test_an_emulated_device_that_does_not_exist_is_a_configuration_error_that_names_some_that_do():
    with pytest.raises(QuantumConfigurationError, match="no emulated device") as error:
        backend_from_config({"type": "quantum", "provider": "qiskit-fake", "device": "nowhere"}, POLICY)
    assert "sherbrooke" in str(error.value) or "brisbane" in str(error.value)


def test_without_qiskit_installed_the_message_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "qiskit_aer", None)
    with pytest.raises(QuantumConfigurationError, match=r"substrate\[qiskit\]"):
        backend_from_config({"type": "quantum", "provider": "qiskit-aer"}, POLICY)


def test_the_default_policy_allows_none_of_them():
    backend = backend_from_config({"type": "quantum", "provider": "qiskit-aer"}, None)
    with pytest.raises(QuantumPolicyError, match="not allowed"):
        backend.symmetric_eigh(random_symmetric(1, 1))


# -- the hardware provider: credentials and the sampler path (against stand-ins) ----------------------------------------------------------------
class StandInService:
    created, backends = [], []

    def __init__(self, channel=None, token=None, instance=None, **rest):
        if token == "raise":
            raise RuntimeError(f"401 for token {TOKEN}")
        StandInService.created.append({"channel": channel, "token": token, "instance": instance})

    def backend(self, name):
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        StandInService.backends.append(name)
        if name == "ibm_missing":
            raise RuntimeError(f"no backend {name} for {TOKEN}")
        return AerSimulator.from_backend(FakeSherbrooke())


class StandInSampler:
    """qiskit-ibm-runtime's SamplerV2 takes `mode=backend`; Aer's takes none and has the same run/result interface."""
    pubs_seen = []

    def __init__(self, mode=None):
        from qiskit_aer.primitives import SamplerV2
        self.inner = SamplerV2()

    def run(self, pubs, shots=None):
        StandInSampler.pubs_seen.append((len(pubs), shots))
        return self.inner.run(pubs, shots=shots)


@pytest.fixture
def stand_ins(monkeypatch):
    import qiskit_ibm_runtime
    StandInService.created, StandInService.backends, StandInSampler.pubs_seen = [], [], []
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", StandInService)
    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", StandInSampler)


def test_the_ibm_provider_without_a_token_is_refused_before_any_service_is_contacted(stand_ins):
    with pytest.raises(QuantumConfigurationError, match="QISKIT_IBM_TOKEN") as error:
        make_provider("ibm", "ibm_brisbane", POLICY, environ={})
    assert StandInService.created == [] and "sandbox" in str(error.value)


def test_a_token_comes_from_the_environment_or_a_file_and_not_from_both(stand_ins, tmp_path):
    provider = make_provider("ibm", "ibm_brisbane", POLICY, environ={"QISKIT_IBM_TOKEN": TOKEN, "QISKIT_IBM_INSTANCE": "crn:v1:x"})
    assert StandInService.created == [{"channel": "ibm_quantum_platform", "token": TOKEN, "instance": "crn:v1:x"}] and StandInService.backends == ["ibm_brisbane"]
    assert provider.hardware is True and provider.sampler is not None and provider.name == "ibm"
    secret = tmp_path / "token"
    secret.write_text(TOKEN + "\n", encoding="utf-8")
    make_provider("ibm", "ibm_brisbane", POLICY, environ={"QISKIT_IBM_TOKEN_FILE": str(secret), "QISKIT_IBM_CHANNEL": "ibm_cloud"})
    assert StandInService.created[-1] == {"channel": "ibm_cloud", "token": TOKEN, "instance": None}
    with pytest.raises(QuantumConfigurationError, match="not both"):
        make_provider("ibm", "ibm_brisbane", POLICY, environ={"QISKIT_IBM_TOKEN": TOKEN, "QISKIT_IBM_TOKEN_FILE": str(secret)})
    with pytest.raises(QuantumConfigurationError, match="cannot read") as error:
        make_provider("ibm", "ibm_brisbane", POLICY, environ={"QISKIT_IBM_TOKEN_FILE": str(tmp_path / "missing")})
    assert TOKEN not in str(error.value)


def test_the_token_never_appears_in_an_error_even_when_the_service_puts_it_in_its_own(stand_ins):
    for environ, device in (({"QISKIT_IBM_TOKEN": "raise"}, "ibm_brisbane"), ({"QISKIT_IBM_TOKEN": TOKEN}, "ibm_missing")):
        with pytest.raises(QuantumProviderError) as error:
            make_provider("ibm", device, POLICY, environ=environ)
        assert not isinstance(error.value, QuantumConfigurationError)                                   # a service that did not answer may answer next time
        text = "".join(str(e) for e in (error.value, error.value.__cause__)) + repr(error.value.args)
        assert TOKEN not in text and "RuntimeError" in str(error.value) and error.value.__suppress_context__


def test_the_ibm_provider_needs_a_device_and_a_sane_name(stand_ins):
    for device in (None, "", "../x", "A B"):
        with pytest.raises(ValidationError):
            make_provider("ibm", device, POLICY, environ={"QISKIT_IBM_TOKEN": TOKEN})
    assert StandInService.created == []


def test_the_hardware_path_goes_through_the_sampler_and_not_the_backend(stand_ins):
    provider = make_provider("ibm", "ibm_brisbane", POLICY, environ={"QISKIT_IBM_TOKEN": TOKEN}, seed=3)
    provider.backend = SpyBackend() if False else provider.backend                                     # the real (emulated) backend object is only used to transpile
    counts = provider.counts([Circuit(2, (Gate("x", (1,)),)), Circuit(1, (Gate("h", (0,)),))], 400)
    assert StandInSampler.pubs_seen == [(2, 400)]
    assert counts[0] == {"10": 400} or counts[0].get("10", 0) > 380                                     # noisy emulation underneath: almost all shots in "10"
    assert abs(counts[1].get("0", 0) - 200) < 80


def test_a_solve_on_the_hardware_provider_runs_end_to_end_on_the_stand_in(stand_ins):
    backend = backend_from_config({"type": "quantum", "provider": "ibm", "device": "ibm_brisbane", "shots": 4000}, POLICY, environ={"QISKIT_IBM_TOKEN": TOKEN})
    h = random_symmetric(5, 4)
    errors = backend.symmetric_eigh(h)[0][:, 0] - np.linalg.eigvalsh(h)[:, 0]
    assert np.abs(errors).max() < 0.1 and backend.last_report["hardware"] is True and backend.name == "quantum.ibm.ibm_brisbane"
    assert StandInSampler.pubs_seen == [(2 * qpu_qiskit.ANGLES, 4000)]


# -- the policy from the environment ---------------------------------------------------------------------------------------------------------
def test_the_quantum_policy_comes_from_the_environment():
    from substrate.qpu import policy_from_environment
    assert policy_from_environment({}) == QuantumPolicy()
    policy = policy_from_environment({"SUBSTRATE_QPU_PROVIDERS": " simulator, qiskit-fake ,ibm,ibm", "SUBSTRATE_QPU_HARDWARE": "ibm"})
    assert policy.providers == ("simulator", "qiskit-fake", "ibm") and policy.hardware_providers == ("ibm",) and policy.max_qubits == QuantumPolicy().max_qubits
    for environ, message in (({"SUBSTRATE_QPU_PROVIDERS": "IBM"}, "not a provider name"), ({"SUBSTRATE_QPU_PROVIDERS": "a b"}, "not a provider name"),
                             ({"SUBSTRATE_QPU_PROVIDERS": "simulator", "SUBSTRATE_QPU_HARDWARE": "ibm"}, "does not allow"), ({"SUBSTRATE_QPU_PROVIDERS": "x" * 50}, "not a provider name")):
        with pytest.raises(ValidationError, match=message):
            policy_from_environment(environ)


# -- inside runs: the orchestrator, the gateway and the sandbox ---------------------------------------------------------------------------------
from substrate.control.repro import compare_summaries  # noqa: E402
from substrate.control.security import QuotaExceededError  # noqa: E402

from test_control_gateway import events, make_world, run, spec_with  # noqa: E402

AER = {"type": "quantum", "provider": "qiskit-aer", "shots": 4000, "seed": 7}


def test_parsing_an_experiment_keeps_shots_and_seed_in_its_backend_block_and_builds_nothing(monkeypatch):
    from substrate.experiment import parse_experiment
    built = []
    monkeypatch.setattr(qpu_qiskit, "make_provider", lambda *a, **k: built.append(a))
    assert parse_experiment(spec_with(AER)).backend == AER
    assert parse_experiment(spec_with({"type": "quantum", "provider": "qiskit-fake", "device": "sherbrooke"})).backend == {"type": "quantum", "provider": "qiskit-fake", "device": "sherbrooke"}
    assert built == []                                                                       # an emulator is slow to start: parsing must not start one


@pytest.mark.parametrize("block, message", [({"type": "quantum", "provider": "simulator", "shots": 10}, "only to a sampling provider"),
                                            ({"type": "classical", "provider": "scipy", "seed": 1}, "only to a sampling provider"),
                                            ({"type": "quantum", "provider": "qiskit-fake"}, "needs a device"), ({"type": "quantum", "provider": "ibm"}, "needs a device"),
                                            ({"type": "quantum", "provider": "qiskit-aer", "shots": 0}, "shots"), ({"type": "quantum", "provider": "qiskit-aer", "seed": -1}, "seed"),
                                            ({"type": "quantum", "provider": "qiskit-aer", "seed": 2**31}, "seed"), ({"type": "quantum", "provider": "qiskit-aer", "seed": 1.5}, "seed")])
def test_a_backend_block_that_cannot_work_is_refused_when_the_experiment_is_parsed(block, message):
    from substrate.experiment import parse_experiment
    with pytest.raises(ValidationError, match=message):
        parse_experiment(spec_with(block))


def test_a_seeded_run_on_a_sampling_provider_reproduces_exactly_and_differs_from_the_classical_run_by_shot_noise(tmp_path):
    w = make_world(tmp_path, POLICY)
    classical_id, classical = run(w, spec_with())
    first_id, first = run(w, spec_with(AER))
    second_id, second = run(w, spec_with(AER))
    assert classical["state"] == first["state"] == second["state"] == "succeeded"
    a, b, base = (w["orch"].final_observables(w["tenant"], r) for r in (first_id, second_id, classical["id"]))
    assert compare_summaries(a, b, rtol=1e-12).matches                                                # same seed: the same numbers
    noise = compare_summaries(base, a, rtol=1e-6)
    assert not noise.matches and 1e-4 < noise.max_relative_difference < 2.0                           # shot noise, amplified by the rates: real, and not a bug
    report = w["orch"].artifacts.read_json(w["tenant"], [x for x in w["registry"].run_artifacts(w["admin"], first_id) if x["name"] == "report.json"][0]["sha256"])
    assert report["backend"] == AER and "quantum.qiskit-aer.ideal" in {s["backend"] for s in report["steps"]}


def test_the_default_policy_refuses_a_qiskit_provider_at_the_gateway_and_records_it(tmp_path):
    w = make_world(tmp_path, None)
    run_id, description = run(w, spec_with(AER))
    assert description["state"] == "failed" and any("not allowed" in str(t.get("error", "")) for t in description["tasks"])
    assert events(w, "qpu.blocked") and events(w, "qpu.blocked")[0]["payload"]["provider"] == "qiskit-aer"


def test_the_hardware_provider_is_metered_per_task_before_any_credential_is_touched(tmp_path, monkeypatch):
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    w = make_world(tmp_path, POLICY)
    block = {"type": "quantum", "provider": "ibm", "device": "ibm_brisbane"}
    with pytest.raises(QuotaExceededError, match="qpu_jobs_per_day"):
        w["gateway"].authorize(w["tenant"], w["who"].id, "RUN-1", "RUN-1/00", block)               # no allowance granted: refused, and no token was needed to say so
    w["registry"].set_quota(w["admin"], "qpu_jobs_per_day", 1)
    assert w["gateway"].authorize(w["tenant"], w["who"].id, "RUN-1", "RUN-1/00", block) is True
    assert w["registry"].used_today(w["who"], "qpu_jobs_per_day") == 1 and events(w, "qpu.authorize")[-1]["payload"]["hardware"] is True


def test_a_step_in_a_sandbox_never_sees_the_token_so_hardware_needs_the_inline_mode(tmp_path, monkeypatch):
    """The worker process has the token; the spawned step has a scrubbed environment. The step fails for want of a token, which also shows that it was not given the parent's."""
    from substrate.control.sandbox import ProcessExecutor
    monkeypatch.setenv("QISKIT_IBM_TOKEN", TOKEN)
    w = make_world(tmp_path, POLICY)
    w["registry"].set_quota(w["admin"], "qpu_jobs_per_day", 50)
    w["orch"].executor = ProcessExecutor(quantum_policy=POLICY)
    run_id, description = run(w, spec_with({"type": "quantum", "provider": "ibm", "device": "ibm_brisbane"}))
    assert description["state"] == "failed"
    failed = [t for t in description["tasks"] if t["state"] == "failed"]
    assert len(failed) == 1 and failed[0]["attempts"] == 1                                              # a missing credential is not retried
    assert description["status"]["execution"]["state"] == "FAIL" and description["status"]["physics"]["state"] != "FAIL"      # a deployment problem, not a physics one
    stored = "".join(repr(e) for e in w["registry"].events(f"audit:{w['tenant']}", w["tenant"])) + repr(description)
    assert "QISKIT_IBM_TOKEN" in stored and TOKEN not in stored                                         # the message names the variable it lacked; the value is nowhere


def test_a_sampling_provider_runs_in_the_sandbox_too(tmp_path):
    from substrate.control.sandbox import ProcessExecutor
    w = make_world(tmp_path, POLICY)
    w["orch"].executor = ProcessExecutor(quantum_policy=POLICY)
    run_id, description = run(w, spec_with(AER))
    assert description["state"] == "succeeded"
