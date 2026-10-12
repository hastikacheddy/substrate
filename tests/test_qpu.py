"""The statevector simulator, the policy in front of quantum jobs, and the variational solver that stands in for the classical eigensolver."""
import math

import numpy as np
import pytest

from substrate import ResourceLimitError, ValidationError
from substrate.backends import ClassicalBackend
from substrate.qpu import (Circuit, Gate, QuantumPolicy, QuantumPolicyError, QuantumSolverBackend, StatevectorSimulator, backend_from_config, gate_matrix)

SIM = StatevectorSimulator()


def circuit(n, *gates):
    return Circuit(n, tuple(Gate(*g) if len(g) == 3 else Gate(g[0], g[1]) for g in gates))


# -- gates and states -----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["h", "x", "y", "z", "s", "t"])
def test_fixed_gates_are_unitary(name):
    u = gate_matrix(name)
    assert np.allclose(u @ u.conj().T, np.eye(2))


@pytest.mark.parametrize("name", ["rx", "ry", "rz"])
@pytest.mark.parametrize("theta", [0.0, 0.3, math.pi / 2, math.pi, -2.0, 7.0])
def test_rotations_are_unitary_and_compose_by_adding_angles(name, theta):
    u = gate_matrix(name, theta)
    assert np.allclose(u @ u.conj().T, np.eye(2)) and np.allclose(gate_matrix(name, theta) @ gate_matrix(name, 0.4), gate_matrix(name, theta + 0.4))


def test_known_identities():
    h, x, z = gate_matrix("h"), gate_matrix("x"), gate_matrix("z")
    assert np.allclose(h @ z @ h, x) and np.allclose(h @ x @ h, z) and np.allclose(gate_matrix("s") @ gate_matrix("s"), z) and np.allclose(gate_matrix("t") @ gate_matrix("t"), gate_matrix("s"))
    assert np.allclose(gate_matrix("ry", math.pi), np.array([[0, -1], [1, 0]])) and np.allclose(gate_matrix("rx", math.pi), -1j * x)


def test_an_empty_circuit_is_the_zero_state_and_x_flips_the_chosen_qubit():
    assert np.allclose(SIM.run(Circuit(2)), [1, 0, 0, 0])
    assert np.allclose(SIM.run(circuit(2, ("x", (0,)))), [0, 1, 0, 0])               # qubit 0 is the least significant bit
    assert np.allclose(SIM.run(circuit(2, ("x", (1,)))), [0, 0, 1, 0])
    assert np.allclose(SIM.run(circuit(3, ("x", (2,)), ("x", (0,)))), np.eye(8)[5])


def test_a_bell_state_has_the_right_amplitudes_correlations_and_statistics():
    bell = SIM.run(circuit(2, ("h", (0,)), ("cx", (0, 1))))
    assert np.allclose(bell, [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
    assert SIM.expectation(bell, {0: "Z", 1: "Z"}) == pytest.approx(1.0) and SIM.expectation(bell, {0: "X", 1: "X"}) == pytest.approx(1.0)
    assert SIM.expectation(bell, {0: "Z"}) == pytest.approx(0.0, abs=1e-12) and SIM.expectation(bell, {0: "Y", 1: "Y"}) == pytest.approx(-1.0)
    counts = SIM.sample(bell, 10_000, np.random.default_rng(1))
    assert set(counts) == {"00", "11"} and sum(counts.values()) == 10_000 and abs(counts["00"] - 5000) < 250


def test_a_ghz_state_and_the_direction_of_cx_and_cz():
    ghz = SIM.run(circuit(3, ("h", (0,)), ("cx", (0, 1)), ("cx", (1, 2))))
    assert np.allclose(np.abs(ghz) ** 2, [0.5, 0, 0, 0, 0, 0, 0, 0.5])
    assert np.allclose(SIM.run(circuit(2, ("x", (1,)), ("cx", (1, 0)))), np.eye(4)[3]) and np.allclose(SIM.run(circuit(2, ("x", (0,)), ("cx", (1, 0)))), np.eye(4)[1])
    plus = circuit(2, ("h", (0,)), ("h", (1,)), ("cz", (0, 1)))
    assert np.allclose(SIM.run(plus), np.array([1, 1, 1, -1]) / 2)


def test_gates_keep_the_state_normalised_and_unitary_evolution_is_reversible():
    gates = (("h", (0,)), ("ry", (1,), 0.7), ("cx", (0, 2)), ("rz", (2,), 1.9), ("t", (1,)), ("cz", (1, 2)), ("rx", (0,), -0.4))
    state = SIM.run(circuit(3, *gates))
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-14) and not np.allclose(state, np.eye(8)[0])
    reversible = (("h", (0,)), ("ry", (1,), 0.7), ("cx", (0, 2)), ("rz", (2,), 1.9), ("cz", (1, 2)), ("rx", (0,), -0.4))
    undo = (("rx", (0,), 0.4), ("cz", (1, 2)), ("rz", (2,), -1.9), ("cx", (0, 2)), ("ry", (1,), -0.7), ("h", (0,)))       # each gate's inverse, in reverse order
    assert np.allclose(SIM.run(circuit(3, *reversible, *undo)), np.eye(8)[0], atol=1e-12)                              # back to the zero state


def test_the_circuit_depth_counts_layers_not_gates():
    assert circuit(3, ("h", (0,)), ("h", (1,)), ("h", (2,))).depth == 1
    assert circuit(3, ("h", (0,)), ("cx", (0, 1)), ("cx", (1, 2))).depth == 3
    assert circuit(2, ("h", (0,)), ("x", (0,)), ("z", (0,)), ("h", (1,))).depth == 3 and Circuit(2).depth == 0


@pytest.mark.parametrize("build", [
    lambda: Circuit(0), lambda: Circuit(-1), lambda: Circuit(True), lambda: Circuit(2.5),
    lambda: circuit(2, ("swap", (0, 1))), lambda: circuit(2, ("h", (2,))), lambda: circuit(2, ("h", (-1,))), lambda: circuit(2, ("cx", (0, 0))), lambda: circuit(2, ("cx", (0,))),
    lambda: circuit(2, ("h", (0, 1))), lambda: circuit(2, ("rx", (0,))), lambda: circuit(2, ("h", (0,), 0.5)), lambda: circuit(2, ("rx", (0,), float("nan"))),
    lambda: circuit(2, ("rz", (0,), float("inf"))), lambda: Circuit(2, (Gate("h", (True,)),)),
])
def test_malformed_circuits_are_refused_when_built(build):
    with pytest.raises(ValidationError):
        build()


# -- the policy ----------------------------------------------------------------------------------------------------------------------------
def test_each_limit_of_the_policy_is_enforced_at_exactly_its_value():
    policy = QuantumPolicy(max_qubits=4, max_depth=5, max_gates=6, max_shots=7, max_circuits=8)
    policy.check(provider="simulator", qubits=4, depth=5, gates=6, shots=7, circuits=8)
    for what in ("qubits", "depth", "gates", "shots", "circuits"):
        sizes = dict(qubits=4, depth=5, gates=6, shots=7, circuits=8)
        sizes[what] += 1
        with pytest.raises(QuantumPolicyError, match=f"{sizes[what]} {what} exceeds the policy limit"):
            policy.check(provider="simulator", **sizes)
    with pytest.raises(QuantumPolicyError, match="negative"):
        policy.check(provider="simulator", qubits=1, shots=-1)


def test_a_provider_that_is_not_listed_is_refused_before_any_size_is_considered():
    policy = QuantumPolicy()
    with pytest.raises(QuantumPolicyError, match="'ibm_brisbane' is not allowed"):
        policy.check(provider="ibm_brisbane", qubits=1)
    assert QuantumPolicy(providers=("simulator", "ibm_brisbane"), hardware_providers=("ibm_brisbane",)).is_hardware("ibm_brisbane")
    assert not QuantumPolicy().is_hardware("simulator")


def test_a_policy_error_is_a_resource_limit_so_it_counts_as_a_security_matter_and_stops_ensembles():
    assert issubclass(QuantumPolicyError, ResourceLimitError) and not issubclass(QuantumPolicyError, ValidationError)


def test_the_simulator_checks_the_policy_before_it_runs_anything():
    tight = StatevectorSimulator(QuantumPolicy(max_qubits=2, max_shots=100))
    with pytest.raises(QuantumPolicyError, match="3 qubits"):
        tight.run(Circuit(3))
    with pytest.raises(QuantumPolicyError, match="shots"):
        tight.sample(np.array([1, 0, 0, 0], dtype=complex), 101, np.random.default_rng(0))
    tight.run(Circuit(2))


def test_a_circuit_over_the_depth_or_gate_limit_is_refused_by_the_simulator():
    gates = tuple(Gate("x", (0,)) for _ in range(10))
    with pytest.raises(QuantumPolicyError, match="depth"):
        StatevectorSimulator(QuantumPolicy(max_depth=9)).run(Circuit(1, gates))
    with pytest.raises(QuantumPolicyError, match="gates"):
        StatevectorSimulator(QuantumPolicy(max_gates=9)).run(Circuit(1, gates))


# -- the variational solver --------------------------------------------------------------------------------------------------------------------
def random_symmetric(shape, seed=0, scale=3.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=shape + (2, 2)) * scale
    return (a + np.swapaxes(a, -1, -2)) / 2


def test_the_quantum_solver_reproduces_the_classical_eigenvalues_and_vectors():
    h = random_symmetric((500,))
    e, v = QuantumSolverBackend().symmetric_eigh(h)
    ce, cv = ClassicalBackend().symmetric_eigh(h)
    assert np.max(np.abs(e - ce)) < 1e-12
    for k in (0, 1):
        assert np.min(np.abs(np.einsum("...i,...i->...", v[..., :, k], cv[..., :, k]))) > 1 - 1e-12       # the same vectors, up to sign
    assert np.allclose(np.einsum("...ik,...jk->...ij", v, v), np.eye(2), atol=1e-12) and np.all(e[..., 0] <= e[..., 1] + 1e-14)
    assert np.allclose(np.einsum("...ik,...k,...jk->...ij", v, e, v), h, atol=1e-11)                       # V diag(e) V^T is H


def test_it_handles_any_batch_shape_and_a_single_matrix():
    h = random_symmetric((7, 5), seed=1)
    e, v = QuantumSolverBackend().symmetric_eigh(h)
    assert e.shape == (7, 5, 2) and v.shape == (7, 5, 2, 2) and np.allclose(e, ClassicalBackend().symmetric_eigh(h)[0], atol=1e-12)
    single = QuantumSolverBackend().symmetric_eigh(np.array([[1.0, 0.5], [0.5, -1.0]]))
    assert single[0].shape == (2,) and np.allclose(single[0], [-math.sqrt(1.25), math.sqrt(1.25)])


@pytest.mark.parametrize("h", [
    np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([[2.0, 3.0], [3.0, 2.0]]), np.array([[-2.0, 3.0], [3.0, -2.0]]),
    np.array([[1.0, 1e-12], [1e-12, 1.0 + 1e-12]]), np.array([[1e6, 1.0], [1.0, -1e6]]), np.array([[5.0, 0.0], [0.0, -5.0]]), np.array([[-5.0, 0.0], [0.0, 5.0]]),
    np.array([[1e-9, 1e-9], [1e-9, -1e-9]]),
])
def test_degenerate_and_extreme_matrices_come_out_right(h):
    e, v = QuantumSolverBackend().symmetric_eigh(h)
    scale = max(float(np.abs(h).max()), 1e-300)
    assert np.allclose(e, np.linalg.eigvalsh(h), atol=1e-12 * scale, rtol=0) and np.allclose(v @ np.diag(e) @ v.T, h, atol=1e-11 * scale, rtol=0)


def test_the_energies_come_only_from_expectation_values_of_the_ansatz_not_from_an_eigensolver(monkeypatch):
    for name in ("eigh", "eigvalsh", "eig", "eigvals"):
        monkeypatch.setattr(np.linalg, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("the classical eigensolver was called")))
    backend = QuantumSolverBackend()
    e, _ = backend.symmetric_eigh(random_symmetric((20,)))
    assert e.shape == (20, 2) and backend.circuits_run == 20 * backend.EVALUATIONS == 20 * 57 and backend.jobs == 1


def test_it_counts_its_circuits_and_jobs():
    backend = QuantumSolverBackend()
    backend.symmetric_eigh(random_symmetric((10,)))
    backend.symmetric_eigh(random_symmetric((10,)))
    assert backend.jobs == 2 and backend.circuits_run == 2 * 10 * 57


@pytest.mark.parametrize("h, message", [
    (np.eye(3), "2x2"), (np.zeros((4, 3, 3)), "2x2"), (np.zeros(2), "2x2"), (np.zeros((2, 2, 1)), "2x2"),
    (np.array([[1.0, 2.0], [3.0, 1.0]]), "symmetric"), (np.array([[1.0, np.nan], [np.nan, 1.0]]), "NaN"), (np.array([[np.inf, 0.0], [0.0, 1.0]]), "NaN or infinite"),
])
def test_anything_that_is_not_a_real_symmetric_2x2_is_refused_not_solved_classically(h, message):
    with pytest.raises(ValidationError, match=message):
        QuantumSolverBackend().symmetric_eigh(h)


def test_the_solver_obeys_the_policy_on_the_number_of_circuits():
    backend = QuantumSolverBackend(QuantumPolicy(max_circuits=1000))
    with pytest.raises(QuantumPolicyError, match="circuits exceeds the policy limit of 1,000"):
        backend.symmetric_eigh(random_symmetric((100,)))
    assert backend.circuits_run == 0 and backend.jobs == 0                                           # refused before anything ran
    with pytest.raises(QuantumPolicyError, match="not allowed"):
        QuantumSolverBackend(QuantumPolicy(providers=("elsewhere",))).symmetric_eigh(random_symmetric((2,)))


def test_the_other_routines_of_the_backend_are_the_classical_ones():
    backend = QuantumSolverBackend()
    d, off = np.array([2.0, 3.0, 4.0]), np.array([-1.0, -1.0])
    assert np.allclose(backend.lowest_eigenpairs(d, off, 2)[0], ClassicalBackend().lowest_eigenpairs(d, off, 2)[0])
    assert backend.name == "quantum.simulator" and ClassicalBackend().name == "classical"


# -- the experiment's backend block -------------------------------------------------------------------------------------------------------------
def test_the_backend_block_picks_a_solver_and_refuses_what_is_not_provided():
    assert isinstance(backend_from_config(None), ClassicalBackend) and isinstance(backend_from_config({}), ClassicalBackend)
    assert isinstance(backend_from_config({"type": "classical", "provider": "scipy"}), ClassicalBackend)
    assert isinstance(backend_from_config({"type": "quantum"}), QuantumSolverBackend) and isinstance(backend_from_config({"type": "quantum", "provider": "simulator"}), QuantumSolverBackend)
    for bad in ({"type": "quantum", "provider": "ibm"}, {"type": "classical", "provider": "cuda"}, {"type": "hpc"}, {"type": "quantum", "provider": "simulator", "token": "x"},
                {"type": "quantum", "provider": "ibm_brisbane", "device": "ibm_brisbane"}):
        with pytest.raises(ValidationError):
            backend_from_config(bad)


def test_the_policy_given_to_the_backend_is_the_one_it_uses():
    policy = QuantumPolicy(max_circuits=50)
    assert backend_from_config({"type": "quantum"}, policy).policy is policy
