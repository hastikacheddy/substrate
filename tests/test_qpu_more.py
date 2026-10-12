"""Cases of the simulator that the first tests did not pin, found by mutation testing: the exact matrices of Y and RZ, the printed order of qubits in a bitstring, which qubit an expectation
value belongs to, and the corners of circuit validation."""
import math

import numpy as np
import pytest

from substrate import ValidationError
from substrate.qpu import Circuit, Gate, StatevectorSimulator, gate_matrix

SIM = StatevectorSimulator()


def circuit(n, *gates):
    return Circuit(n, tuple(Gate(*g) if len(g) == 3 else Gate(g[0], g[1]) for g in gates))


def test_y_and_rz_are_the_textbook_matrices():
    assert np.allclose(gate_matrix("y"), [[0, -1j], [1j, 0]])
    assert np.allclose(gate_matrix("y") @ np.array([1, 0]), [0, 1j]) and np.allclose(gate_matrix("y") @ np.array([0, 1]), [-1j, 0])
    assert np.allclose(gate_matrix("y"), 1j * gate_matrix("x") @ gate_matrix("z"))                  # Y = iXZ
    for theta in (0.3, 1.0, -2.0, math.pi):
        assert np.allclose(gate_matrix("rz", theta), np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]))
    assert np.allclose(gate_matrix("rz", math.pi), -1j * gate_matrix("z"))
    # H RZ(pi) H |0> = -i X |0>: all the weight on |1>, which only the right sign and half-angle of RZ give
    state = SIM.run(circuit(1, ("h", (0,)), ("rz", (0,), math.pi), ("h", (0,))))
    assert np.allclose(np.abs(state) ** 2, [0, 1])


def test_a_gate_must_name_exactly_as_many_distinct_qubits_as_it_acts_on():
    for gate in (Gate("cx", (0, 1, 1)), Gate("x", (0, 1)), Gate("cx", (0,)), Gate("h", ()), Gate("cz", (0, 0))):
        with pytest.raises(ValidationError, match="acts on"):
            Circuit(3, (gate,))
    Circuit(3, (Gate("cx", (0, 2)), Gate("h", (1,))))


def test_bitstrings_are_printed_with_the_highest_qubit_on_the_left():
    state = SIM.run(circuit(3, ("x", (0,))))                                          # only qubit 0 is set
    assert SIM.sample(state, 50, np.random.default_rng(0)) == {"001": 50}
    assert SIM.sample(SIM.run(circuit(3, ("x", (2,)))), 50, np.random.default_rng(0)) == {"100": 50}
    assert SIM.sample(SIM.run(circuit(2, ("x", (1,)))), 10, np.random.default_rng(0)) == {"10": 10}


def test_a_state_that_is_not_normalised_is_sampled_in_proportion():
    counts = SIM.sample(np.array([2.0, 0.0, 0.0, 2.0], dtype=complex), 4000, np.random.default_rng(1))
    assert set(counts) == {"00", "11"} and sum(counts.values()) == 4000 and abs(counts["00"] - 2000) < 200


def test_an_expectation_value_belongs_to_the_qubit_it_names():
    state = SIM.run(circuit(2, ("x", (0,))))                                          # qubit 0 is |1>, qubit 1 is |0>
    assert SIM.expectation(state, {0: "Z"}) == pytest.approx(-1.0) and SIM.expectation(state, {1: "Z"}) == pytest.approx(1.0)
    assert SIM.expectation(state, {0: "Z", 1: "Z"}) == pytest.approx(-1.0)
    plus_on_one = SIM.run(circuit(2, ("h", (1,))))
    assert SIM.expectation(plus_on_one, {1: "X"}) == pytest.approx(1.0) and SIM.expectation(plus_on_one, {0: "X"}) == pytest.approx(0.0)
