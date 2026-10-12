"""A quantum solver behind the engines' `SolverBackend` interface, and the policy that stands in front of anything that could cost real quantum time.

Three pieces:

* **A statevector simulator** for small circuits (H, X, Y, Z, S, T, rotations, CX, CZ), exact, in NumPy. Qubit q is bit q of the basis index (qubit 0 is the least significant),
  the convention Qiskit uses.
* **`QuantumPolicy`**, what a job may ask for: which providers, how many qubits, how deep, how many gates, how many shots, how many circuits. A job outside the policy
  is refused with a `QuantumPolicyError` *before* anything runs. Hardware providers sit behind the same check plus a per-user quota (see `control.gateway`). The
  simulator is the only provider in the base package; `qpu_qiskit` (extra `qiskit`) adds Qiskit's ideal simulator, noisy emulators of real IBM devices, and IBM's devices
  themselves for a deployment that supplies credentials.
* **`QuantumSolverBackend`**, which diagonalises the two-level valence-bond Hamiltonians the electronic engines build by a variational calculation on one qubit: an `RY(theta)`
  ansatz, a coarse scan to find the basin, then Newton steps with derivatives from the parameter-shift rule, every energy an expectation value taken from the simulated circuit.
  It refuses anything that is not a real symmetric 2x2 matrix instead of quietly using a classical routine, so a run that says "quantum" is one.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import numpy as np

from .backends import ClassicalBackend
from .errors import ResourceLimitError, SubstrateError, ValidationError

ONE_QUBIT = {"h", "x", "y", "z", "s", "t", "rx", "ry", "rz"}
TWO_QUBIT = {"cx", "cz"}
PARAMETRIC = {"rx", "ry", "rz"}


class QuantumPolicyError(ResourceLimitError):
    """A quantum job asked for more than the policy allows, or for a provider it does not allow. Nothing was run."""


class QuantumProviderError(SubstrateError):
    """A quantum provider could not be used: its service did not answer, or the backend asked for is not there. A retry may help. (An execution failure, not a physics one.)"""


class QuantumConfigurationError(QuantumProviderError):
    """A quantum provider is not set up in this deployment: no credentials, no Qiskit, no such emulated device. Nothing a retry can change."""


# -- circuits ------------------------------------------------------------------------------------------------------------------------------
def gate_matrix(name: str, theta: float | None = None) -> np.ndarray:
    c, s = (math.cos(theta / 2), math.sin(theta / 2)) if theta is not None else (0.0, 0.0)
    table = {
        "h": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2), "x": np.array([[0, 1], [1, 0]], dtype=complex),
        "y": np.array([[0, -1j], [1j, 0]], dtype=complex), "z": np.array([[1, 0], [0, -1]], dtype=complex), "s": np.array([[1, 0], [0, 1j]], dtype=complex),
        "t": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
        "rx": np.array([[c, -1j * s], [-1j * s, c]], dtype=complex), "ry": np.array([[c, -s], [s, c]], dtype=complex),
        "rz": np.array([[np.exp(-1j * (theta or 0.0) / 2), 0], [0, np.exp(1j * (theta or 0.0) / 2)]], dtype=complex),
    }
    return table[name]


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    theta: float | None = None


@dataclass(frozen=True)
class Circuit:
    n_qubits: int
    gates: tuple[Gate, ...] = ()

    def __post_init__(self):
        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, int) or self.n_qubits < 1:
            raise ValidationError(f"a circuit needs a whole number of qubits of at least 1 (got {self.n_qubits!r})")
        for gate in self.gates:
            if gate.name in ONE_QUBIT:
                arity = 1
            elif gate.name in TWO_QUBIT:
                arity = 2
            else:
                raise ValidationError(f"gate '{gate.name}' is not one of {', '.join(sorted(ONE_QUBIT | TWO_QUBIT))}")
            if len(gate.qubits) != arity or len(set(gate.qubits)) != arity or any(isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < self.n_qubits for q in gate.qubits):
                raise ValidationError(f"gate '{gate.name}' acts on {arity} different qubit(s) of 0..{self.n_qubits - 1} (got {gate.qubits})")
            if (gate.name in PARAMETRIC) != (gate.theta is not None):
                raise ValidationError(f"gate '{gate.name}' {'needs' if gate.name in PARAMETRIC else 'takes no'} angle")
            if gate.theta is not None and not math.isfinite(gate.theta):
                raise ValidationError(f"the angle of gate '{gate.name}' must be finite")

    @property
    def depth(self) -> int:
        """The number of layers if gates on different qubits run side by side."""
        busy = [0] * self.n_qubits
        for gate in self.gates:
            layer = 1 + max(busy[q] for q in gate.qubits)
            for q in gate.qubits:
                busy[q] = layer
        return max(busy, default=0)


# -- policy --------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantumPolicy:
    providers: tuple[str, ...] = ("simulator",)
    hardware_providers: tuple[str, ...] = ()           # providers that are physical devices: these also need a quota (see control.gateway)
    max_qubits: int = 12
    max_depth: int = 200
    max_gates: int = 2000
    max_shots: int = 100_000
    max_circuits: int = 10_000_000
    timeout_s: float = 60.0

    def check(self, *, provider: str, qubits: int, depth: int = 0, gates: int = 0, shots: int = 0, circuits: int = 1) -> None:
        """Raise QuantumPolicyError unless a job of this size, on this provider, is allowed."""
        if provider not in self.providers:
            raise QuantumPolicyError(f"quantum provider '{provider}' is not allowed (allowed: {', '.join(self.providers) or 'none'})")
        for what, value, limit in (("qubits", qubits, self.max_qubits), ("depth", depth, self.max_depth), ("gates", gates, self.max_gates), ("shots", shots, self.max_shots),
                                   ("circuits", circuits, self.max_circuits)):
            if value > limit:
                raise QuantumPolicyError(f"a quantum job with {value:,} {what} exceeds the policy limit of {limit:,}")
        if min(qubits, depth, gates, shots, circuits) < 0:
            raise QuantumPolicyError("a quantum job cannot have a negative size")

    def check_circuit(self, circuit: Circuit, *, provider: str, shots: int = 0, circuits: int = 1) -> None:
        self.check(provider=provider, qubits=circuit.n_qubits, depth=circuit.depth, gates=len(circuit.gates), shots=shots, circuits=circuits)

    def is_hardware(self, provider: str) -> bool:
        return provider in self.hardware_providers


# -- the simulator -------------------------------------------------------------------------------------------------------------------------
class StatevectorSimulator:
    def __init__(self, policy: QuantumPolicy | None = None):
        self.policy = policy or QuantumPolicy()

    def run(self, circuit: Circuit, shots: int = 0, circuits: int = 1) -> np.ndarray:
        """The final state vector (length 2**n, complex). The policy is checked first."""
        self.policy.check_circuit(circuit, provider="simulator", shots=shots, circuits=circuits)
        n = circuit.n_qubits
        state = np.zeros(2**n, dtype=complex)
        state[0] = 1.0
        indices = np.arange(2**n)
        for gate in circuit.gates:
            if gate.name in ONE_QUBIT:
                axis = n - 1 - gate.qubits[0]                       # qubit q is axis n-1-q of the C-ordered tensor
                tensor = state.reshape((2,) * n)
                state = np.moveaxis(np.tensordot(gate_matrix(gate.name, gate.theta), tensor, axes=([1], [axis])), 0, axis).reshape(-1)
            else:
                control, target = gate.qubits
                active = ((indices >> control) & 1) == 1
                if gate.name == "cx":
                    new = state.copy()
                    new[active] = state[indices[active] ^ (1 << target)]
                    state = new
                else:
                    state = np.where(active & (((indices >> target) & 1) == 1), -state, state)
        return state

    @staticmethod
    def probabilities(state: np.ndarray) -> np.ndarray:
        return np.abs(state) ** 2

    def sample(self, state: np.ndarray, shots: int, rng: np.random.Generator) -> dict[str, int]:
        """Measurement counts as {bitstring: count}, with qubit n-1 leftmost, as Qiskit prints them."""
        self.policy.check(provider="simulator", qubits=int(math.log2(len(state))), shots=shots)
        p = self.probabilities(state)
        counts = rng.multinomial(shots, p / p.sum())
        n = int(math.log2(len(state)))
        return {format(i, f"0{n}b"): int(c) for i, c in enumerate(counts) if c}

    @staticmethod
    def expectation(state: np.ndarray, paulis: dict[int, str]) -> float:
        """<state| P |state> for the Pauli string P given as {qubit: 'X'|'Y'|'Z'} (qubits not listed are untouched)."""
        n = int(math.log2(len(state)))
        tensor = state.reshape((2,) * n).copy()
        for qubit, letter in paulis.items():
            axis = n - 1 - qubit
            tensor = np.moveaxis(np.tensordot(gate_matrix(letter.lower()), tensor, axes=([1], [axis])), 0, axis)
        value = np.vdot(state, tensor.reshape(-1))
        return float(value.real)


def check_two_by_two(matrices) -> np.ndarray:
    """`matrices` as a float array of real symmetric 2x2 matrices (any batch shape), or ValidationError: what a one-qubit variational solver can diagonalise."""
    h = np.asarray(matrices, dtype=float)
    if h.ndim < 2 or h.shape[-2:] != (2, 2):
        raise ValidationError(f"the quantum solver diagonalises 2x2 Hamiltonians (one qubit); got matrices of shape {h.shape[-2:] if h.ndim >= 2 else h.shape}")
    if not np.all(np.isfinite(h)):
        raise ValidationError("the quantum solver was given a Hamiltonian with NaN or infinite entries")
    scale = max(float(np.max(np.abs(h))), 1e-300)
    if not np.allclose(h[..., 0, 1], h[..., 1, 0], rtol=0, atol=1e-10 * scale):
        raise ValidationError("the quantum solver needs real symmetric Hamiltonians")
    return h


_PROVIDER_NAME = re.compile(r"[a-z0-9][a-z0-9._\-]{0,39}")


def policy_from_environment(environ=None) -> QuantumPolicy:
    """The quantum policy a deployment asks for: $SUBSTRATE_QPU_PROVIDERS (comma-separated provider names, default `simulator`) and $SUBSTRATE_QPU_HARDWARE (the ones among
    them that are physical devices and so are charged to the submitter's quota; default none). Everything else keeps the policy's defaults. Raises ValidationError."""
    environ = os.environ if environ is None else environ

    def names(variable: str, default: tuple[str, ...]) -> tuple[str, ...]:
        raw = (environ.get(variable) or "").strip()
        if not raw:
            return default
        found = tuple(part.strip() for part in raw.split(",") if part.strip())
        for name in found:
            if not _PROVIDER_NAME.fullmatch(name):
                raise ValidationError(f"{variable}: '{name[:40]}' is not a provider name (lower-case letters, digits, . _ -)")
        return tuple(dict.fromkeys(found))
    providers, hardware = names("SUBSTRATE_QPU_PROVIDERS", ("simulator",)), names("SUBSTRATE_QPU_HARDWARE", ())
    if not set(hardware) <= set(providers):
        raise ValidationError(f"SUBSTRATE_QPU_HARDWARE names {sorted(set(hardware) - set(providers))}, which SUBSTRATE_QPU_PROVIDERS does not allow")
    return QuantumPolicy(providers=providers, hardware_providers=hardware)


# -- the variational solver --------------------------------------------------------------------------------------------------------------
class QuantumSolverBackend(ClassicalBackend):
    """Diagonalises batches of real symmetric 2x2 matrices on a simulated qubit. The ODE and tridiagonal routines are inherited unchanged (classical)."""

    name = "quantum.simulator"
    GRID = 16
    NEWTON_STEPS = 10
    EVALUATIONS = GRID + 4 * NEWTON_STEPS + 1          # per problem: the scan, then four energies per Newton step (E, E at +/- pi/2, E at +pi), then the final one

    def __init__(self, policy: QuantumPolicy | None = None):
        self.policy = policy or QuantumPolicy()
        self.simulator = StatevectorSimulator(self.policy)
        self.circuits_run = 0
        self.jobs = 0

    def _ry_states(self, theta: np.ndarray) -> np.ndarray:
        """The states RY(theta)|0> for a batch of angles, from the same gate matrix the circuit simulator uses."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.stack([c, s], axis=-1)

    def _energy(self, theta: np.ndarray, h00, h01, h11) -> np.ndarray:
        """<H> in RY(theta)|0>, per problem. This is the only way the energy is obtained: no eigenvalue routine is called."""
        psi = self._ry_states(theta)
        self.circuits_run += theta.size
        return h00 * psi[..., 0] ** 2 + 2.0 * h01 * psi[..., 0] * psi[..., 1] + h11 * psi[..., 1] ** 2

    def symmetric_eigh(self, matrices):
        h = check_two_by_two(matrices)
        batch = h.shape[:-2]
        size = int(np.prod(batch)) if batch else 1
        evaluations = size * self.EVALUATIONS                                  # what the policy is asked about is exactly what will be run
        self.policy.check(provider="simulator", qubits=1, depth=1, gates=1, circuits=evaluations)
        self.jobs += 1
        flat = h.reshape(size, 2, 2)
        h00, h01, h11 = flat[:, 0, 0], flat[:, 0, 1], flat[:, 1, 1]
        # 1. coarse scan of the ansatz angle: the basin of the minimum
        grid = np.linspace(0.0, 2.0 * math.pi, self.GRID, endpoint=False)
        energies = np.stack([self._energy(np.full(size, t), h00, h01, h11) for t in grid])
        theta = grid[np.argmin(energies, axis=0)]
        # 2. Newton steps; the derivatives are the parameter-shift rule, exact for a rotation gate: E' = (E(t + pi/2) - E(t - pi/2)) / 2, E'' = (E(t + pi) - E(t)) / 2
        for _ in range(self.NEWTON_STEPS):
            e = self._energy(theta, h00, h01, h11)
            first = 0.5 * (self._energy(theta + math.pi / 2, h00, h01, h11) - self._energy(theta - math.pi / 2, h00, h01, h11))
            second = 0.5 * (self._energy(theta + math.pi, h00, h01, h11) - e)
            step = np.where(second > 1e-300, first / np.where(second > 1e-300, second, 1.0), 0.0)
            theta = theta - step
        ground = self._energy(theta, h00, h01, h11)
        excited = (h00 + h11) - ground                                   # the trace is conserved: the other eigenvalue
        psi0 = self._ry_states(theta)
        psi1 = np.stack([-psi0[:, 1], psi0[:, 0]], axis=-1)              # orthogonal to the ground state
        evals = np.stack([ground, excited], axis=-1).reshape(batch + (2,))
        evecs = np.stack([psi0, psi1], axis=-1).reshape(batch + (2, 2))   # columns are eigenvectors
        return evals, evecs


QISKIT_PROVIDERS = ("qiskit-aer", "qiskit-fake", "ibm")
_DEVICE_NAME = re.compile(r"[a-z0-9_\-]{1,60}")


def check_backend_block(config) -> dict:
    """An experiment's `backend:` block, validated and normalised, without building anything (parsing an experiment must not start an emulator). Keeps `device`, and for the
    Qiskit providers `shots` and `seed`; refuses anything that is not a combination this package provides."""
    if config is not None and not isinstance(config, dict):
        raise ValidationError(f"`backend` must be a mapping with a type and a provider (got {type(config).__name__})")
    config = dict(config or {})
    unknown = set(config) - {"type", "provider", "device", "shots", "seed"}
    if unknown:
        raise ValidationError(f"a backend block takes only type, provider, device, shots and seed (got {sorted(unknown)})")
    kind = config.get("type", "classical")
    provider = config.get("provider", "scipy" if kind == "classical" else "simulator")
    sampling = kind == "quantum" and provider in QISKIT_PROVIDERS
    if not ((kind == "classical" and provider == "scipy") or (kind == "quantum" and provider == "simulator") or sampling):
        available = "classical/scipy, quantum/simulator, quantum/" + ", quantum/".join(QISKIT_PROVIDERS) + " (the last three need the qiskit extra)"
        raise ValidationError(f"backend {kind}/{provider} is not available (have: {available})")
    block = {"type": kind, "provider": provider}
    if "device" in config:
        block["device"] = str(config["device"])
    if sampling:
        if provider in ("qiskit-fake", "ibm") and not block.get("device"):
            raise ValidationError(f"the {provider} provider needs a device (for example device: {'sherbrooke' if provider == 'qiskit-fake' else 'an IBM backend name'})")
        if "device" in block and not _DEVICE_NAME.fullmatch(block["device"]):
            raise ValidationError("a device name uses lower-case letters, digits, - and _ (up to 60 characters)")
        shots, seed = config.get("shots"), config.get("seed")
        if "shots" in config:
            if isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= 10**7:
                raise ValidationError(f"shots must be a whole number from 1 to 10,000,000 (got {shots!r})")
            block["shots"] = shots
        if "seed" in config:
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
                raise ValidationError("seed must be a whole number from 0 to 2147483647")
            block["seed"] = seed
    elif "shots" in config or "seed" in config:
        raise ValidationError("shots and seed apply only to a sampling provider (" + ", ".join(QISKIT_PROVIDERS) + ")")
    return block


def backend_from_config(config: dict | None, policy: QuantumPolicy | None = None, environ=None):
    """The solver backend an experiment's `backend:` block asks for. Anything that is not a combination this package provides is refused. `environ` is where a hardware
    provider looks for its credentials (the process environment when None; a sandboxed step has none, by design)."""
    block = check_backend_block(config)
    kind, provider = block["type"], block["provider"]
    if kind == "classical":
        return ClassicalBackend()
    if provider == "simulator":
        return QuantumSolverBackend(policy)
    from .qpu_qiskit import qiskit_backend                          # imported only when asked for: Qiskit is an optional extra and a slow import
    return qiskit_backend(provider, block.get("device"), block.get("shots"), policy or QuantumPolicy(), environ, block.get("seed"))
