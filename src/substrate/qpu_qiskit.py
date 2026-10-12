"""Qiskit as a provider for the quantum solver: Qiskit's ideal simulator, noisy emulators of real IBM devices, and (for a deployment that supplies credentials) the devices.

Needs the `qiskit` extra (qiskit, qiskit-aer, qiskit-ibm-runtime). Three providers, named in the experiment's `backend:` block and in the quantum policy:

* `qiskit-aer`   the ideal Aer simulator, sampling shots (no noise: the difference from `simulator` is only that the answer comes from finite shots).
* `qiskit-fake`  Aer with the noise model, coupling map and basis gates of a real IBM device (`device: sherbrooke`, ...): what a run on that device would roughly look like.
                 Nothing leaves the machine and no credentials are involved. It is an emulation, and the numbers it gives are the emulation's.
* `ibm`          IBM's devices, through qiskit-ibm-runtime (`device: <backend name>`). It is a hardware provider (it spends the submitter's `qpu_jobs_per_day` allowance, via
                 the gateway) and it needs a token that only the deployment supplies: $QISKIT_IBM_TOKEN or the file named by $QISKIT_IBM_TOKEN_FILE (and optionally
                 $QISKIT_IBM_INSTANCE and $QISKIT_IBM_CHANNEL). The token is never logged, never put in a message, and never given to a sandboxed step: a step run with
                 `--isolation process` has a scrubbed environment, so for that mode hardware is unavailable, by design; run the worker with `--isolation inline`.
                 A device cannot be seeded: a run on hardware cannot be reproduced to the last digit, and its reproduction will say so.
                 **This path has not been run on a real device**: it was written from the runtime's documented interface and is tested against a stand-in sampler.

Reproducibility: the emulators sample shots, and every shot count is a random draw. With no `seed` in the backend block each run draws differently (a reproduction of it will
differ at the level of the shot noise, which the tunnelling rates amplify: an electronic-structure experiment moved its final observables by about 20% from shot noise at 8000
shots); with `seed: <whole number>` the draws are fixed for the same Qiskit and Aer versions, and the run reproduces.

The solver measures <Z> and <X> of RY(theta)|0> at a few angles. For H = a I + b Z + c X (every real symmetric 2x2) the energy is E(theta) = a + b<Z> + c<X>, which is exactly
alpha + beta cos(theta) + gamma sin(theta); a least-squares fit of those three numbers to the measured energies gives the ground energy alpha - hypot(beta, gamma) and
the angle that reaches it. Noise damps beta and gamma, so on a noisy provider the ground energy comes out too high; `last_report` says how far the data are from the fit.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

from .backends import ClassicalBackend
from .errors import ValidationError
from .qpu import Circuit, Gate, QuantumConfigurationError, QuantumPolicy, QuantumProviderError, check_two_by_two

ANGLES = 8                                          # RY angles per call; three numbers are fitted, so this is five more measurements than are needed
DEFAULT_SHOTS = 4000
_DEVICE = re.compile(r"[a-z0-9_\-]{1,60}")


def _need_qiskit():
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
    except ImportError:
        raise QuantumConfigurationError("this quantum provider needs Qiskit: pip install 'substrate[qiskit]'") from None


def to_qiskit(circuit: Circuit, *, measure: bool = False):
    """A Qiskit QuantumCircuit with the same gates (names and qubit numbering are Qiskit's own, which this package's simulator follows), measured into one register if asked."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(circuit.n_qubits, circuit.n_qubits if measure else 0)
    for gate in circuit.gates:
        method = getattr(qc, gate.name)
        method(*(([gate.theta] if gate.theta is not None else []) + list(gate.qubits)))
    if measure:
        qc.measure(range(circuit.n_qubits), range(circuit.n_qubits))
    return qc


class QiskitProvider:
    """Runs measured circuits on a Qiskit backend (or sampler) and returns counts. The policy is checked before anything is built, transpiled or sent."""

    def __init__(self, backend, *, name: str, device: str | None, policy: QuantumPolicy, sampler=None, optimization_level: int = 1, seed: int | None = None):
        self.backend, self.name, self.device, self.policy, self.sampler = backend, name, device, policy, sampler
        self.optimization_level, self.seed = optimization_level, seed
        self.jobs = self.circuits_run = 0

    @property
    def hardware(self) -> bool:
        return self.policy.is_hardware(self.name)

    def counts(self, circuits: Sequence[Circuit], shots: int) -> list[dict[str, int]]:
        """Measurement counts, {bitstring: count} (qubit n-1 leftmost), for each circuit, from one job."""
        if not circuits:
            return []
        if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
            raise ValidationError(f"shots must be a positive whole number (got {shots!r})")
        self.policy.check(provider=self.name, qubits=max(c.n_qubits for c in circuits), depth=max(c.depth for c in circuits), gates=max(len(c.gates) for c in circuits),
                          shots=shots, circuits=len(circuits))
        from qiskit import transpile
        transpiled = transpile([to_qiskit(c, measure=True) for c in circuits], self.backend, optimization_level=self.optimization_level, seed_transpiler=self.seed)
        if self.sampler is None:
            options = {"seed_simulator": self.seed} if self.seed is not None else {}
            result = self.backend.run(transpiled, shots=shots, **options).result()
            found = [dict(result.get_counts(i)) for i in range(len(transpiled))]
        else:
            found = [dict(pub.data.c.get_counts()) for pub in self.sampler.run([(t,) for t in transpiled], shots=shots).result()]
        self.jobs += 1
        self.circuits_run += len(circuits)
        return found


def mean_z(counts: dict[str, int]) -> float:
    """<Z> of one qubit from its counts."""
    total = sum(counts.values())
    if total <= 0:
        raise ValidationError("a measurement returned no counts")
    return (counts.get("0", 0) - counts.get("1", 0)) / total


class QiskitSolverBackend(ClassicalBackend):
    """Diagonalises batches of real symmetric 2x2 matrices from measurements on a Qiskit provider (see the module docstring). The ODE and tridiagonal routines are inherited."""

    def __init__(self, provider: QiskitProvider, shots: int = DEFAULT_SHOTS):
        self.provider, self.shots, self.policy = provider, shots, provider.policy
        self.name = f"quantum.{provider.name}" + (f".{provider.device}" if provider.device else "")
        self.jobs = self.circuits_run = 0
        self.last_report: dict | None = None

    def _measure(self, thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """<Z> and <X> of RY(theta)|0> at each angle: 2 * len(thetas) circuits, one job."""
        circuits = []
        for theta in thetas:
            circuits.append(Circuit(1, (Gate("ry", (0,), float(theta)),)))                           # measured as is: <Z>
            circuits.append(Circuit(1, (Gate("ry", (0,), float(theta)), Gate("h", (0,)))))           # rotated first: <X>
        counts = self.provider.counts(circuits, self.shots)
        values = np.array([mean_z(c) for c in counts])
        self.jobs += 1
        self.circuits_run += len(circuits)
        return values[0::2], values[1::2]

    def symmetric_eigh(self, matrices):
        h = check_two_by_two(matrices)
        batch = h.shape[:-2]
        flat = h.reshape(-1, 2, 2)
        a, b, c = (flat[:, 0, 0] + flat[:, 1, 1]) / 2.0, (flat[:, 0, 0] - flat[:, 1, 1]) / 2.0, flat[:, 0, 1]           # H = a I + b Z + c X
        thetas = np.linspace(0.0, 2.0 * math.pi, ANGLES, endpoint=False)
        z, x = self._measure(thetas)
        energies = a[:, None] + b[:, None] * z[None, :] + c[:, None] * x[None, :]                                    # measured E(theta), per matrix
        design = np.stack([np.ones_like(thetas), np.cos(thetas), np.sin(thetas)], axis=1)
        alpha, beta, gamma = np.linalg.lstsq(design, energies.T, rcond=None)[0]
        ground = alpha - np.hypot(beta, gamma)
        theta_star = np.arctan2(-gamma, -beta)
        residual = energies - (alpha[:, None] + beta[:, None] * np.cos(thetas)[None, :] + gamma[:, None] * np.sin(thetas)[None, :])
        self.last_report = {"provider": self.provider.name, "device": self.provider.device, "hardware": self.provider.hardware, "shots": self.shots, "angles": ANGLES,
                            "circuits": 2 * ANGLES, "matrices": int(flat.shape[0]), "fit_rms": float(np.sqrt(np.mean(residual**2)))}
        excited = (flat[:, 0, 0] + flat[:, 1, 1]) - ground                                 # the trace is known, so the other eigenvalue follows
        psi0 = np.stack([np.cos(theta_star / 2.0), np.sin(theta_star / 2.0)], axis=-1)
        psi1 = np.stack([-psi0[:, 1], psi0[:, 0]], axis=-1)
        evals = np.stack([ground, excited], axis=-1).reshape(batch + (2,))
        evecs = np.stack([psi0, psi1], axis=-1).reshape(batch + (2, 2))
        return evals, evecs


# -- making a provider --------------------------------------------------------------------------------------------------------------------------
def _fake_device(device: str):
    from qiskit_ibm_runtime import fake_provider
    known = {n[4:].lower(): n for n in dir(fake_provider) if n.startswith("Fake") and n[4:5].isupper()}
    if device not in known:
        popular = [d for d in ("sherbrooke", "brisbane", "kyoto", "torino", "fez") if d in known] or sorted(known)[:5]
        raise QuantumConfigurationError(f"no emulated device '{device}' (qiskit-ibm-runtime has {len(known)}, for example {', '.join(popular)})")
    return getattr(fake_provider, known[device])()


def _token(environ) -> str:
    token, path = (environ.get("QISKIT_IBM_TOKEN") or "").strip(), (environ.get("QISKIT_IBM_TOKEN_FILE") or "").strip()
    if token and path:
        raise QuantumConfigurationError("set QISKIT_IBM_TOKEN or QISKIT_IBM_TOKEN_FILE, not both")
    if path:
        try:
            with open(path, encoding="utf-8") as handle:
                token = handle.read().strip()
        except OSError as error:
            raise QuantumConfigurationError(f"cannot read the IBM token from {path} ({type(error).__name__})") from None
    if not token:
        raise QuantumConfigurationError("the ibm provider needs a token in QISKIT_IBM_TOKEN or QISKIT_IBM_TOKEN_FILE (a step run in a sandbox does not get one: use --isolation inline for hardware)")
    return token


def make_provider(provider: str, device: str | None, policy: QuantumPolicy, environ=None, *, seed: int | None = None) -> QiskitProvider:
    _need_qiskit()
    import os
    environ = os.environ if environ is None else environ
    if device is not None and (not isinstance(device, str) or not _DEVICE.fullmatch(device)):
        raise ValidationError("a device name uses lower-case letters, digits, - and _ (up to 60 characters)")
    if provider == "qiskit-aer":
        from qiskit_aer import AerSimulator
        return QiskitProvider(AerSimulator(), name=provider, device="ideal", policy=policy, seed=seed)
    if provider == "qiskit-fake":
        if not device:
            raise ValidationError("the qiskit-fake provider needs a device (for example device: sherbrooke)")
        from qiskit_aer import AerSimulator
        return QiskitProvider(AerSimulator.from_backend(_fake_device(device)), name=provider, device=device, policy=policy, seed=seed)
    if provider == "ibm":
        if not device:
            raise ValidationError("the ibm provider needs a device (an IBM backend name)")
        token = _token(environ)
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            service = QiskitRuntimeService(channel=(environ.get("QISKIT_IBM_CHANNEL") or "ibm_quantum_platform"), token=token, instance=(environ.get("QISKIT_IBM_INSTANCE") or None))
            backend = service.backend(device)
            return QiskitProvider(backend, name=provider, device=device, policy=policy, sampler=SamplerV2(mode=backend), seed=seed)
        except Exception as error:                                # noqa: BLE001 (whatever the service raises, the message must not carry the token)
            raise QuantumProviderError(f"could not open IBM backend '{device}' ({type(error).__name__})") from None
    raise ValidationError(f"no Qiskit provider '{provider}'")


def qiskit_backend(provider: str, device: str | None, shots, policy: QuantumPolicy, environ=None, seed=None) -> QiskitSolverBackend:
    """The solver backend for a `backend:` block naming a Qiskit provider."""
    if shots is None:
        shots = DEFAULT_SHOTS
    if isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= policy.max_shots:
        raise ValidationError(f"shots must be a whole number from 1 to {policy.max_shots:,} (got {shots!r})")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31):
        raise ValidationError("seed must be a whole number from 0 to 2147483647")
    return QiskitSolverBackend(make_provider(provider, device, policy, environ, seed=seed), shots)
