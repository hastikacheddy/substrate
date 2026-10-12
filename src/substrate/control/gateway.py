"""The gateway in front of quantum compute: every task that wants a quantum backend is checked against a policy, and, if the provider is a physical device, against the
submitter's quota, before it is allowed to start.

Workers never talk to a quantum provider on their own authority. The gateway decides, from the experiment's `backend:` block, whether the provider is allowed at all
(`QuantumPolicy.providers`), whether it counts as hardware (`hardware_providers`, which also spends one unit of the submitter's `qpu_jobs_per_day` allowance per task,
off until a tenant administrator grants some), and records every refusal as a security event. The size limits (qubits, depth, shots, circuits) are enforced where the
circuits are built, by the backend itself, against the same policy.

No hardware provider ships with this package, so in practice only the simulator passes; the quota path is exercised by the tests by declaring the simulator a device.
"""
from __future__ import annotations

from ..qpu import QuantumPolicy, QuantumPolicyError
from .registry import Registry


class QPUGateway:
    def __init__(self, registry: Registry, policy: QuantumPolicy | None = None):
        self.registry = registry
        self.policy = policy or QuantumPolicy()

    def authorize(self, tenant_id: str, principal_id: str, run_id: str, task_id: str, backend: dict) -> bool:
        """Allow or refuse the use of `backend` by this task. Returns whether a quantum backend is involved; raises QuantumPolicyError or QuotaExceededError to refuse."""
        if backend.get("type") != "quantum":
            return False
        provider = backend.get("provider", "simulator")
        try:
            self.policy.check(provider=provider, qubits=1)
        except QuantumPolicyError as error:
            self.registry.audit(tenant_id, "qpu.blocked", run=run_id, task=task_id, provider=provider, reason=str(error)[:300], result="BLOCKED")
            raise
        if self.policy.is_hardware(provider):
            principal = self.registry.principal_in_tenant(tenant_id, principal_id)
            if principal is None:
                raise QuantumPolicyError("the submitter of this run no longer exists, so its quantum allowance cannot be charged")
            self.registry.consume(principal, "qpu_jobs_per_day", 1)                  # raises QuotaExceededError (and writes the audit event) when used up
        self.registry.audit(tenant_id, "qpu.authorize", run=run_id, task=task_id, provider=provider, hardware=self.policy.is_hardware(provider))
        return True
