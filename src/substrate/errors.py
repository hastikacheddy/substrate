class SubstrateError(Exception):
    """Base class for all substrate errors."""


class NoPathError(SubstrateError):
    """No chain of registered translators/engines connects the requested scales."""


class AmbiguousPathError(SubstrateError):
    """Several distinct, equally short translator chains connect the scales; the caller must choose."""


class ValidationError(SubstrateError):
    """A model, input, or scale-to-scale transformation failed a validity check."""

    def __init__(self, message: str, issues: tuple = ()):
        super().__init__(message)
        self.issues = tuple(issues)


class QCError(SubstrateError):
    """A quantum-chemistry program could not be run or failed. An environment problem, not a validity problem, so it is
    deliberately not a ValidationError: ensembles drop invalid draws, but a missing program must stop the run."""


class QCUnavailableError(QCError):
    """No usable quantum-chemistry program was found."""


class ResourceLimitError(SubstrateError):
    """A request asks for more than the configured limits allow (a grid, an ensemble, a number of quantum-chemistry jobs). Deliberately not a
    ValidationError: ensembles drop invalid draws, but an oversized request must stop the run, not be skipped."""
