"""Public error taxonomy and CLI exit-code mapping."""

from __future__ import annotations

from enum import IntEnum


class CliExitCode(IntEnum):
    """Documented command-line exit codes."""

    SUCCESS = 0
    USER_INPUT = 2
    DATA_VALIDATION = 3
    OPTIONAL_DEPENDENCY = 4
    INTERNAL = 5


class MnemeError(Exception):
    """Base class for all public Mneme errors."""


class ValidationError(MnemeError):
    """Raised when user data violates a public validation contract."""


class SchemaVersionError(ValidationError):
    """Raised when an object carries an unsupported schema version."""


class FingerprintMismatchError(ValidationError):
    """Raised when keys or queries use incompatible encoder fingerprints."""


class ShapeError(ValidationError):
    """Raised when an array, tensor, or latent has an invalid shape."""


class DTypeError(ValidationError):
    """Raised when an array, tensor, or latent has an invalid dtype."""


class StoreError(MnemeError):
    """Raised for store failures that are not more specific."""


class StoreCorruptionError(StoreError):
    """Raised when persisted store state is internally inconsistent."""


class TransactionError(StoreError):
    """Raised when a store transaction cannot complete or recover."""


class IndexError(MnemeError):
    """Raised for index failures that are not more specific."""


class IndexUnavailableError(IndexError):
    """Raised when a requested index backend is unavailable."""


class QueryError(MnemeError):
    """Raised when a query request violates query semantics."""


class EmptyStoreError(QueryError):
    """Raised when strict query policy rejects an empty store."""


class UnsupportedOperationError(MnemeError):
    """Raised when a store or backend does not support a requested operation."""


class ReceiptVerificationError(MnemeError):
    """Raised when receipt verification fails."""


class OptionalDependencyError(MnemeError):
    """Raised when a requested optional dependency is not installed."""

    def __init__(
        self,
        message: str,
        *,
        extra: str | None = None,
        package: str | None = None,
    ) -> None:
        super().__init__(message)
        self.extra = extra
        self.package = package


class EvaluationError(MnemeError):
    """Raised when evaluation report generation or validation fails."""


def cli_exit_code(error: BaseException | None = None) -> int:
    """Return the documented CLI exit code for an error.

    The helper is pure so CLI commands can use it without importing command
    modules. Backend-specific exceptions should be wrapped with ``raise ... from
    cause`` before reaching this helper.
    """

    if error is None:
        return int(CliExitCode.SUCCESS)
    if isinstance(error, OptionalDependencyError):
        return int(CliExitCode.OPTIONAL_DEPENDENCY)
    if isinstance(error, QueryError | UnsupportedOperationError):
        return int(CliExitCode.USER_INPUT)
    if isinstance(
        error, ValidationError | StoreCorruptionError | ReceiptVerificationError
    ):
        return int(CliExitCode.DATA_VALIDATION)
    return int(CliExitCode.INTERNAL)


__all__ = [
    "CliExitCode",
    "DTypeError",
    "EmptyStoreError",
    "EvaluationError",
    "FingerprintMismatchError",
    "IndexError",
    "IndexUnavailableError",
    "MnemeError",
    "OptionalDependencyError",
    "QueryError",
    "ReceiptVerificationError",
    "SchemaVersionError",
    "ShapeError",
    "StoreCorruptionError",
    "StoreError",
    "TransactionError",
    "UnsupportedOperationError",
    "ValidationError",
    "cli_exit_code",
]
