from __future__ import annotations

import pytest

from mneme.core import (
    CliExitCode,
    DTypeError,
    EmptyStoreError,
    EvaluationError,
    FingerprintMismatchError,
    IndexError,
    IndexUnavailableError,
    MnemeError,
    OptionalDependencyError,
    QueryError,
    ReceiptVerificationError,
    SchemaVersionError,
    ShapeError,
    StoreCorruptionError,
    StoreError,
    TransactionError,
    UnsupportedOperationError,
    ValidationError,
    cli_exit_code,
)


@pytest.mark.parametrize(
    "error_type",
    [
        ValidationError,
        SchemaVersionError,
        FingerprintMismatchError,
        ShapeError,
        DTypeError,
        StoreError,
        StoreCorruptionError,
        TransactionError,
        IndexError,
        IndexUnavailableError,
        QueryError,
        EmptyStoreError,
        UnsupportedOperationError,
        ReceiptVerificationError,
        OptionalDependencyError,
        EvaluationError,
    ],
)
def test_public_errors_subclass_mneme_error(error_type: type[MnemeError]) -> None:
    assert issubclass(error_type, MnemeError)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (None, CliExitCode.SUCCESS),
        (QueryError("bad k"), CliExitCode.USER_INPUT),
        (EmptyStoreError("empty"), CliExitCode.USER_INPUT),
        (UnsupportedOperationError("receipts unavailable"), CliExitCode.USER_INPUT),
        (ValidationError("bad item"), CliExitCode.DATA_VALIDATION),
        (SchemaVersionError("bad schema"), CliExitCode.DATA_VALIDATION),
        (FingerprintMismatchError("mismatch"), CliExitCode.DATA_VALIDATION),
        (ShapeError("bad shape"), CliExitCode.DATA_VALIDATION),
        (DTypeError("bad dtype"), CliExitCode.DATA_VALIDATION),
        (StoreCorruptionError("manifest mismatch"), CliExitCode.DATA_VALIDATION),
        (ReceiptVerificationError("bad proof"), CliExitCode.DATA_VALIDATION),
        (OptionalDependencyError("missing faiss"), CliExitCode.OPTIONAL_DEPENDENCY),
        (StoreError("disk failed"), CliExitCode.INTERNAL),
        (TransactionError("recovery failed"), CliExitCode.INTERNAL),
        (IndexUnavailableError("backend unavailable"), CliExitCode.INTERNAL),
        (EvaluationError("report failed"), CliExitCode.INTERNAL),
        (RuntimeError("not typed"), CliExitCode.INTERNAL),
    ],
)
def test_cli_exit_code_returns_documented_mapping(
    error: BaseException | None, expected: CliExitCode
) -> None:
    assert cli_exit_code(error) == int(expected)


def test_optional_dependency_error_records_package_and_extra() -> None:
    error = OptionalDependencyError(
        "FAISS is required for the approximate index",
        package="faiss-cpu",
        extra="index",
    )

    assert error.package == "faiss-cpu"
    assert error.extra == "index"


def test_backend_cause_is_preserved_through_exception_chaining() -> None:
    backend_error = RuntimeError("backend exploded")

    with pytest.raises(IndexUnavailableError) as raised:
        raise IndexUnavailableError("index backend failed") from backend_error

    assert raised.value.__cause__ is backend_error
