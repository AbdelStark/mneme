"""Content-id validation helpers shared by boundary parsers."""

from __future__ import annotations

from typing import Final

CID_SIZE: Final = 32


def require_cid_bytes(
    value: object,
    field_name: str,
    *,
    type_error: type[Exception],
    value_error: type[Exception],
) -> bytes:
    """Return a validated 32-byte content id."""

    if not isinstance(value, bytes):
        raise type_error(f"{field_name} must be bytes")
    if len(value) != CID_SIZE:
        raise value_error(f"{field_name} must be {CID_SIZE} bytes")
    return value


def cid_from_hex(
    value: object,
    field_name: str,
    *,
    error_type: type[Exception],
) -> bytes:
    """Parse and validate a hex-encoded content id."""

    if not isinstance(value, str) or not value:
        raise error_type(f"{field_name} must be a non-empty string")
    try:
        cid = bytes.fromhex(value)
    except ValueError as exc:
        raise error_type(f"{field_name} must be hex bytes") from exc
    return require_cid_bytes(
        cid,
        field_name,
        type_error=error_type,
        value_error=error_type,
    )
