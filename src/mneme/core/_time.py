"""Shared UTC timestamp helpers for schema-versioned artifacts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mneme.core._errors import ValidationError


def utc_now_iso() -> str:
    """Return the current UTC time in the project's JSON timestamp format."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def require_utc_timestamp(
    value: object,
    field_name: str,
    *,
    error_type: type[Exception] = ValidationError,
) -> str:
    """Validate an ISO 8601 timestamp with an explicit UTC offset."""

    if not isinstance(value, str) or not value:
        raise error_type(f"{field_name} must be an ISO 8601 UTC timestamp")
    iso_text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError as exc:
        raise error_type(
            f"{field_name} must be an ISO 8601 UTC timestamp",
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise error_type(f"{field_name} must be an ISO 8601 UTC timestamp")
    return value
