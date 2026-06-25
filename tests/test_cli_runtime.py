from __future__ import annotations

from io import StringIO

import pytest

from mneme.cli._runtime import JsonResult, print_json
from mneme.core import ValidationError


def test_json_result_copies_payload_and_keeps_wrapper_ok_authoritative() -> None:
    payload = {"schema_version": "mneme.fixture.v1"}

    result = JsonResult(ok=True, payload=payload)
    payload["schema_version"] = "mutated"

    assert result.to_json() == {
        "ok": True,
        "schema_version": "mneme.fixture.v1",
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"ok": 1}, "ok must be a bool"),
        ({"payload": []}, "payload must be a dict"),
        ({"payload": {"ok": False}}, "payload must not contain ok"),
        ({"payload": {"": "empty"}}, "payload keys must be non-empty strings"),
        ({"payload": {1: "numeric"}}, "payload keys must be non-empty strings"),
    ),
)
def test_json_result_constructor_rejects_malformed_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "ok": True,
        "payload": {"schema_version": "mneme.fixture.v1"},
    }
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        JsonResult(**values)


def test_print_json_emits_deterministic_pretty_json() -> None:
    stream = StringIO()

    print_json({"z": 1, "a": True}, stream)

    assert stream.getvalue() == '{\n  "a": true,\n  "z": 1\n}\n'


def test_print_json_rejects_nonfinite_numbers() -> None:
    stream = StringIO()

    with pytest.raises(ValueError, match="Out of range float values"):
        print_json({"metric": float("nan")}, stream)

    assert stream.getvalue() == ""
