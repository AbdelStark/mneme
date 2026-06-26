from __future__ import annotations

import argparse
import json
from io import StringIO

import pytest

import mneme.cli.__main__ as cli_main_module
from mneme.cli._runtime import JsonResult, print_json, report_to_json
from mneme.core import CliExitCode, ValidationError


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


def test_report_to_json_copies_valid_payloads() -> None:
    payload = {"schema_version": "mneme.fixture.v1"}

    normalized = report_to_json(payload)
    payload["schema_version"] = "mutated"

    assert normalized == {"schema_version": "mneme.fixture.v1"}


@pytest.mark.parametrize("payload", ([], {"": "empty"}, {1: "numeric"}))
def test_report_to_json_rejects_malformed_payloads(payload: object) -> None:
    with pytest.raises(TypeError, match="command handler returned"):
        report_to_json(payload)


def test_main_wraps_output_serialization_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    parser = argparse.ArgumentParser(prog="mneme")
    parser.set_defaults(
        command="fixture",
        handler=lambda _args: {"schema_version": "mneme.fixture.v1", "bad": object()},
    )
    monkeypatch.setattr(cli_main_module, "_build_parser", lambda: parser)

    returncode = cli_main_module.main([], stdout=stream)

    assert returncode == int(CliExitCode.INTERNAL)
    payload = json.loads(stream.getvalue())
    assert payload["schema_version"] == "mneme.cli_error.v1"
    assert payload["ok"] is False
    assert payload["error_type"] == "TypeError"


def test_main_wraps_invalid_output_key_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    parser = argparse.ArgumentParser(prog="mneme")
    parser.set_defaults(command="fixture", handler=lambda _args: {"": "empty"})
    monkeypatch.setattr(cli_main_module, "_build_parser", lambda: parser)

    returncode = cli_main_module.main([], stdout=stream)

    assert returncode == int(CliExitCode.INTERNAL)
    payload = json.loads(stream.getvalue())
    assert payload["schema_version"] == "mneme.cli_error.v1"
    assert payload["ok"] is False
    assert payload["error_type"] == "TypeError"
