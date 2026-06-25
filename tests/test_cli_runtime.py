from __future__ import annotations

from io import StringIO

import pytest

from mneme.cli._runtime import print_json


def test_print_json_emits_deterministic_pretty_json() -> None:
    stream = StringIO()

    print_json({"z": 1, "a": True}, stream)

    assert stream.getvalue() == '{\n  "a": true,\n  "z": 1\n}\n'


def test_print_json_rejects_nonfinite_numbers() -> None:
    stream = StringIO()

    with pytest.raises(ValueError, match="Out of range float values"):
        print_json({"metric": float("nan")}, stream)

    assert stream.getvalue() == ""
