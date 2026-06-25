from __future__ import annotations

import pytest

from mneme.core._json import dumps_strict_json, loads_strict_json


def test_dumps_strict_json_rejects_nonstandard_constants() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        dumps_strict_json({"metric": float("nan")})


def test_loads_strict_json_rejects_nonstandard_constants() -> None:
    with pytest.raises(ValueError, match="invalid JSON constant: NaN"):
        loads_strict_json('{"metric": NaN}')


def test_strict_json_preserves_deterministic_formatting() -> None:
    assert dumps_strict_json({"b": 1, "a": True}, sort_keys=True, indent=2) == (
        '{\n  "a": true,\n  "b": 1\n}'
    )
