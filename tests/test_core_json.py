from __future__ import annotations

import pytest

from mneme.core._json import (
    dumps_strict_json,
    loads_strict_json,
    write_strict_json_file,
)


def test_dumps_strict_json_rejects_nonstandard_constants() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        dumps_strict_json({"metric": float("nan")})


def test_dumps_strict_json_rejects_non_string_object_keys() -> None:
    with pytest.raises(TypeError, match="JSON object keys must be strings"):
        dumps_strict_json({1: "coerced"})

    with pytest.raises(TypeError, match="JSON object keys must be strings"):
        dumps_strict_json({"nested": [{None: "coerced"}]})


def test_loads_strict_json_rejects_nonstandard_constants() -> None:
    with pytest.raises(ValueError, match="invalid JSON constant: NaN"):
        loads_strict_json('{"metric": NaN}')


def test_loads_strict_json_rejects_overflowed_numbers() -> None:
    with pytest.raises(ValueError, match="JSON number must be finite: 1e999"):
        loads_strict_json('{"metric": 1e999}')

    with pytest.raises(ValueError, match="JSON number must be finite: -1e999"):
        loads_strict_json('{"metrics": [-1e999]}')


def test_loads_strict_json_rejects_duplicate_object_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON object key: metric"):
        loads_strict_json('{"metric": 1, "metric": 2}')

    with pytest.raises(ValueError, match="duplicate JSON object key: case"):
        loads_strict_json('{"metrics": [{"case": 1, "case": 2}]}')


def test_strict_json_preserves_deterministic_formatting() -> None:
    assert dumps_strict_json({"b": 1, "a": True}, sort_keys=True, indent=2) == (
        '{\n  "a": true,\n  "b": 1\n}'
    )


def test_strict_json_exposes_ascii_escape_policy() -> None:
    payload = {"label": "caf\u00e9"}

    assert dumps_strict_json(payload) == '{"label": "caf\\u00e9"}'
    assert dumps_strict_json(payload, ensure_ascii=False) == '{"label": "caf\u00e9"}'


def test_write_strict_json_file_serializes_before_filesystem_side_effects(
    tmp_path,
) -> None:
    output = tmp_path / "reports" / "bad.json"

    with pytest.raises(ValueError, match="Out of range float values"):
        write_strict_json_file(output, {"metric": float("nan")}, indent=2)

    assert not output.exists()
    assert not output.parent.exists()


def test_write_strict_json_file_rejects_non_string_keys_before_side_effects(
    tmp_path,
) -> None:
    output = tmp_path / "reports" / "bad.json"

    with pytest.raises(TypeError, match="JSON object keys must be strings"):
        write_strict_json_file(output, {"nested": [{1: "coerced"}]}, indent=2)

    assert not output.exists()
    assert not output.parent.exists()


def test_write_strict_json_file_rejects_malformed_output_paths() -> None:
    with pytest.raises(ValueError, match="JSON output path must not be empty"):
        write_strict_json_file("", {"ok": True})

    with pytest.raises(TypeError, match="JSON output path must be a path-like value"):
        write_strict_json_file(object(), {"ok": True})  # type: ignore[arg-type]


def test_write_strict_json_file_rejects_bytes_pathlike_output() -> None:
    class BytesPath:
        def __fspath__(self) -> bytes:
            return b"out.json"

    with pytest.raises(TypeError, match="must resolve to a text path"):
        write_strict_json_file(BytesPath(), {"ok": True})  # type: ignore[arg-type]


def test_write_strict_json_file_writes_deterministic_json_with_newline(
    tmp_path,
) -> None:
    output = tmp_path / "reports" / "ok.json"

    written = write_strict_json_file(output, {"b": 1, "a": True}, sort_keys=True)

    assert written == output
    assert output.read_text(encoding="utf-8") == '{"a": true, "b": 1}\n'
