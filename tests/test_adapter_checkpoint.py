from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mneme import __version__
from mneme.adapter import (
    ADAPTER_CHECKPOINT_METADATA_FILE,
    ADAPTER_CHECKPOINT_SCHEMA,
    DEFAULT_ADAPTER_WEIGHTS_FILE,
    AdapterCheckpoint,
    AdapterCheckpointMetadata,
    load_adapter_checkpoint,
    load_adapter_checkpoint_metadata,
    save_adapter_checkpoint_metadata,
)
from mneme.core import (
    ENCODER_FINGERPRINT_SCHEMA,
    EncoderFingerprint,
    FingerprintMismatchError,
    SchemaVersionError,
    ValidationError,
)


def _fingerprint(
    *,
    encoder_id: str = "encoder.fixture",
    config_digest: str = "blake3:config",
) -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id=encoder_id,
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest=config_digest,
    )


def _metadata() -> AdapterCheckpointMetadata:
    return AdapterCheckpointMetadata(
        adapter_kind="cross_attention",
        adapter_config={
            "latent_dim": 3,
            "hidden_dim": 4,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "tags": ["fixture", "checkpoint"],
        },
        base_fingerprint=_fingerprint(),
        training_report_uri="reports/adapter-training.json",
    )


def test_checkpoint_metadata_round_trips_json_fields() -> None:
    metadata = _metadata()

    payload = metadata.to_json()
    loaded = AdapterCheckpointMetadata.from_json(payload)

    assert payload["schema_version"] == ADAPTER_CHECKPOINT_SCHEMA
    assert payload["adapter_kind"] == "cross_attention"
    assert payload["weights_file"] == DEFAULT_ADAPTER_WEIGHTS_FILE
    assert payload["package_version"] == __version__
    assert payload["base_fingerprint"] == {
        "schema_version": ENCODER_FINGERPRINT_SCHEMA,
        "encoder_id": "encoder.fixture",
        "summarizer_id": "meanpool-v1",
        "weights_digest": None,
        "config_digest": "blake3:config",
    }
    assert loaded == metadata
    assert loaded.adapter_config["tags"] == ("fixture", "checkpoint")


def test_save_and_load_checkpoint_validates_weights_and_base_fingerprint(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    weights_path = tmp_path / metadata.weights_file
    weights_path.write_bytes(b"fixture weights")
    saved_path = save_adapter_checkpoint_metadata(tmp_path, metadata)

    checkpoint = load_adapter_checkpoint(
        tmp_path,
        expected_base_fingerprint=_fingerprint(),
    )

    assert saved_path == tmp_path / ADAPTER_CHECKPOINT_METADATA_FILE
    assert checkpoint.metadata == metadata
    assert checkpoint.metadata_path == saved_path
    assert checkpoint.weights_path == weights_path

    with pytest.raises(FingerprintMismatchError, match="base fingerprint"):
        load_adapter_checkpoint(
            tmp_path,
            expected_base_fingerprint=_fingerprint(encoder_id="other.encoder"),
        )


def test_adapter_checkpoint_constructor_normalizes_path_fields() -> None:
    metadata = _metadata()

    checkpoint = AdapterCheckpoint(
        metadata=metadata,
        metadata_path="adapter.json",
        weights_path="adapter.safetensors",
    )

    assert checkpoint.metadata == metadata
    assert checkpoint.metadata_path == Path("adapter.json")
    assert checkpoint.weights_path == Path("adapter.safetensors")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"metadata": object()}, "metadata must be AdapterCheckpointMetadata"),
        ({"metadata_path": object()}, "metadata_path must be a path-like value"),
        ({"metadata_path": ""}, "metadata_path must not be empty"),
        ({"weights_path": object()}, "weights_path must be a path-like value"),
        ({"weights_path": ""}, "weights_path must not be empty"),
    ),
)
def test_adapter_checkpoint_constructor_rejects_malformed_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "metadata": _metadata(),
        "metadata_path": Path("adapter.json"),
        "weights_path": Path("adapter.safetensors"),
    }
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        AdapterCheckpoint(**values)


def test_load_checkpoint_metadata_accepts_sidecar_path(tmp_path: Path) -> None:
    metadata = _metadata()
    sidecar = save_adapter_checkpoint_metadata(tmp_path / "custom.json", metadata)

    assert load_adapter_checkpoint_metadata(sidecar) == metadata


def test_load_checkpoint_rejects_missing_metadata_fields(tmp_path: Path) -> None:
    sidecar = tmp_path / ADAPTER_CHECKPOINT_METADATA_FILE
    payload = _metadata().to_json()
    payload.pop("training_report_uri")
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="missing fields: training_report_uri"):
        load_adapter_checkpoint_metadata(tmp_path)


def test_load_checkpoint_wraps_unreadable_metadata_path(tmp_path: Path) -> None:
    sidecar = tmp_path / ADAPTER_CHECKPOINT_METADATA_FILE
    sidecar.mkdir()

    with pytest.raises(ValidationError, match="metadata could not be read"):
        load_adapter_checkpoint_metadata(tmp_path)


def test_save_checkpoint_metadata_wraps_unwritable_path(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")

    with pytest.raises(ValidationError, match="metadata could not be written"):
        save_adapter_checkpoint_metadata(blocked_parent / "adapter.json", _metadata())


def test_save_checkpoint_metadata_wraps_runtime_serialization_errors(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    output = tmp_path / "reports" / "adapter.json"
    object.__setattr__(metadata, "adapter_config", {"bad": object()})

    with pytest.raises(
        ValidationError,
        match="metadata could not be serialized",
    ) as exc_info:
        save_adapter_checkpoint_metadata(output, metadata)

    assert isinstance(exc_info.value.__cause__, TypeError)
    assert not output.exists()
    assert not output.parent.exists()


def test_checkpoint_metadata_rejects_malformed_base_fingerprint_fields() -> None:
    payload = _metadata().to_json()
    base_fingerprint = payload["base_fingerprint"]
    assert isinstance(base_fingerprint, dict)
    base_fingerprint["config_digest"] = None

    with pytest.raises(ValidationError, match=r"base_fingerprint\.config_digest"):
        AdapterCheckpointMetadata.from_json(payload)


def test_load_checkpoint_rejects_nonstandard_json_constants(tmp_path: Path) -> None:
    sidecar = tmp_path / ADAPTER_CHECKPOINT_METADATA_FILE
    sidecar.write_text('{"schema_version": NaN}', encoding="utf-8")

    with pytest.raises(ValidationError, match="not valid JSON"):
        load_adapter_checkpoint_metadata(tmp_path)


def test_load_checkpoint_rejects_unsupported_schema() -> None:
    payload = _metadata().to_json()
    payload["schema_version"] = "mneme.adapter_checkpoint.v2"

    with pytest.raises(SchemaVersionError, match="unsupported adapter checkpoint"):
        AdapterCheckpointMetadata.from_json(payload)


def test_checkpoint_metadata_accepts_nested_relative_weight_path() -> None:
    metadata = AdapterCheckpointMetadata(
        adapter_kind="cross_attention",
        adapter_config={"latent_dim": 3},
        base_fingerprint=_fingerprint(),
        training_report_uri="reports/adapter-training.json",
        weights_file="weights/adapter.safetensors",
    )

    assert metadata.weights_file == "weights/adapter.safetensors"


@pytest.mark.parametrize(
    "weights_file",
    (
        "../adapter.safetensors",
        "/tmp/adapter.safetensors",
        "C:/Users/abdel/adapter.safetensors",
        "\\\\server\\share\\adapter.safetensors",
        "weights\\adapter.safetensors",
        "~/adapter.safetensors",
        "weights/",
    ),
)
def test_checkpoint_metadata_rejects_unsafe_weight_paths(weights_file: str) -> None:
    with pytest.raises(ValidationError, match="weights_file"):
        AdapterCheckpointMetadata(
            adapter_kind="cross_attention",
            adapter_config={"latent_dim": 3},
            base_fingerprint=_fingerprint(),
            training_report_uri="reports/adapter-training.json",
            weights_file=weights_file,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"adapter_kind": object()}, "adapter_kind must be a non-empty string"),
        ({"adapter_config": []}, "adapter_config must be a mapping"),
        ({"base_fingerprint": object()}, "base_fingerprint must be"),
        (
            {"training_report_uri": object()},
            "training_report_uri must be a non-empty string",
        ),
        ({"weights_file": object()}, "weights_file must be a non-empty string"),
        ({"weights_file": ""}, "weights_file must be a non-empty string"),
        ({"package_version": object()}, "package_version must be a non-empty string"),
    ),
)
def test_checkpoint_metadata_constructor_rejects_malformed_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "adapter_kind": "cross_attention",
        "adapter_config": {"latent_dim": 3},
        "base_fingerprint": _fingerprint(),
        "training_report_uri": "reports/adapter-training.json",
    }
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        AdapterCheckpointMetadata(**values)


def test_load_checkpoint_rejects_missing_weights_file(tmp_path: Path) -> None:
    save_adapter_checkpoint_metadata(tmp_path, _metadata())

    with pytest.raises(ValidationError, match="weights file not found"):
        load_adapter_checkpoint(tmp_path)
    with pytest.raises(ValidationError, match="require_weights must be a bool"):
        load_adapter_checkpoint(tmp_path, require_weights="yes")  # type: ignore[arg-type]

    checkpoint = load_adapter_checkpoint(tmp_path, require_weights=False)

    assert checkpoint.weights_path == tmp_path / DEFAULT_ADAPTER_WEIGHTS_FILE


def test_adapter_checkpoint_import_does_not_load_optional_ml_backends() -> None:
    script = (
        "import sys; "
        "from mneme.adapter import AdapterCheckpointMetadata; "
        "blocked = {'torch', 'faiss', 'cryptography', 'pydantic'}; "
        "loaded = sorted(blocked & set(sys.modules)); "
        "print(','.join(loaded)); "
        "raise SystemExit(1 if loaded else 0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
