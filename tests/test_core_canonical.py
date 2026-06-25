from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from uuid import UUID

import numpy as np
import pytest

from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    Transition,
    build_item,
    canonical_bytes,
    content_id,
)

EXPECTED_CANONICAL_HEX = (
    "03646f63000000000000058e03737472000000000000000b6d656d6f72795f697465"
    "6d05627974657300000000000000126d6e656d652e63616e6f6e6963616c2e763106"
    "7265636f7264000000000000054803737472000000000000000b6d656d6f72795f69"
    "74656d0000000603737472000000000000000e736368656d615f76657273696f6e03"
    "73747200000000000000146d6e656d652e6d656d6f72795f6974656d2e7631037374"
    "72000000000000000a656e636f6465725f6670067265636f72640000000000000120"
    "037374720000000000000013656e636f6465725f66696e6765727072696e74000000"
    "0503737472000000000000000e736368656d615f76657273696f6e03737472000000"
    "000000001c6d6e656d652e656e636f6465725f66696e6765727072696e742e763103"
    "737472000000000000000a656e636f6465725f696403737472000000000000000f65"
    "6e636f6465722e6669787475726503737472000000000000000d73756d6d6172697a"
    "65725f696403737472000000000000000c6d65616e5f706f6f6c2e76310373747200"
    "0000000000000e776569676874735f646967657374046e6f6e650000000000000000"
    "03737472000000000000000d636f6e6669675f646967657374037374720000000000"
    "00000d7368613235363a6162636465660373747200000000000000036b6579056172"
    "7261790000000000000035037374720000000000000007666c6f6174333200000001"
    "000000000000000205627974657300000000000000089a99193fcdcc4c3f03737472"
    "000000000000000a76616c75655f6b696e6403737472000000000000000a7472616e"
    "736974696f6e03737472000000000000000576616c7565067265636f726400000000"
    "0000025a03737472000000000000000a7472616e736974696f6e0000000803737472"
    "000000000000000e736368656d615f76657273696f6e037374720000000000000013"
    "6d6e656d652e7472616e736974696f6e2e76310373747200000000000000057a5f73"
    "72630561727261790000000000000045037374720000000000000007666c6f617433"
    "32000000020000000000000002000000000000000205627974657300000000000000"
    "100000803f000000400000404000008040037374720000000000000006616374696f"
    "6e0561727261790000000000000035037374720000000000000007666c6f61743332"
    "00000001000000000000000205627974657300000000000000080000803e000000bf"
    "0373747200000000000000067a5f6e65787405617272617900000000000000450373"
    "74720000000000000007666c6f617433320000000200000000000000020000000000"
    "00000205627974657300000000000000100000c03f00002040000060400000904003"
    "737472000000000000000564656c7461056172726179000000000000004503737472"
    "0000000000000007666c6f6174333200000002000000000000000200000000000000"
    "0205627974657300000000000000100000003f0000003f0000003f0000003f037374"
    "7200000000000000017403696e740000000000000001370373747200000000000000"
    "0a657069736f64655f69640562797465730000000000000010123456781234567812"
    "3456781234567803737472000000000000000672657761726404736f6d6500000000"
    "0000001807666c6f617436340000000000000008000000000000c03f037374720000"
    "0000000000046d657461016d000000000000009c0000000203737472000000000000"
    "00066e6573746564016d000000000000005700000002037374720000000000000001"
    "61046c697374000000000000001f0000000204626f6f6c000000000000000101046e"
    "6f6e6500000000000000000373747200000000000000016203696e74000000000000"
    "000132037374720000000000000006736f7572636503737472000000000000000766"
    "697874757265"
)
EXPECTED_CONTENT_ID_HEX = (
    "aa4896b7f28bb7865c5f3a6ee079d286e47b7bad0a3557572837a6b309c21271"
)


def _fingerprint(**overrides: object) -> EncoderFingerprint:
    kwargs = {
        "encoder_id": "encoder.fixture",
        "summarizer_id": "mean_pool.v1",
        "weights_digest": None,
        "config_digest": "sha256:abcdef",
    } | overrides
    return EncoderFingerprint(**kwargs)


def _transition(**overrides: object) -> Transition:
    z_src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    z_next = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    kwargs = {
        "z_src": z_src,
        "action": np.array([0.25, -0.5], dtype=np.float32),
        "z_next": z_next,
        "delta": z_next - z_src,
        "t": 7,
        "episode_id": UUID("12345678-1234-5678-1234-567812345678"),
        "reward": 0.125,
    } | overrides
    return Transition(**kwargs)


def _item(**overrides: object) -> MemoryItem:
    kwargs = {
        "content_id": None,
        "key": np.array([0.6, 0.8], dtype=np.float32),
        "value": _transition(),
        "meta": {"source": "fixture", "nested": {"b": 2, "a": [True, None]}},
        "encoder_fp": _fingerprint(),
    } | overrides
    return MemoryItem(**kwargs)


def test_canonical_bytes_and_content_id_match_golden_fixture() -> None:
    item = _item()

    assert len(canonical_bytes(item)) == 1434
    assert canonical_bytes(item).hex() == EXPECTED_CANONICAL_HEX
    assert content_id(item).hex() == EXPECTED_CONTENT_ID_HEX


def test_build_item_fills_content_id_and_content_id_field_is_excluded() -> None:
    item = _item()
    built = build_item(item.value, item.key, item.encoder_fp, item.meta)
    poisoned = replace(built, content_id=b"\xff" * 32)

    assert built.content_id == bytes.fromhex(EXPECTED_CONTENT_ID_HEX)
    assert content_id(built) == built.content_id
    assert content_id(poisoned) == built.content_id


def test_content_id_changes_when_identity_inputs_change() -> None:
    base = content_id(_item())

    changed_key = _item(key=np.array([0.8, 0.6], dtype=np.float32))
    changed_value = _item(value=_transition(t=8))
    changed_meta = _item(meta={"source": "other"})
    changed_fingerprint = _item(encoder_fp=_fingerprint(config_digest="sha256:fedcba"))

    assert content_id(changed_key) != base
    assert content_id(changed_value) != base
    assert content_id(changed_meta) != base
    assert content_id(changed_fingerprint) != base


def test_mapping_order_does_not_affect_canonical_bytes_or_digest() -> None:
    left = _item(meta={"z": 1, "a": {"b": 2, "a": [True, None]}})
    right = _item(meta={"a": {"a": [True, None], "b": 2}, "z": 1})

    assert canonical_bytes(left) == canonical_bytes(right)
    assert content_id(left) == content_id(right)


def test_content_id_is_stable_across_process_restarts() -> None:
    script = """
from uuid import UUID
import numpy as np
from mneme.core import EncoderFingerprint, MemoryItem, Transition, content_id
fp = EncoderFingerprint('encoder.fixture', 'mean_pool.v1', None, 'sha256:abcdef')
z_src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
z_next = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
transition = Transition(
    z_src=z_src,
    action=np.array([0.25, -0.5], dtype=np.float32),
    z_next=z_next,
    delta=z_next - z_src,
    t=7,
    episode_id=UUID('12345678-1234-5678-1234-567812345678'),
    reward=0.125,
)
item = MemoryItem(
    content_id=None,
    key=np.array([0.6, 0.8], dtype=np.float32),
    value=transition,
    meta={'nested': {'a': [True, None], 'b': 2}, 'source': 'fixture'},
    encoder_fp=fp,
)
print(content_id(item).hex())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == EXPECTED_CONTENT_ID_HEX


def test_unsupported_metadata_is_rejected_before_digesting() -> None:
    with pytest.raises((TypeError, ValueError), match="JSON-compatible"):
        build_item(
            _transition(),
            np.array([0.6, 0.8], dtype=np.float32),
            _fingerprint(),
            {"raw": b"bytes"},
        )


def test_canonical_bytes_rejects_unsupported_objects() -> None:
    with pytest.raises(TypeError, match="unsupported canonical object"):
        canonical_bytes(object())
