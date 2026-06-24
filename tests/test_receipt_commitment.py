from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from blake3 import blake3

from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    ReceiptVerificationError,
    Transition,
    UnsupportedOperationError,
    content_id,
)
from mneme.receipts import (
    COMMITMENT_SCHEMA,
    CommitmentState,
    InclusionProof,
    load_commitment_state,
    save_commitment_state,
    verify_inclusion_proof,
)
from mneme.store import init_store, open_store

_GOLDEN_ROOTS = (
    "ecf590294a28f6b9b77f4c19ffcc3769448e494849021eab7be7faf58d73654a",
    "76d5c922c123a3bbd380cf6ab07d065551d5e7bf449fc6b935199bd0e806228a",
    "0c286999b06aeca5ea683e9bbdad9435050cb8ca1945ded09a0af8b25dcfeaac",
    "6dfd6a6c8e9a5358a3a21337fcb08932f033af5a547b02911c2e5d7b19358d33",
    "9815c874cd733c4fa88083682ffefb1a415086107da97a9e3ce7e7601b1e9a8f",
    "47e6817a0c64d1246b4444de594699fc74ef075071160af456ad2b6f5f375c46",
)


def _cids(count: int) -> tuple[bytes, ...]:
    return tuple(
        blake3(f"cid-{index}".encode("ascii")).digest() for index in range(count)
    )


def _fingerprint() -> EncoderFingerprint:
    return EncoderFingerprint(
        encoder_id="encoder.fixture",
        summarizer_id="meanpool-v1",
        weights_digest=None,
        config_digest="blake3:config",
    )


def _item(key_value: float, *, step: int = 0) -> MemoryItem:
    z_src = np.array([key_value, 0.0], dtype=np.float32)
    z_next = np.array([key_value + 1.0, 0.0], dtype=np.float32)
    return MemoryItem(
        content_id=None,
        key=np.array([key_value, 0.0], dtype=np.float32),
        value=Transition(
            z_src=z_src,
            action=np.array([0.1], dtype=np.float32),
            z_next=z_next,
            delta=z_next - z_src,
            t=step,
            episode_id=uuid4(),
        ),
        meta={"source": "commitment-fixture", "step": step},
        encoder_fp=_fingerprint(),
    )


def test_mmr_append_roots_match_golden_vectors() -> None:
    state = CommitmentState.empty()

    assert state.schema_version == COMMITMENT_SCHEMA
    assert state.root_hex == _GOLDEN_ROOTS[0]
    for index, cid in enumerate(_cids(5), start=1):
        state = state.append(cid)
        assert state.root_hex == _GOLDEN_ROOTS[index]
        assert state.item_count == index


def test_inclusion_proofs_verify_and_unknown_ids_fail() -> None:
    cids = _cids(5)
    state = CommitmentState.from_cids(cids)

    for cid in cids:
        proof = state.prove(cid)
        assert isinstance(proof, InclusionProof)
        assert verify_inclusion_proof(cid, proof, state.root)
        assert not verify_inclusion_proof(
            blake3(b"mutated").digest(),
            proof,
            state.root,
        )
        assert not verify_inclusion_proof(cid, proof, blake3(b"wrong-root").digest())

    with pytest.raises(ReceiptVerificationError, match="not committed"):
        state.prove(blake3(b"missing").digest())


def test_commitment_state_json_round_trips_and_validates(tmp_path: Path) -> None:
    state = CommitmentState.from_cids(_cids(3))
    path = tmp_path / "commitment.json"

    saved = save_commitment_state(path, state)
    loaded = load_commitment_state(saved)

    assert loaded == state
    assert loaded.to_json()["leaf_ids"] == [cid.hex() for cid in _cids(3)]

    data = state.to_json()
    data["root"] = blake3(b"wrong").hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReceiptVerificationError, match="root does not match"):
        load_commitment_state(path)


def test_store_commit_persists_mmr_sidecar_and_proves_value_log_order(
    tmp_path: Path,
) -> None:
    store = init_store(tmp_path / "store")
    cids = store.put_batch([_item(float(index), step=index) for index in range(3)])

    root = store.commit()
    manifest_json = json.loads((store.path / "manifest.json").read_text())
    sidecar = store.path / "receipts" / "commitment-mmr-v1.json"
    reopened = open_store(store.path)
    proofs = reopened.prove(cids)

    assert manifest_json["commitment"] == {
        "enabled": True,
        "backend": "mmr-v1",
        "root": root.hex(),
        "files": ["receipts/commitment-mmr-v1.json"],
    }
    assert sidecar.is_file()
    assert reopened.root() == root
    assert reopened.commitment_state().leaf_ids == tuple(cids)
    for cid, proof in zip(cids, proofs, strict=True):
        assert verify_inclusion_proof(cid, proof, root)

    with pytest.raises(ReceiptVerificationError, match="not committed"):
        reopened.prove([blake3(b"missing").digest()])


def test_uncommitted_store_root_and_prove_fail_closed(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")
    store.put(_item(1.0))

    with pytest.raises(UnsupportedOperationError, match="commitments"):
        store.root()
    with pytest.raises(UnsupportedOperationError, match="commitments"):
        store.prove([content_id(_item(2.0))])
