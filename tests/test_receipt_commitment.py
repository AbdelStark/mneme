from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from blake3 import blake3

from mneme.core import (
    EncoderFingerprint,
    MemoryItem,
    Metric,
    QuerySpec,
    ReceiptVerificationError,
    StoreCorruptionError,
    Transition,
    UnsupportedOperationError,
    ValidationError,
    content_id,
)
from mneme.receipts import (
    COMMITMENT_SCHEMA,
    CommitmentState,
    InclusionProof,
    QueryReceiptParams,
    RetrievalReceipt,
    build_retrieval_receipt,
    load_commitment_state,
    save_commitment_state,
    verify_inclusion_proof,
    verify_retrieval_receipt,
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


def _query_params() -> QueryReceiptParams:
    return QueryReceiptParams.from_query(
        QuerySpec(
            vector=np.array([1.0, 0.0], dtype=np.float32),
            k=1,
            metric=Metric.L2,
        )
    )


def _receipt_parts() -> tuple[bytes, bytes, InclusionProof, QueryReceiptParams]:
    cid = _cids(1)[0]
    state = CommitmentState.from_cids((cid,))
    return state.root, cid, state.prove(cid), _query_params()


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


def test_store_root_rejects_short_manifest_commitment_root(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")
    store.put(_item(1.0))
    store.commit()
    manifest_path = store.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["commitment"]["root"] = "00"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(StoreCorruptionError, match="commitment root must be 32 bytes"):
        open_store(store.path).root()


def test_retrieval_receipt_verifies_items_root_and_query(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")
    store.put_batch([_item(float(index), step=index) for index in range(3)])
    root = store.commit()
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=2,
        metric=Metric.L2,
        with_receipt=True,
    )

    retrieval = open_store(store.path).query(spec)

    assert isinstance(retrieval.receipt, RetrievalReceipt)
    assert retrieval.receipt.root == root
    assert retrieval.receipt.ids == tuple(
        item.content_id or content_id(item) for item in retrieval.items
    )
    assert verify_retrieval_receipt(
        retrieval.receipt,
        retrieval.items,
        root=root,
        query=spec,
    )
    reloaded = RetrievalReceipt.from_json(retrieval.receipt.to_json())
    assert verify_retrieval_receipt(reloaded, retrieval.items, root=root, query=spec)


def test_retrieval_receipt_fails_for_altered_items_and_root(
    tmp_path: Path,
) -> None:
    store = init_store(tmp_path / "store")
    store.put_batch([_item(float(index), step=index) for index in range(2)])
    root = store.commit()
    spec = QuerySpec(
        vector=np.array([0.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
        with_receipt=True,
    )
    retrieval = open_store(store.path).query(spec)
    assert isinstance(retrieval.receipt, RetrievalReceipt)

    item = retrieval.items[0]
    altered_item = replace(item, meta={**dict(item.meta), "tampered": True})

    assert not verify_retrieval_receipt(
        retrieval.receipt,
        (altered_item,),
        root=root,
        query=spec,
    )
    assert not verify_retrieval_receipt(
        retrieval.receipt,
        retrieval.items,
        root=blake3(b"wrong-root").digest(),
        query=spec,
    )
    assert not verify_retrieval_receipt(
        retrieval.receipt,
        retrieval.items,
        root=root,
        query=replace(spec, k=2),
    )


def test_signed_retrieval_receipts_fail_closed_until_signing_backend_exists(
    tmp_path: Path,
) -> None:
    store = init_store(tmp_path / "store")
    (cid,) = store.put_batch([_item(1.0)])
    root = store.commit()
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
    )
    receipt = build_retrieval_receipt(
        root=root,
        ids=(cid,),
        proofs=tuple(store.prove([cid])),
        query=spec,
        store_id=str(store.manifest.store_id),
        signer="ed25519:test-key",
        signature=b"unsigned-fixture-signature",
    )

    reloaded = RetrievalReceipt.from_json(receipt.to_json())

    assert reloaded.signer == "ed25519:test-key"
    assert reloaded.signature == b"unsigned-fixture-signature"
    assert not verify_retrieval_receipt(receipt, root=root, query=spec)
    assert not verify_retrieval_receipt(reloaded, root=root, query=spec)


def test_retrieval_receipt_direct_constructor_validates_container_types() -> None:
    root, cid, proof, params = _receipt_parts()

    receipt = RetrievalReceipt(
        root=root,
        ids=[cid],
        proofs=[proof],
        params=params,
        store_id="store-fixture",
        created_at="2026-06-24T00:00:00Z",
    )

    assert receipt.ids == (cid,)
    assert receipt.proofs == (proof,)
    with pytest.raises(ValidationError, match="receipt ids must be a sequence"):
        RetrievalReceipt(
            root=root,
            ids=None,
            proofs=(proof,),
            params=params,
            store_id="store-fixture",
            created_at="2026-06-24T00:00:00Z",
        )
    with pytest.raises(ValidationError, match="receipt proofs must be a sequence"):
        RetrievalReceipt(
            root=root,
            ids=(cid,),
            proofs=None,
            params=params,
            store_id="store-fixture",
            created_at="2026-06-24T00:00:00Z",
        )


def test_retrieval_receipt_direct_constructor_requires_signature_bytes() -> None:
    root, cid, proof, params = _receipt_parts()

    with pytest.raises(ValidationError, match="signature must be non-empty bytes"):
        RetrievalReceipt(
            root=root,
            ids=(cid,),
            proofs=(proof,),
            params=params,
            store_id="store-fixture",
            created_at="2026-06-24T00:00:00Z",
            signer="ed25519:test-key",
            signature="not-bytes",
        )
    with pytest.raises(ValidationError, match="signature must be non-empty bytes"):
        RetrievalReceipt(
            root=root,
            ids=(cid,),
            proofs=(proof,),
            params=params,
            store_id="store-fixture",
            created_at="2026-06-24T00:00:00Z",
            signer="ed25519:test-key",
            signature=b"",
        )


def test_build_retrieval_receipt_requires_matching_ids_and_proofs(
    tmp_path: Path,
) -> None:
    store = init_store(tmp_path / "store")
    (cid,) = store.put_batch([_item(1.0)])
    root = store.commit()
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
    )

    receipt = build_retrieval_receipt(
        root=root,
        ids=(cid,),
        proofs=tuple(store.prove([cid])),
        query=spec,
        store_id=str(store.manifest.store_id),
        created_at="2026-06-24T00:00:00Z",
    )

    assert receipt.to_json()["created_at"] == "2026-06-24T00:00:00Z"
    with pytest.raises(ValidationError, match="ids and proofs"):
        build_retrieval_receipt(
            root=root,
            ids=(cid,),
            proofs=(),
            query=spec,
            store_id=str(store.manifest.store_id),
        )


def test_uncommitted_store_root_and_prove_fail_closed(tmp_path: Path) -> None:
    store = init_store(tmp_path / "store")
    store.put(_item(1.0))
    spec = QuerySpec(
        vector=np.array([1.0, 0.0], dtype=np.float32),
        k=1,
        metric=Metric.L2,
        with_receipt=True,
    )

    with pytest.raises(UnsupportedOperationError, match="commitments"):
        store.root()
    with pytest.raises(UnsupportedOperationError, match="commitments"):
        store.prove([content_id(_item(2.0))])
    with pytest.raises(UnsupportedOperationError, match="commitments"):
        store.query(spec)
