"""Merkle Mountain Range commitment state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike, fspath
from pathlib import Path
from typing import Final, Literal

from blake3 import blake3

from mneme.core import (
    Cid,
    ReceiptVerificationError,
    SchemaVersionError,
    ValidationError,
)
from mneme.core._json import loads_strict_json, write_strict_json_file

COMMITMENT_SCHEMA: Final = "mneme.commitment.v1"
INCLUSION_PROOF_SCHEMA: Final = "mneme.inclusion_proof.v1"
MMR_SCHEME: Final = "mmr-v1"
_DIGEST_SIZE: Final = 32
_EMPTY_PREFIX: Final = b"mneme.mmr.v1.empty"
_LEAF_PREFIX: Final = b"mneme.mmr.v1.leaf"
_NODE_PREFIX: Final = b"mneme.mmr.v1.node"

ProofDirection = Literal["left", "right"]


@dataclass(frozen=True)
class ProofStep:
    """One sibling hash in an inclusion proof."""

    direction: ProofDirection
    digest: bytes

    def __post_init__(self) -> None:
        if self.direction not in {"left", "right"}:
            raise ValidationError("proof step direction must be 'left' or 'right'")
        _require_digest(self.digest, "proof step digest")

    def to_json(self) -> dict[str, str]:
        """Return a JSON-ready proof step."""

        return {"direction": self.direction, "digest": self.digest.hex()}

    @classmethod
    def from_json(cls, data: object) -> ProofStep:
        mapping = _require_mapping(data, "proof step")
        direction = mapping.get("direction")
        if direction != "left" and direction != "right":
            raise ValidationError("proof step direction must be 'left' or 'right'")
        return cls(
            direction=direction,
            digest=_bytes_from_hex(mapping.get("digest"), "proof step digest"),
        )


@dataclass(frozen=True)
class InclusionProof:
    """Inclusion proof from a committed content id to an MMR root."""

    leaf_index: int
    item_count: int
    steps: tuple[ProofStep, ...]
    schema_version: str = INCLUSION_PROOF_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(
            self.schema_version,
            INCLUSION_PROOF_SCHEMA,
            "inclusion proof",
        )
        _require_non_negative_int(self.leaf_index, "leaf_index")
        _require_positive_int(self.item_count, "item_count")
        if self.leaf_index >= self.item_count:
            raise ValidationError("leaf_index must be less than item_count")
        object.__setattr__(
            self,
            "steps",
            tuple(_require_proof_step(step) for step in self.steps),
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready proof object."""

        return {
            "schema_version": self.schema_version,
            "leaf_index": self.leaf_index,
            "item_count": self.item_count,
            "steps": [step.to_json() for step in self.steps],
        }

    @classmethod
    def from_json(cls, data: object) -> InclusionProof:
        mapping = _require_mapping(data, "inclusion proof")
        steps = mapping.get("steps")
        if not isinstance(steps, list):
            raise ValidationError("inclusion proof steps must be a list")
        return cls(
            schema_version=_require_string(
                mapping.get("schema_version"),
                "inclusion proof schema_version",
            ),
            leaf_index=_require_non_negative_int(
                mapping.get("leaf_index"),
                "leaf_index",
            ),
            item_count=_require_positive_int(mapping.get("item_count"), "item_count"),
            steps=tuple(ProofStep.from_json(step) for step in steps),
        )


@dataclass(frozen=True)
class CommitmentState:
    """Append-order MMR commitment over memory-item content ids."""

    scheme: Literal["mmr-v1"] = MMR_SCHEME
    root: bytes = field(default_factory=lambda: _empty_root())
    item_count: int = 0
    peaks: tuple[bytes, ...] = ()
    leaf_ids: tuple[Cid, ...] = ()
    schema_version: str = COMMITMENT_SCHEMA

    def __post_init__(self) -> None:
        _validate_schema(self.schema_version, COMMITMENT_SCHEMA, "commitment")
        if self.scheme != MMR_SCHEME:
            raise ValidationError("commitment scheme must be 'mmr-v1'")
        _require_digest(self.root, "commitment root")
        _require_non_negative_int(self.item_count, "item_count")
        object.__setattr__(
            self,
            "leaf_ids",
            tuple(_require_cid(cid, "leaf id") for cid in self.leaf_ids),
        )
        if len(self.leaf_ids) != self.item_count:
            raise ValidationError("leaf_ids length must match item_count")
        object.__setattr__(
            self,
            "peaks",
            tuple(_require_digest(peak, "commitment peak") for peak in self.peaks),
        )
        expected = _state_from_cids(self.leaf_ids)
        if self.peaks != expected.peaks:
            raise ValidationError("commitment peaks do not match leaf ids")
        if self.root != expected.root:
            raise ValidationError("commitment root does not match leaf ids")

    @classmethod
    def empty(cls) -> CommitmentState:
        """Return the empty MMR commitment state."""

        return cls()

    @classmethod
    def from_cids(cls, cids: Sequence[Cid]) -> CommitmentState:
        """Build a commitment state from content ids in append order."""

        return _state_from_cids(tuple(_require_cid(cid, "content id") for cid in cids))

    def append(self, cid: Cid) -> CommitmentState:
        """Return a new state with one appended content id."""

        return type(self).from_cids((*self.leaf_ids, cid))

    def extend(self, cids: Sequence[Cid]) -> CommitmentState:
        """Return a new state with multiple appended content ids."""

        return type(self).from_cids((*self.leaf_ids, *cids))

    def prove(self, cid: Cid) -> InclusionProof:
        """Return an inclusion proof for a committed content id."""

        target = _require_cid(cid, "content id")
        try:
            index = self.leaf_ids.index(target)
        except ValueError as exc:
            raise ReceiptVerificationError("content id is not committed") from exc
        peaks = _build_forest(self.leaf_ids)
        peak_index, peak = _peak_for_index(peaks, index)
        steps = [*_leaf_steps(peak, index), *_bagging_steps(peaks, peak_index)]
        return InclusionProof(
            leaf_index=index,
            item_count=self.item_count,
            steps=tuple(steps),
        )

    @property
    def root_hex(self) -> str:
        """Return the commitment root as lowercase hex."""

        return self.root.hex()

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready commitment state."""

        return {
            "schema_version": self.schema_version,
            "scheme": self.scheme,
            "root": self.root.hex(),
            "item_count": self.item_count,
            "peaks": [peak.hex() for peak in self.peaks],
            "leaf_ids": [cid.hex() for cid in self.leaf_ids],
        }

    @classmethod
    def from_json(cls, data: object) -> CommitmentState:
        mapping = _require_mapping(data, "commitment")
        peaks = mapping.get("peaks")
        if not isinstance(peaks, list):
            raise ValidationError("commitment peaks must be a list")
        leaf_ids = mapping.get("leaf_ids")
        if not isinstance(leaf_ids, list):
            raise ValidationError("commitment leaf_ids must be a list")
        return cls(
            schema_version=_require_string(
                mapping.get("schema_version"),
                "commitment schema_version",
            ),
            scheme=_require_scheme(mapping.get("scheme")),
            root=_bytes_from_hex(mapping.get("root"), "commitment root"),
            item_count=_require_non_negative_int(
                mapping.get("item_count"),
                "item_count",
            ),
            peaks=tuple(_bytes_from_hex(peak, "commitment peak") for peak in peaks),
            leaf_ids=tuple(_bytes_from_hex(cid, "leaf id") for cid in leaf_ids),
        )


@dataclass(frozen=True)
class _Node:
    start: int
    end: int
    height: int
    digest: bytes
    left: _Node | None = None
    right: _Node | None = None


def verify_inclusion_proof(
    cid: Cid,
    proof: InclusionProof,
    root: bytes,
) -> bool:
    """Return whether a proof connects a content id to a commitment root."""

    try:
        target = _require_cid(cid, "content id")
        expected_root = _require_digest(root, "commitment root")
        checked = _require_proof(proof)
    except (TypeError, ValidationError):
        return False
    digest = _hash_leaf(target)
    for step in checked.steps:
        if step.direction == "left":
            digest = _hash_parent(step.digest, digest)
        else:
            digest = _hash_parent(digest, step.digest)
    return digest == expected_root


def save_commitment_state(path: str | Path, state: CommitmentState) -> Path:
    """Write a commitment state sidecar and return its path."""

    if not isinstance(state, CommitmentState):
        raise ValidationError("state must be a CommitmentState")
    target = _require_path(path, "path")
    try:
        return write_strict_json_file(target, state.to_json(), sort_keys=True, indent=2)
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise ReceiptVerificationError(
            f"commitment state could not be serialized: {target}"
        ) from exc
    except OSError as exc:
        raise ReceiptVerificationError(
            f"commitment state could not be written: {target}"
        ) from exc


def load_commitment_state(path: str | Path) -> CommitmentState:
    """Load and validate a commitment state sidecar."""

    target = _require_path(path, "path")
    try:
        data = loads_strict_json(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReceiptVerificationError(f"commitment state not found: {target}") from exc
    except OSError as exc:
        raise ReceiptVerificationError(
            f"commitment state could not be read: {target}"
        ) from exc
    except ValueError as exc:
        raise ReceiptVerificationError(
            f"commitment state is not valid JSON: {target}"
        ) from exc
    try:
        return CommitmentState.from_json(data)
    except (SchemaVersionError, ValidationError) as exc:
        raise ReceiptVerificationError(str(exc)) from exc


def _state_from_cids(cids: tuple[Cid, ...]) -> CommitmentState:
    peaks = tuple(node.digest for node in _build_forest(cids))
    root = _root_from_peaks(peaks)
    state = object.__new__(CommitmentState)
    object.__setattr__(state, "scheme", MMR_SCHEME)
    object.__setattr__(state, "root", root)
    object.__setattr__(state, "item_count", len(cids))
    object.__setattr__(state, "peaks", peaks)
    object.__setattr__(state, "leaf_ids", cids)
    object.__setattr__(state, "schema_version", COMMITMENT_SCHEMA)
    return state


def _build_forest(cids: Sequence[Cid]) -> tuple[_Node, ...]:
    peaks: list[_Node] = []
    for index, cid in enumerate(cids):
        peaks.append(
            _Node(
                start=index,
                end=index + 1,
                height=0,
                digest=_hash_leaf(cid),
            )
        )
        while len(peaks) >= 2 and peaks[-2].height == peaks[-1].height:
            right = peaks.pop()
            left = peaks.pop()
            peaks.append(
                _Node(
                    start=left.start,
                    end=right.end,
                    height=left.height + 1,
                    digest=_hash_parent(left.digest, right.digest),
                    left=left,
                    right=right,
                )
            )
    return tuple(peaks)


def _peak_for_index(peaks: Sequence[_Node], index: int) -> tuple[int, _Node]:
    for peak_index, peak in enumerate(peaks):
        if peak.start <= index < peak.end:
            return peak_index, peak
    raise ReceiptVerificationError("proof index is outside committed range")


def _leaf_steps(node: _Node, index: int) -> list[ProofStep]:
    if node.left is None or node.right is None:
        return []
    if index < node.left.end:
        return [
            *_leaf_steps(node.left, index),
            ProofStep(direction="right", digest=node.right.digest),
        ]
    return [
        *_leaf_steps(node.right, index),
        ProofStep(direction="left", digest=node.left.digest),
    ]


def _bagging_steps(peaks: Sequence[_Node], peak_index: int) -> list[ProofStep]:
    steps: list[ProofStep] = []
    if peak_index > 0:
        left_bag = peaks[0].digest
        for peak in peaks[1:peak_index]:
            left_bag = _hash_parent(left_bag, peak.digest)
        steps.append(ProofStep(direction="left", digest=left_bag))
    for peak in peaks[peak_index + 1 :]:
        steps.append(ProofStep(direction="right", digest=peak.digest))
    return steps


def _root_from_peaks(peaks: Sequence[bytes]) -> bytes:
    if not peaks:
        return _empty_root()
    root = peaks[0]
    for peak in peaks[1:]:
        root = _hash_parent(root, peak)
    return root


def _hash_leaf(cid: Cid) -> bytes:
    return blake3(_LEAF_PREFIX + _require_cid(cid, "content id")).digest(
        length=_DIGEST_SIZE
    )


def _hash_parent(left: bytes, right: bytes) -> bytes:
    return blake3(
        _NODE_PREFIX
        + _require_digest(left, "left digest")
        + _require_digest(right, "right digest")
    ).digest(length=_DIGEST_SIZE)


def _empty_root() -> bytes:
    return blake3(_EMPTY_PREFIX).digest(length=_DIGEST_SIZE)


def _validate_schema(schema_version: str, expected: str, name: str) -> None:
    if not isinstance(schema_version, str):
        raise SchemaVersionError(f"{name} schema_version must be a string")
    if schema_version != expected:
        raise SchemaVersionError(f"unsupported {name} schema: {schema_version!r}")


def _require_scheme(value: object) -> Literal["mmr-v1"]:
    if value != MMR_SCHEME:
        raise ValidationError("commitment scheme must be 'mmr-v1'")
    return MMR_SCHEME


def _require_cid(value: object, field_name: str) -> Cid:
    if not isinstance(value, bytes) or len(value) != _DIGEST_SIZE:
        raise ValidationError(f"{field_name} must be {_DIGEST_SIZE} bytes")
    return value


def _require_digest(value: object, field_name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != _DIGEST_SIZE:
        raise ValidationError(f"{field_name} must be {_DIGEST_SIZE} bytes")
    return value


def _require_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{field_name} must be a positive integer")
    return value


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_path(value: object, field_name: str) -> Path:
    if not isinstance(value, str | PathLike):
        raise ValidationError(f"{field_name} must be a path-like value")
    raw = fspath(value)
    if not isinstance(raw, str):
        raise ValidationError(f"{field_name} must resolve to a text path")
    if not raw:
        raise ValidationError(f"{field_name} must not be empty")
    return Path(raw)


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    return value


def _require_proof_step(value: object) -> ProofStep:
    if not isinstance(value, ProofStep):
        raise ValidationError("proof steps must be ProofStep instances")
    return value


def _require_proof(value: object) -> InclusionProof:
    if not isinstance(value, InclusionProof):
        raise ValidationError("proof must be an InclusionProof")
    return value


def _bytes_from_hex(value: object, field_name: str) -> bytes:
    text = _require_string(value, field_name)
    try:
        return bytes.fromhex(text)
    except ValueError as exc:
        raise ValidationError(f"{field_name} must be hex bytes") from exc


__all__ = [
    "COMMITMENT_SCHEMA",
    "INCLUSION_PROOF_SCHEMA",
    "MMR_SCHEME",
    "CommitmentState",
    "InclusionProof",
    "ProofStep",
    "load_commitment_state",
    "save_commitment_state",
    "verify_inclusion_proof",
]
