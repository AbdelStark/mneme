from __future__ import annotations

from pathlib import Path


def test_security_docs_state_confidentiality_boundary_and_redaction_link() -> None:
    security = Path("SECURITY.md").read_text(encoding="utf-8").lower()

    assert "does not provide confidentiality by default" in security
    assert "persisted stores are untrusted" in security
    assert "tests/test_observability_events.py" in security
    assert "rfc-0012-security-boundaries-and-privacy-tiers.md" in security
    assert "redacted" in security
    assert "do not expose anonymous readable or writable stores" in security
    assert "validate_query_response" in security
    assert "verify_retrieval_receipt" in security
    assert "operator-managed" in security


def test_release_checklist_includes_security_boundary_review() -> None:
    release = Path("docs/spec/09-release-and-versioning.md").read_text(encoding="utf-8")

    assert "security boundary review" in release
    assert "do not claim store" in release
    assert "redaction regression tests" in release


def test_shared_store_checklist_covers_deployment_controls() -> None:
    spec = Path("docs/spec/06-security.md").read_text(encoding="utf-8").lower()
    rfc = (
        Path("docs/rfcs/RFC-0012-security-boundaries-and-privacy-tiers.md")
        .read_text(encoding="utf-8")
        .lower()
    )
    remote = Path("docs/rfcs/RFC-0008-remote-store-protocol-messages.md").read_text(
        encoding="utf-8"
    )

    assert "shared-store deployment checklist" in spec
    assert "does not provide encrypted stores" in spec
    assert "private retrieval" in spec
    assert "validate_query_response" in spec
    assert "verify_retrieval_receipt" in spec
    assert "anonymous writable shared store" in spec
    assert "root publication" in rfc
    assert "log retention" in rfc
    assert "shared-store deployment checklist" in remote
    assert "validate_query_response" in remote


def test_signed_receipt_docs_preserve_reserved_fail_closed_boundary() -> None:
    security = Path("docs/spec/06-security.md").read_text(encoding="utf-8").lower()
    rfc = " ".join(
        Path("docs/rfcs/RFC-0007-commitments-and-retrieval-receipts.md")
        .read_text(encoding="utf-8")
        .lower()
        .split()
    )
    tiers = " ".join(
        Path("docs/rfcs/RFC-0012-security-boundaries-and-privacy-tiers.md")
        .read_text(encoding="utf-8")
        .lower()
        .split()
    )

    assert "signer and signature fields" in security
    assert "signing backend first" in security
    assert "signed receipts until a signing backend" in rfc
    assert "verification fails closed" in rfc
    assert (
        "must be added with tests before documentation can claim signed provenance"
        in rfc
    )
    assert "signed-root publication is reserved" in tiers
