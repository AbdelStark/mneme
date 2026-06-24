from __future__ import annotations

from pathlib import Path

RFC = Path("docs/rfcs/RFC-0013-cross-source-memory-provenance.md")


def test_cross_source_rfc_defines_source_identity_metadata() -> None:
    text = RFC.read_text(encoding="utf-8")

    assert "SourceIdentity" in text
    assert "mneme.source.v1" in text
    assert "source_id" in text
    assert "source_kind" in text
    assert "store_id" in text
    assert "root_scheme" in text
    assert "encoder_fingerprint" in text
    assert "policy_tags" in text
    assert "disclosure_level" in text
    assert 'meta["mneme_source"]' in text


def test_cross_source_rfc_defines_provenance_receipt_requirements() -> None:
    text = " ".join(RFC.read_text(encoding="utf-8").split())

    assert "CrossSourceProvenanceReceipt" in text
    assert "retrieval_receipts_by_source" in text
    assert "validate each remote response schema" in text
    assert "recompute returned item content ids" in text
    assert "reject encoder fingerprint mismatches" in text
    assert "verify every requested per-source `RetrievalReceipt`" in text
    assert "signed provenance" in text
    assert "aggregation policy" in text


def test_cross_source_rfc_preserves_security_and_claim_boundaries() -> None:
    text = RFC.read_text(encoding="utf-8").lower()

    assert "does not provide confidentiality" in text
    assert "private retrieval is research-only" in text
    assert "does not prove private retrieval" in text
    assert "does not prove" in text
    assert "search optimality" in text
    assert "consent automation" in text
    assert "external benchmark report" in text


def test_cross_source_rfc_identifies_transfer_metrics_and_issue_boundary() -> None:
    text = RFC.read_text(encoding="utf-8")

    assert "source count and returned item count per source" in text
    assert "cross-source improvement rate" in text
    assert "negative-transfer rate" in text
    assert "source-diversity score" in text
    assert "receipt verification failure count by source" in text
    assert "proof bytes by source" in text
    assert "redaction failure count" in text
    assert "#48" in text
    assert "#50" in text
    assert "create separate issues" in text


def test_overview_and_release_docs_link_cross_source_rfc() -> None:
    overview = Path("docs/spec/00-overview.md").read_text(encoding="utf-8")
    release = " ".join(
        Path("docs/spec/09-release-and-versioning.md")
        .read_text(encoding="utf-8")
        .split()
    )
    security = Path(
        "docs/rfcs/RFC-0012-security-boundaries-and-privacy-tiers.md"
    ).read_text(encoding="utf-8")

    assert "RFC-0013" in overview
    assert "cross-source memory sharing" in overview
    assert "RFC-0013" in release
    assert "private retrieval" in release
    assert "encrypted storage" in release
    assert "RFC-0013-cross-source-memory-provenance.md" in security
