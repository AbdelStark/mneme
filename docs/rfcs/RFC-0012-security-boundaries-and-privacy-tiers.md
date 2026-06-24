# RFC-0012: Security Boundaries and Privacy Tiers

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.1

## Summary

Mneme treats caller inputs, persisted stores, and remote responses as untrusted. v0.x provides validation, integrity, provenance, and redaction controls. It does not provide confidentiality by default. Confidentiality features are explicit later tiers rather than implicit promises.

## Motivation

[Security](../spec/06-security.md) defines the project's integrity boundary and states that episodic memories can contain sensitive environment data. The PRD makes verifiability central but also warns that v0 does not encrypt stores. This RFC locks the security posture so implementation and documentation cannot overclaim.

## Goals

- Define trust boundaries for inputs, stores, indexes, and remote responses.
- Require fail-closed validation for schemas, fingerprints, content ids, and receipts.
- Require redaction defaults for logs and reports.
- Separate integrity tiers from confidentiality tiers.
- Produce public security documentation before release.

## Non-Goals

- Provide encryption at rest in v0.1.
- Provide private retrieval in v0.x.
- Prove approximate search correctness.
- Sandbox malicious Python code in the same process.

## Proposed Design

Trust boundaries:

- observations, latents, actions, metadata, and query filters are untrusted;
- persisted stores are untrusted until manifest, schema, checksums, and content ids validate;
- indexes are rebuildable acceleration structures, not authoritative records;
- remote responses are untrusted until schema, content id, fingerprint, and optional receipt checks pass.

Integrity tiers:

- T0: schema validation, content ids, manifest checks, index rebuild; v0.1.
- T1: committed content ids and inclusion proofs; v0.3.
- T2: signed roots and append-history binding; v0.3 to v0.4.
- T3: verifiable search correctness; research only, not promised for v1.0.

Privacy tiers:

- P0: redaction defaults and user documentation; v0.1.
- P1: deployment guidance for access control and encryption at rest; v0.4.
- P2: first-class encrypted store support; open for v1.x planning.
- P3: private retrieval; research only.

Security documentation must state that v0.x stores are not confidential unless deployment controls provide confidentiality outside Mneme.

## Alternatives Considered

- Promise encryption in the initial release: stronger posture, but risks delaying the core memory contract and would require a separate key-management design.
- Treat remote stores as trusted when using a secure transport: insufficient because payload schema and content integrity still need validation.
- Defer security docs until receipts exist: unsafe because v0.1 already persists potentially sensitive memory values.

## Drawbacks

- Explicit privacy tiers make early releases look less complete, but they prevent misleading claims.
- Fail-closed validation can reject stores that a permissive tool might recover.
- Security docs and tests add release work before model-quality results exist.

## Migration / Rollout

v0.1 ships validation, redaction, `.gitignore` protections for raw data and run outputs, and `SECURITY.md`. v0.3 adds receipt verification. v0.4 adds remote-store deployment guidance. Any confidentiality feature requires a separate RFC.

## Testing Strategy

- Reject invalid schemas, content ids, fingerprints, and malformed filters.
- Verify redaction of logs and reports.
- Ensure `.gitignore` excludes raw data, run outputs, and local artifacts.
- Security doc test or checklist in release CI.
- Receipt validation failure tests when RFC-0007 lands.

## Resolved Bootstrap Decisions

- v1.0 requires documented deployment controls, redaction defaults, validation, and receipt integrity. Built-in encrypted stores are deferred to a post-v1.0 RFC because encryption requires a key-management and recovery design that is separate from Mneme's v1 integrity contract.

## References

- [Security](../spec/06-security.md)
- [Observability](../spec/05-observability.md#redaction)
- [RFC-0007](RFC-0007-commitments-and-retrieval-receipts.md)
