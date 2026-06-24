# v1.0 Security Review

- Status: Accepted
- Reviewed: 2026-06-24
- Scope: v1.0 integrity, privacy, remote-response, redaction, documentation, and
  release-artifact gates.

## Result

No release-critical security blockers were found for the v1.0 gate. Mneme v1.0
can claim integrity and provenance controls only: schema validation, canonical
content ids, manifest and value-log validation, receipt verification, remote
response validation, and redacted structured events. It must not claim built-in
confidentiality, private retrieval, encrypted stores, encrypted search, signed
provenance, or exact approximate-search optimality.

## Checklist

| Area | Review requirement | Evidence |
|---|---|---|
| Persisted validation | Unsupported schemas, malformed manifests, invalid value logs, content-id mismatch, and fingerprint mismatch fail closed. | `tests/test_core_types.py`, `tests/test_store_verify_rebuild.py`, `tests/test_store_value_log_transactions.py`, `tests/test_fingerprint_helpers.py` |
| Receipts | Receipt roots, inclusion proofs, query parameters, and canonical returned item bytes are verified before replay or remote conditioning. | `tests/test_receipt_commitment.py`, `tests/test_eval_receipts.py`, `tests/test_eval_replay.py` |
| Remote responses | Remote query responses are untrusted until schema, content id, fingerprint, and requested receipts validate. Malformed server responses and bearer-token failures are typed failures. | `tests/test_remote_validation.py`, `tests/test_remote_http.py`, `tests/test_remote_messages.py` |
| Redaction | Default events redact arrays, observations, local paths, secrets, private dataset ids, unsafe metadata, and raw content ids. | `tests/test_observability_events.py` |
| Public docs | README, SECURITY, release checklist, and specs state that stores are not confidential by default and reject private-retrieval/encrypted-storage claims. | `tests/test_public_docs.py`, `tests/test_security_docs.py`, `tests/test_release_docs.py` |
| Release artifacts | Source and wheel artifacts include security docs, release checklist, tests, and fixture evidence; artifact validation rejects missing required docs or invalid fixture reports. | `tests/test_release_artifacts.py`, `tests/test_public_api_compatibility.py` |

## Confidentiality Decision

Built-in encryption at rest is deferred past v1.0. This is an explicit product
decision, not an accidental omission.

Rationale:

- Key management, recovery, rotation, backup access, and multi-operator sharing
  need a separate design from Mneme's v1 integrity contract.
- Encrypting value logs without a key-management model would create misleading
  confidentiality claims.
- v1.0 already requires documentation, redaction, fail-closed validation,
  receipt integrity, and shared-store deployment guidance.

Required v1.0 wording:

- Mneme stores are not confidential by default.
- Deployments requiring confidentiality must provide access control,
  authenticated transport, filesystem or volume encryption, backup controls,
  secret management, and host isolation outside Mneme.
- Private retrieval and encrypted search remain research-only or post-v1.0
  planning items.

Any built-in encrypted-store support requires a post-v1.0 RFC covering key
management, migration, recovery, threat model, and test fixtures before
implementation.

## Release Gate

Before a v1.0 release candidate, run the full local and hosted gates plus this
security slice:

```bash
pytest tests/test_security_review.py \
  tests/test_security_docs.py \
  tests/test_remote_validation.py \
  tests/test_remote_http.py \
  tests/test_receipt_commitment.py \
  tests/test_eval_replay.py \
  tests/test_observability_events.py \
  tests/test_release_artifacts.py
```

The release notes must link the generated fixture report and must keep the
claim boundary from `docs/release/RELEASE_CHECKLIST.md`.
