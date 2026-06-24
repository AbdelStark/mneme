# Security

- Status: Accepted
- Created: 2026-06-24
- Source: [../../prd.md](../../prd.md#16-security-and-privacy-considerations)

## Security Boundary

Mneme v0.x provides integrity and provenance controls for episodic memory. It does not provide confidentiality by default. Users must treat readable stores as sensitive when they contain real environment data.

## Threat Model

In scope:

- an attacker modifies, deletes, or injects memory items after a root has been logged;
- an operator serves a retrieved item that does not belong to the committed store;
- a store accidentally mixes encoder fingerprints;
- a caller tries to load an unsupported or malformed persisted artifact;
- a public log accidentally records sensitive latent metadata.

Out of scope for v0.x:

- private retrieval;
- encrypted search;
- proving approximate top-k optimality;
- protecting a process from malicious Python code in the same interpreter;
- confidentiality of a store readable by an attacker.

## Trust Boundaries

- Caller-provided observations, latents, actions, metadata, and query filters are untrusted.
- Persisted stores are untrusted until manifest, schema, content ids, and optional commitments validate.
- Optional backend indexes are treated as acceleration structures, not source of truth.
- Remote-store responses are untrusted until schema, content id, fingerprint, and receipt validation pass.
  Remote clients use `validate_query_response` before exposing retrieved items to
  conditioners, and `raise_for_remote_error` maps remote failures to local typed
  exceptions.

## Integrity Controls

- Content ids are computed over canonical memory item bytes.
- Store manifests record file offsets and transaction state.
- Commitment roots bind append order when commitments are enabled. The local
  store persists the MMR sidecar under `receipts/commitment-mmr-v1.json`.
- Receipts prove returned ids are committed members of the store root.
- Encoder fingerprints prevent unsafe key comparison across model versions.

## Privacy Controls

Required in v0.1:

- redaction defaults for logs and reports;
- metadata field allowlist for filters and logs;
- documentation that stores can contain sensitive data;
- `.gitignore` entries for local run outputs and raw data.

Deferred:

- encryption at rest;
- access-controlled remote store;
- private retrieval;
- deletion proofs.

## Security Failure Responses

- Invalid content id: reject on write; fail verification on read.
- Invalid receipt proof: fail closed with `ReceiptVerificationError`.
- Unknown schema major version: reject.
- Metadata redaction failure in tests: fail CI.
- Remote response with mismatched fingerprint: reject before conditioning.
- Malformed remote error envelope: reject with the local schema or validation error.

## Resolved Bootstrap Decisions

- Signing backend: the receipt schema reserves signer and signature fields for
  an Ed25519 backend behind the receipts extra. Local unsigned receipts can
  verify membership today; any remote/shared-store example that claims signed
  provenance must add and test a signing backend first.
- v1.0 confidentiality requirement: built-in encryption at rest is not required for v1.0. v1.0 requires clear deployment guidance, redaction defaults, validation, and receipt integrity. Built-in encrypted stores require a separate post-v1.0 RFC because key management is a distinct design problem.
