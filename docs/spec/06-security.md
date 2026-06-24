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

## Integrity Controls

- Content ids are computed over canonical memory item bytes.
- Store manifests record file offsets and transaction state.
- Commitment roots bind append order when commitments are enabled.
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

## Open Questions

- OPEN QUESTION: Which signing backend is selected for v0.3 roots. Owner: maintainer. Target: v0.3 design freeze.
- OPEN QUESTION: Whether v1.0 requires encryption-at-rest support or only deployment guidance. Owner: maintainer. Target: v1.0 planning.
