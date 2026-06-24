# Security Policy

## Boundary

Mneme v0.x provides integrity and provenance controls for episodic memory. It
does not provide confidentiality by default. Treat readable store directories,
value logs, manifests, fixture reports, and run outputs as sensitive whenever
they contain real environment data. This boundary follows
[RFC-0012](docs/rfcs/RFC-0012-security-boundaries-and-privacy-tiers.md).

Persisted stores are untrusted until Mneme validates the manifest schema, store
paths, value-log checksums, content ids, encoder fingerprints, and typed payload
schemas. Optional indexes are acceleration structures and can be rebuilt from
value logs; they are not authoritative records.

Default structured events are redacted. The regression coverage for that
boundary lives in `tests/test_observability_events.py`; it checks array,
observation, path, secret, private dataset, metadata, and content-id prefix
handling.

## Not Provided By Default

- Encryption at rest
- Private retrieval
- Production remote authentication or secret management
- Sandboxing malicious Python code in the same process
- Proofs that approximate search returned the exact top-k set

Deployments that require confidentiality must add access control, filesystem or
volume encryption, secret management, and transport controls outside Mneme until
an explicit encrypted-store RFC lands.

## Remote And Shared Stores

Do not expose anonymous readable or writable stores. Remote/shared deployments
must provide authenticated transport, network policy, operator-managed
credentials or equivalent deployment authentication, and external
confidentiality controls when stores contain sensitive data. Mneme bearer-token
checks are an application boundary, not a replacement for TLS, credential
storage, rotation, or host access control.

Remote clients must validate responses before conditioning on returned items:
use `validate_query_response` for schema, content-id, fingerprint, and requested
receipt checks. Receipt-based examples must also link to
`verify_retrieval_receipt` and explain that receipts prove committed membership
and canonical item bytes only. Operators own publication, retention, backup, and
audit policy for roots, manifests, value logs, commitment sidecars, and run
logs.

## Reporting

Report suspected vulnerabilities through GitHub security advisories for this
repository. Do not include private datasets, raw store archives, or secrets in a
public issue.
