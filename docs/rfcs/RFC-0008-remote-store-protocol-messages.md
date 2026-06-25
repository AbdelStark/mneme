# RFC-0008: Remote Store Protocol Messages

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.4

## Summary

Mneme defines schema-versioned remote store messages that mirror local `put`, `query`, `prove`, `root`, and `stats` operations. Remote clients must validate schemas, fingerprints, content ids, and receipts before conditioning on returned values.

## Motivation

The PRD requires a local in-process store first and remote/shared stores later. [Architecture](../spec/01-architecture.md#package-boundaries) reserves `mneme.wmcp` for this boundary. This RFC prevents remote behavior from drifting away from local semantics.

## Goals

- Define local-equivalent remote operation messages.
- Use schema versions for every request and response.
- Keep binary arrays explicit through dtype, shape, byte order, and bytes.
- Require client-side validation before conditioning.
- Support conformance tests that run against local and remote stores.

## Non-Goals

- Select a transport protocol in v0.1.
- Define authentication or authorization for all deployments.
- Add confidentiality or private retrieval.
- Replace the local store API.

## Proposed Design

Message families:

```text
mneme.put.request.v1
mneme.put.response.v1
mneme.query.request.v1
mneme.query.response.v1
mneme.prove.request.v1
mneme.prove.response.v1
mneme.root.request.v1
mneme.root.response.v1
mneme.stats.request.v1
mneme.stats.response.v1
mneme.error.v1
```

Array payload:

```json
{
  "dtype": "float32",
  "shape": [512],
  "byte_order": "little",
  "encoding": "base64",
  "data": "..."
}
```

Query request includes the same fields as `QuerySpec`: vector payload, k, metric, ef, filters, temporal decay, receipt flag, and encoder fingerprint. Query response includes item envelopes, distances, and optional receipt.

Example query request:

```json
{
  "schema_version": "mneme.query.request.v1",
  "query": {
    "schema_version": "mneme.query_spec.v1",
    "vector": {
      "dtype": "float32",
      "shape": [2],
      "byte_order": "little",
      "encoding": "base64",
      "data": "AACAPwAAAAA="
    },
    "k": 1,
    "metric": "l2",
    "ef": null,
    "filters": null,
    "temporal_decay": null,
    "with_receipt": true,
    "encoder_fp": null
  }
}
```

Example root response:

```json
{
  "schema_version": "mneme.root.response.v1",
  "root": "ecf590294a28f6b9b77f4c19ffcc3769448e494849021eab7be7faf58d73654a"
}
```

Client validation:

- reject unknown major schema version;
- validate vector and value arrays;
- recompute content ids for returned items;
- reject fingerprint mismatch;
- verify receipt when requested;
- map remote errors to local typed errors.

The reference Python helper is
`mneme.remote.validate_query_response(response, request)`. Transport adapters
must run an equivalent check before passing returned items to conditioners.
`mneme.remote.raise_for_remote_error(error)` converts remote error envelopes to
the corresponding local typed exception.

## HTTP JSON Adapter

`RemoteHttpClient` is the first concrete adapter. It posts the message payloads
to fixed paths:

- `POST /put` with `mneme.put.request.v1`;
- `POST /query` with `mneme.query.request.v1`;
- `POST /prove` with `mneme.prove.request.v1`;
- `POST /root` with `mneme.root.request.v1`;
- `POST /stats` with `mneme.stats.request.v1`.

`MemoryStoreASGIApp` wraps a local `MemoryStore`-compatible object and returns
the matching response schema or a `mneme.error.v1` envelope. The client treats
non-2xx envelopes as local typed errors and still validates successful query
responses before returning retrieved items.

Remote/shared examples must link to the
[shared-store deployment checklist](../spec/06-security.md#shared-store-deployment-checklist),
`validate_query_response`, and receipt verification steps before showing
conditioning against a remote response.

## Alternatives Considered

- Let each transport define its own JSON: fast initially, but fragments semantics.
- Send Python pickles over the wire: unsafe and not interoperable.
- Require receipts for all remote queries: stronger provenance, but v0.4 should allow remote stores without commitments for compatibility.
- Hide binary metadata in opaque blobs: easier, but weakens validation and auditing.

## Drawbacks

- Base64 adds overhead for large values.
- Remote protocol support increases test matrix size.
- Authentication and access control remain deployment concerns until a later security pass.

## Migration / Rollout

v0.4 adds remote message models and a conformance suite. The first transport adapter must pass the same tests as the in-process store for query semantics and error mapping. Existing local stores need no migration.

## Testing Strategy

- JSON schema validation for all messages.
- Round-trip tests for arrays and memory items.
- Local-vs-remote conformance tests for put, query, prove, root, and stats.
- Remote error mapping tests.
- Receipt verification tests over remote query responses.
- `mneme eval remote-conformance --out reports/remote-conformance.json`
  writes a fixture-scale `mneme.eval_report.v1` with transport metadata,
  package version, deterministic local-vs-remote scenario checks, and typed
  remote error-case coverage.

## Resolved Bootstrap Decisions

- First supported transport: v0.4 implements HTTP JSON over an ASGI-compatible service boundary. The message schema remains transport-independent, but the first adapter uses request/response semantics that match `put`, `query`, `prove`, `root`, and `stats`.
- Minimum shared-store authentication guidance: remote/shared examples must require authenticated transport, operator-managed bearer credentials or equivalent deployment authentication, and signed roots for provenance claims. `MemoryStoreASGIApp` can require a bearer token at the application boundary, but TLS, network policy, credential storage, root publication, log retention, backup controls, and key rotation are operator responsibilities. Anonymous writable stores are not a supported deployment pattern.

## References

- [Architecture](../spec/01-architecture.md#package-boundaries)
- [Security](../spec/06-security.md#trust-boundaries)
- [PRD Section 7.4](https://github.com/AbdelStark/mneme/blob/main/prd.md#74-json-message-schema-abridged)
