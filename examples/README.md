# Mneme Examples

These examples are small synthetic rehearsals for the public API, CLI, and docs
workflow. They are not external benchmark evidence and do not claim broad task
success.

## Prerequisites

From the repository root:

```bash
uv sync --locked --group dev
```

The two scripts below use only core dependencies. Serving a remote ASGI app with
`uvicorn` requires `uv sync --locked --extra remote` plus the deployment controls
in [Security](../docs/spec/06-security.md#shared-store-deployment-checklist).

## Local Corrector

```bash
uv run python examples/local_corrector.py
```

Expected success signal: JSON with `"ok": true`, `"example":
`"local-corrector"`, and `corrected_l2 < no_memory_l2` on the synthetic
fixture. This demonstrates the training-free kNN corrector path from
[RFC-0005](../docs/rfcs/RFC-0005-training-free-knn-conditioning.md) and the
public APIs in [SPEC](../SPEC.md).

## Remote Shared Store

```bash
uv run python examples/remote_shared_store.py
```

Expected success signal: JSON with `"ok": true`, `"example":
`"remote-shared-store"`, and `"receipt_verified": true`. The example uses
`RemoteHttpClient`, `MemoryStoreASGIApp`, `validate_query_response`, and
`verify_retrieval_receipt` over an in-process ASGI requester. It demonstrates
the message boundary in
[RFC-0008](../docs/rfcs/RFC-0008-remote-store-protocol-messages.md) and the
receipt boundary in
[RFC-0007](../docs/rfcs/RFC-0007-commitments-and-retrieval-receipts.md).

Remote/shared deployments still require authenticated transport, network policy,
credential management, backup controls, and external confidentiality controls
when memories are sensitive. This example does not provide private retrieval,
encrypted storage, signed provenance, or benchmark evidence.

## Generated Reports

```bash
mkdir -p .artifacts/examples
uv run mneme eval fixtures --out .artifacts/examples/fixtures.json
uv run mneme eval remote-conformance --out .artifacts/examples/remote-conformance.json
uv run mneme eval cross-source --out .artifacts/examples/cross-source.json
```

Expected success signals:

- `.artifacts/examples/fixtures.json` is a valid `mneme.eval_report.v1` fixture
  report with caveats.
- `.artifacts/examples/remote-conformance.json` is a valid `mneme.eval_report.v1`
  report with `"transport": "http-json-asgi"` and matching scenario counts.
- `.artifacts/examples/cross-source.json` is a valid `mneme.eval_report.v1`
  report with source identities, per-source receipt evidence, and caveats.

These report commands are part of the documentation gate in
[RFC-0010](../docs/rfcs/RFC-0010-packaging-ci-and-release-discipline.md).
