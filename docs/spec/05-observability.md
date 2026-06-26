# Observability

- Status: Accepted
- Created: 2026-06-24
- Source: [https://github.com/AbdelStark/mneme/blob/main/prd.md](https://github.com/AbdelStark/mneme/blob/main/prd.md#13-evaluation-plan)

## Principles

Observability must support debugging retrieval quality, numerical behavior, store health, and audit reconstruction without leaking raw latent values by default. Logs are structured. Metrics use stable names. Evaluation reports are machine-readable.

## Structured Events

Events are JSON-serializable dictionaries with:

- `event`: stable event name
- `schema_version`
- `store_id`
- `operation`
- `duration_ms`
- `status`
- `error_type` when failed

Library embedders configure event emission with `mneme.observability.ObservabilityConfig`
and an `EventSink` exposing a callable `emit(event)` method. When no sink is
configured, core operations do not construct or dispatch events. Malformed
observability configuration raises
`ValidationError` before any event is emitted. Event sink dispatch is best
effort: sink failures must not fail core operations or mask the original typed
error from an operation.

Required events:

- `mneme.store.put`
- `mneme.store.query`
- `mneme.store.commit`
- `mneme.store.verify`
- `mneme.index.search`
- `mneme.condition.apply`
- `mneme.receipt.verify`
- `mneme.eval.run`

## Metrics

Query metrics:

- `mneme_query_latency_ms`
- `mneme_query_k`
- `mneme_query_hits`
- `mneme_query_distance_min`
- `mneme_query_distance_mean`
- `mneme_query_backend`
- `mneme_query_fingerprint_match`
- `mneme_query_duplicate_results`

Conditioning metrics:

- `mneme_condition_gate_lambda`
- `mneme_condition_empty_retrieval`
- `mneme_condition_mode`
- `mneme_condition_output_finite`

Store metrics:

- `mneme_store_items`
- `mneme_store_value_bytes`
- `mneme_store_index_bytes`
- `mneme_store_manifest_transactions`
- `mneme_store_retention_evictions`

Receipt metrics:

- `mneme_receipt_build_latency_ms`
- `mneme_receipt_verify_latency_ms`
- `mneme_receipt_proof_count`
- `mneme_receipt_root_mismatch`

Evaluation metrics:

- `latent_rollout_error_horizon`
- `loop_closure_consistency`
- `retrieval_recall_at_k`
- `retrieval_mrr`
- `query_latency_p50_ms`
- `query_latency_p99_ms`
- `memory_footprint_bytes_per_item`

## Redaction

Default logs must not include:

- raw latent arrays
- raw action arrays
- environment observations
- full metadata values unless marked safe
- secrets, tokens, local absolute paths, or private dataset names

Event sanitization redacts arrays to shape and dtype summaries. Metadata is
omitted by default except keys explicitly prefixed with `safe_`; those values are
still passed through path, secret, and array redaction. Generic bytes are redacted
unless a field is produced by the content-id prefix helper.

Logs may include:

- content id prefixes when `include_content_id_prefixes=True`
- shape, dtype, device
- schema version
- encoder fingerprint digests
- aggregate distances and gate values
- configured backend names

## Evaluation Reports

Every evaluation command writes a JSON report with:

- `schema_version` equal to `mneme.eval_report.v1`
- command arguments
- package version
- git commit when available
- platform summary
- seed values
- dataset or fixture identifier
- metric values
- pass/fail criteria
- caveats

Reports must be suitable for later README or paper claims. A result without the report is not public evidence.
Fixture reports must include caveats stating that fixture evidence cannot prove
external task success.
The v0.1 fixture command is `mneme eval fixtures --out REPORT.json`;
it writes deterministic synthetic drift and gate metrics suitable for CI and
README claim discipline, not external benchmark claims.

## Resolved Bootstrap Decisions

- Metrics export surface: core continues to emit structured events through `EventSink`. Long-running services add an optional OpenTelemetry adapter behind an observability extra; no telemetry backend becomes a core dependency.
