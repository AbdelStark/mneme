# Observability

- Status: Accepted
- Created: 2026-06-24
- Source: [../../prd.md](../../prd.md#13-evaluation-plan)

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

Logs may include:

- content id prefixes
- shape, dtype, device
- schema version
- encoder fingerprint digests
- aggregate distances and gate values
- configured backend names

## Evaluation Reports

Every evaluation command writes a JSON report with:

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

## Open Questions

- OPEN QUESTION: Metrics export surface for long-running services. Owner: maintainer. Target: v0.4 remote-store implementation.
