# Performance Budget

- Status: Accepted
- Created: 2026-06-24
- Source: [../../prd.md](../../prd.md#9-indexing-storage-and-runtime)

## Reference Mode

The primary v0.1 runtime mode is one retrieval per real environment step. Per-imagined-step retrieval is opt-in and must be batched before it is documented as suitable for planning loops.

## Budgets

These are targets to validate, not achieved results.

| Operation | v0.1 target | v1.0 target | Measurement |
|---|---:|---:|---|
| flat query on 10k fixture keys | under 20 ms p99 | under 10 ms p99 | local latency report |
| approximate query on 1M keys | characterize only | within control-loop budget | benchmark report |
| store append, single item | under 10 ms p99 on fixture values | under 5 ms p99 | store benchmark |
| kNN conditioning, k=16 | under 2 ms p99 excluding query | under 1 ms p99 | conditioner benchmark |
| receipt build | not in v0.1 | logarithmic proof size, measured latency | receipt report |
| index rebuild | report items/sec | report items/sec | rebuild report |

No public release may claim a production control rate unless a report identifies hardware, store size, k, backend, and workload.
Use generated evaluation reports for performance claims; do not cite ad hoc
terminal timings without the report envelope.

The v0.1 local profile command is:

```bash
python -m mneme.cli eval profile --store STORE --out reports/profile.json
```

It writes `mneme.eval_report.v1` with FlatIndex recall ground truth, query and
conditioning latency percentiles, footprint fields, hardware fields, and caveats
for exact-only runs when an approximate backend is unavailable.

## Memory Budget

- Summary keys are `float32`, one-dimensional arrays.
- The default key dimension target is 256 to 1024.
- Values may live in memory-mapped files and must not be loaded wholesale for query unless the store is explicitly configured as in-memory.
- Retention policies are required before any claim about multi-million-item stores.

## Profiling Requirements

Each benchmark report records:

- package version and git commit;
- CPU, GPU, memory, OS, Python version;
- index backend and parameters;
- item count, key dimension, value kind, k, metric, ef;
- warmup count and measurement count;
- p50, p95, p99, mean, and max latency;
- caveats for synthetic fixtures or external datasets.

## Concurrency

v0.1 may use a single-writer model. Concurrent readers are allowed only after tests prove manifest and index consistency. Remote-store concurrency is deferred to v0.4.

## Risks

- RISK: Python orchestration overhead may dominate per-imagined-step retrieval. Resolution: default to per-real-step retrieval and require batched-query evidence before recommending the expensive mode.
- RISK: value loading may dominate query latency for large latents. Resolution: keep values separate from keys and measure value load time separately.
