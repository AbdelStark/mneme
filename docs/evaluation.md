# Evaluation

Mneme evaluation commands write schema-versioned JSON reports. Each report states
its command, seed, package version, dataset reference, metrics, artifacts, and
caveats.

## Fixture Drift And Gate Report

```bash
uv run mneme eval fixtures --out .artifacts/fixtures.json
```

Use this for deterministic CI and release checks. It cannot prove external task
success or broad benchmark improvement.

## Remote Conformance Report

```bash
uv run mneme eval remote-conformance --out .artifacts/remote-conformance.json
```

Use this to compare local-store semantics with the in-process HTTP JSON ASGI
transport fixture. It does not certify network deployment, authentication
operations, load, or confidentiality.

## Cross-Source Transfer Report

```bash
uv run mneme eval cross-source --out .artifacts/cross-source.json
```

Use this to inspect source identities, per-source receipt evidence, no-memory and
single-source baselines, pooled-memory metrics, and caveats on a synthetic
fixture. It does not prove general transfer, private retrieval, consent
compliance, or search optimality.

## External Benchmark Runner

```bash
uv run mneme eval benchmark --dry-run \
  --dataset dataset.json \
  --checkpoint CHECKPOINT \
  --out reports/benchmark.json
```

The built-in dry-run runner validates report plumbing only. Real benchmark
claims require an opt-in runner, pinned inputs, generated artifacts, and public
caveats.
