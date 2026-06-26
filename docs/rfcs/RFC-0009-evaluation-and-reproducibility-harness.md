# RFC-0009: Evaluation and Reproducibility Harness

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.1

## Summary

Mneme evaluations produce schema-versioned reports that distinguish fixture-scale validation from external benchmark evidence. v0.1 must report drift fixtures, gate behavior, exact-vs-approximate recall, and latency. Later milestones add adapter comparisons, receipt overhead, remote-store conformance, and cross-source transfer fixtures.

## Motivation

[Overview](../spec/00-overview.md#claim-boundary) says performance and drift improvements are not achieved results yet. [Testing Strategy](../spec/07-testing-strategy.md#required-test-groups) requires reports with commands, seeds, metrics, and caveats. This RFC defines what evidence must exist before public claims change.

## Goals

- Define a report schema for evaluations.
- Require reproducible commands and seed recording.
- Separate synthetic fixtures from external benchmarks.
- Cover drift, gate, recall, latency, adapter comparison, receipt overhead, and replay.
- Prevent unsupported public claims.

## Non-Goals

- Define a new benchmark suite.
- Require large external datasets in unit CI.
- Claim v0.1 scientific success from synthetic fixtures.

## Proposed Design

Report envelope:

```python
@dataclass(frozen=True)
class EvalReport:
    schema_version: str
    report_id: str
    command: list[str]
    package_version: str
    git_commit: str | None
    created_at: str
    platform: Mapping[str, str]
    seed: int | None
    dataset: DatasetRef
    metrics: Mapping[str, float | int | str]
    artifacts: Mapping[str, str]
    caveats: tuple[str, ...]
    passed: bool
```

`created_at` values are ISO 8601 UTC timestamps so report artifacts can be
ordered and compared without relying on local timezone context.

Required v0.1 commands:

```text
mneme eval fixtures --out reports/fixtures.json
mneme eval profile --store STORE --out reports/profile.json
mneme eval recall --store STORE --out reports/recall.json
mneme eval latency --store STORE --out reports/latency.json
mneme eval receipts --store STORE --out reports/receipts.json
mneme eval replay --trace TRACE.json --out reports/replay.json
mneme eval remote-conformance --out reports/remote-conformance.json
mneme eval cross-source --out reports/cross-source.json
mneme eval gate --out reports/gate.json
```

Required v0.1 evidence:

- no-memory and corrector latent rollout error on deterministic fixtures;
- gate lambda curve for near and far neighbors;
- exact flat recall baseline;
- approximate recall when approximate backend is installed;
- query and conditioning latency with p50 and p99;
- item count, key dimension, k, metric, backend, hardware, and footprint fields;
- transport, scenario count, package version, and typed error-case coverage for
  remote conformance reports;
- source identities, target fixture, no-memory baseline, single-source
  baseline, pooled-memory metric, per-source receipt verification, provenance,
  and caveats for cross-source transfer reports;
- caveat field stating fixture evidence cannot prove external task success.

External benchmark reports are opt-in artifacts and must identify dataset, split, model checkpoint, hardware, and command.

External runner interface:

```python
class BenchmarkRunner(Protocol):
    def run(self, spec: BenchmarkSpec) -> BenchmarkResult: ...

@dataclass(frozen=True)
class BenchmarkSpec:
    dataset: DatasetRef
    checkpoint_uri: str
    modes: Sequence[Literal["no_memory", "corrector", "in_context", "adapter"]]
    command: Sequence[str]
    seed: int | None = None
    hardware: Mapping[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class BenchmarkResult:
    metrics: Mapping[str, float | int | str]
    artifacts: Mapping[str, str]
    caveats: Sequence[str]
    passed: bool
```

The built-in `DryRunBenchmarkRunner` exercises report generation for the
no-memory, corrector, in-context, and adapter comparison modes without touching
external datasets or model checkpoints. It is a fixture adapter for interface
validation only. Benchmark mode lists must not contain duplicates, because
report comparison counts and mode-specific slots are interpreted as unique
configuration entries.

## Alternatives Considered

- Use ad hoc notebooks as evaluation output: useful during exploration, but not enough for public claim evidence.
- Put benchmark runs in default CI: too expensive and brittle for early implementation.
- Report only aggregate success/failure: insufficient for claim audit.

## Drawbacks

- Report schema adds maintenance work.
- Fixture metrics may be mistaken for benchmark claims unless caveats are enforced.
- External benchmark reproducibility depends on checkpoint and dataset availability.

## Migration / Rollout

v0.1 implements fixture reports and local latency/recall reports. v0.2 adds adapter comparison reports. v0.3 adds receipt overhead and replay reports. Receipt overhead reports compare receipt-disabled and receipt-enabled query latency, receipt build latency, verification latency, and proof size trends. Receipt replay reports recompute a logged conditioning set after verifying item bytes and inclusion proofs; they do not replay the full environment or prove nearest-neighbor optimality. v0.4 adds a fixture-scale remote conformance report for the first HTTP JSON transport; it compares local and remote store semantics but does not certify network deployment, authentication operations, load, or confidentiality. v0.5 adds a fixture-scale cross-source transfer report under RFC-0013; it records source identities, per-source receipts, baselines, pooled-memory metrics, and caveats, but does not claim general transfer, confidentiality, private retrieval, consent compliance, or federation support. Release docs may cite only reports checked into release artifacts or linked from release notes.

## Testing Strategy

- Schema validation for report JSON.
- Deterministic fixture report under fixed seed.
- Failure test when required caveats are missing.
- CLI tests for report output paths.
- Remote conformance tests for local-vs-remote put, query, prove, root, stats,
  and typed error mapping.
- Cross-source transfer tests for at least two sources, target metric,
  no-pooling baseline, receipt-backed provenance, and caveats.
- Golden report fixture with known metrics.

## Resolved Bootstrap Decisions

- LOOPNAV is the first external benchmark accepted as release evidence. It is narrow enough to test the project's core loop-closure and revisit-consistency claim before broader navigation or manipulation benchmarks are used for public claims.

## References

- [Testing Strategy](../spec/07-testing-strategy.md)
- [Performance Budget](../spec/08-performance-budget.md)
- [RFC-0013](RFC-0013-cross-source-memory-provenance.md)
- [PRD Section 13](https://github.com/AbdelStark/mneme/blob/main/prd.md#13-evaluation-plan)
