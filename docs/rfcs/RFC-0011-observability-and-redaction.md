# RFC-0011: Observability and Redaction

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.1

## Summary

Mneme emits structured operational events, stable metrics, redacted logs, and schema-versioned evaluation reports. Raw latents, observations, actions, secrets, and unsafe metadata are excluded from default logs. Observability is part of the public engineering contract, not an afterthought.

## Motivation

[Observability](../spec/05-observability.md) requires enough signal to debug retrieval quality, numerical behavior, store health, and receipt verification. The same surface can leak sensitive data if it records raw environment traces. This RFC locks the event and redaction boundary.

## Goals

- Define stable structured event names.
- Define default redaction rules.
- Provide metrics hooks for query, store, conditioning, receipt, and evaluation paths.
- Make report outputs suitable for claim audit.
- Keep observability optional for embedders that do not want a logging framework dependency.

## Non-Goals

- Require one external telemetry backend.
- Log raw latent arrays or observations by default.
- Build a hosted metrics service.
- Replace evaluation reports from RFC-0009.

## Proposed Design

Core emits events through a small protocol:

```python
class EventSink(Protocol):
    def emit(self, event: Mapping[str, object]) -> None: ...

@dataclass(frozen=True)
class ObservabilityConfig:
    event_sink: EventSink | None = None
    redact_metadata: bool = True
    include_content_id_prefixes: bool = True
    content_id_prefix_bytes: int = 6
```

Required event names:

```text
mneme.store.put
mneme.store.query
mneme.store.commit
mneme.store.verify
mneme.index.search
mneme.condition.apply
mneme.receipt.verify
mneme.eval.run
```

Default event fields include operation name, schema version, status, duration, store id, backend name, k, hit count, aggregate distances, gate value, receipt status, and typed error name. Events must not include raw `Latent`, `SummaryVec`, `action`, observation, full metadata, local absolute path, or secret values unless the caller opts in through a clearly named unsafe debug mode.

Metrics are exposed first as event fields and later through optional adapters. The core package does not depend on a metrics backend.

## Alternatives Considered

- Use only Python logging strings: easy, but hard to parse and easy to leak raw data.
- Require a metrics backend in core: convenient for services, but too heavy for a library.
- Disable observability until remote stores exist: simpler, but makes early debugging and claim audit weaker.

## Drawbacks

- Event schemas add compatibility surface.
- Redaction may hide details needed for debugging unless unsafe modes are available.
- Content id prefixes can still aid correlation; users with stricter privacy needs must disable them.

## Migration / Rollout

v0.1 adds event emission for core local operations and tests redaction. v0.3 adds receipt events. v0.4 adds remote client/server events. Event fields may add optional keys in minor versions but must not rename required keys without deprecation.

## Testing Strategy

- Unit tests for redaction of arrays, actions, paths, and unsafe metadata.
- Snapshot tests for required event fields.
- Query and condition tests that assert metrics fields are emitted.
- CI test that public fixture logs contain no raw latent arrays.
- Evaluation report test that command, seed, package version, and caveats are present.

## Open Questions

- OPEN QUESTION: First optional metrics adapter for long-running remote stores. Owner: maintainer. Target: v0.4 implementation.

## References

- [Observability](../spec/05-observability.md)
- [Security](../spec/06-security.md#privacy-controls)
- [RFC-0009](RFC-0009-evaluation-and-reproducibility-harness.md)
