# RFC-0006: Trained Memory Adapter

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.2

## Summary

Mneme adds a trained cross-attention memory adapter after the v0.1 corrector. The adapter projects retrieved value latents into predictor hidden space and inserts cross-attention blocks while keeping the base encoder and predictor frozen.

## Motivation

The PRD identifies the trained adapter as the accuracy path. [Overview](../spec/00-overview.md#milestone-map) places it in v0.2 because v0.1 must remain training-free. This RFC locks the adapter boundary so future implementation does not mutate base-model weights or conflate adapter success with base retraining.

## Goals

- Define a trained memory adapter that consumes retrieved latents.
- Keep base encoder and predictor parameters frozen.
- Support offline adapter training with reproducible reports.
- Compare adapter performance against the v0.1 corrector and in-context baseline.
- Preserve the `Conditioner` contract.

## Non-Goals

- Specify one external world-model architecture as mandatory.
- Train online memory weights in v0.2.
- Claim adapter superiority before evaluation evidence exists.
- Implement verifiable search or receipts.

## Proposed Design

Public module:

```python
class CrossAttnAdapter(torch.nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None: ...

    def forward(
        self,
        predictor_hidden: torch.Tensor,
        retrieved_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
```

`predictor_hidden` has shape `(batch, predictor_tokens, hidden_dim)`.
`retrieved_values` has shape `(batch, retrieved, latent_dim)` and is projected
to hidden space before cross-attention. `attention_mask` has shape
`(batch, retrieved)` and uses `True`/`1` for valid slots; invalid slots are
converted to a PyTorch key-padding mask. Retrieved values and masks move to the
`predictor_hidden` dtype and device. The output preserves
`predictor_hidden` shape, dtype, and device.

Wrapper conditioner:

```python
@dataclass
class AdapterConditioner:
    adapter: CrossAttnAdapter
    value_projector: torch.nn.Module
    output_head: torch.nn.Module

    def condition(self, parametric: Latent, retrieval: Retrieval, ctx: CondCtx) -> Latent: ...
```

Training:

- freeze base encoder and predictor parameters;
- assert no gradients on base parameters after backward;
- train only adapter, value projector, and output head;
- use train, calibration, and validation splits recorded in the report;
- log base predictor, corrector, in-context, and adapter metrics in one comparison report.

The fixture-scale harness exposed as `train_frozen_base_adapter` requires the
`ml` extra, a callable frozen base model, a callable adapter, a loss function or
`torch.nn.MSELoss`, and non-empty `train`, `calibration`, and `validation`
splits of `AdapterTrainingBatch`. It emits `mneme.eval_report.v1` with split
counts, seed, train/calibration/validation losses, caveats, and an explicit
`base_gradients_absent` metric. It is a contract and smoke-test harness, not an
external benchmark or accuracy claim.

Adapter insertion is implementation-specific because external predictors differ. Mneme provides adapter modules and reference wrappers rather than modifying foreign model code in place.

In-context baseline:

```python
class InContextPredictor(Protocol):
    def predict_with_context(
        self,
        parametric: Latent,
        retrieved_tokens: Sequence[Latent],
        ctx: CondCtx,
    ) -> Latent: ...

@dataclass(frozen=True)
class InContextConditioner:
    predictor: InContextPredictor
    max_tokens: int | None = None

    def condition(self, parametric: Latent, retrieval: Retrieval, ctx: CondCtx) -> Latent: ...
```

The baseline extracts retrieved `Transition.z_next` values, validates that each
token matches the parametric prediction shape, and delegates to a predictor
wrapper that appends those values to its context. Empty retrievals return the
parametric prediction unchanged. This path is for comparison reports and
architecture compatibility checks; it is not the default because the predictor's
self-attention cost grows with `k`.

## Alternatives Considered

- Fine-tune the full predictor: may improve accuracy but breaks the frozen-model adoption path and complicates attribution.
- LoRA-style predictor edits: lighter than full fine-tuning, but still mutates predictor behavior and was less aligned with the PRD's cross-attention design.
- In-context retrieved tokens only: implemented as `InContextConditioner` for
  compatible predictor wrappers. It is useful as a baseline, but attention cost
  scales poorly with k.
- Online memory-weight updates: deferred because the PRD identifies instability risk.

## Drawbacks

- Adapter training adds data, compute, and experiment-management requirements.
- Predictor wrappers may be architecture-specific despite a common adapter module.
- The adapter can overfit retrieval artifacts if evaluation splits are weak.

## Migration / Rollout

v0.2 ships adapter modules behind an ML extra and at least one reference wrapper. The v0.1 `KnnCorrector` remains the default baseline. Adapter checkpoints include config, schema version, base fingerprint, and training report link.
The in-context baseline remains a comparison path for wrappers that can append
retrieved value tokens without modifying predictor weights.

## Testing Strategy

- Unit tests for tensor shapes, masks, dtype, and device movement.
- Freeze tests proving base parameters receive no gradients.
- Serialization test for adapter config and checkpoint metadata.
- Fixture training smoke test on a tiny synthetic dataset.
- Evaluation comparing no-memory, corrector, in-context, and adapter reports.

## Resolved Bootstrap Decisions

- First predictor wrapper: v0.2 implements a generic PyTorch predictor wrapper with explicit hidden-state hooks first. The first model-specific wrapper is the LeJEPA reference wrapper because it is closest to the project thesis and avoids coupling the adapter milestone to a heavier external integration.
- Adapter checkpoint format: adapter weights use a safetensors-compatible file and a JSON metadata sidecar containing schema version, base fingerprint, adapter config, package version, and training report reference.

## References

- [Overview](../spec/00-overview.md#v10-completion-criteria)
- [Testing Strategy](../spec/07-testing-strategy.md#ml-specific-hygiene)
- [PRD Section 8.2](../../prd.md#82-parametric-memory-adapter-trained-accuracy-path)
