# RFC-0005: Training-Free kNN Conditioning

- Status: Accepted
- Authors: Maintainers
- Created: 2026-06-24
- Target milestone: v0.1

## Summary

Mneme v0.1 ships a training-free kNN conditioner. It computes a distance-weighted nonparametric next-latent estimate from retrieved transitions and blends it with the base predictor through a distance gate. Empty or uninformative retrievals reduce to the parametric prediction.

## Motivation

[Overview](../spec/00-overview.md#v01-scope) requires a zero-training path for early adoption. [Public API](../spec/02-public-api.md#numerical-contracts) requires conditioners to preserve latent space contracts. The PRD maps the kNN language-model interpolation pattern to latent next-state prediction.

## Goals

- Provide a conditioner that works with any conforming frozen predictor.
- Use only retrieved transition values and distances.
- Support delta mode and absolute successor mode.
- Gate memory influence down when neighbors are far.
- Make numerical behavior deterministic and testable.

## Non-Goals

- Learn conditioner parameters in v0.1.
- Guarantee improvement outside store coverage.
- Replace the trained adapter path in RFC-0006.

## Proposed Design

Public class:

```python
@dataclass(frozen=True)
class KnnCorrector:
    tau: float = 0.1
    lambda_max: float = 0.5
    alpha: float = 10.0
    delta0: float = 0.2
    mode: Literal["delta", "absolute"] = "delta"

    def condition(self, parametric: Latent, retrieval: Retrieval, ctx: CondCtx) -> Latent: ...
```

Algorithm:

```text
if retrieval empty:
    return parametric

d_i = finite distances from retrieval
w_i = softmax(-d_i / tau)
if mode == "delta":
    z_knn = ctx.current_latent + sum_i w_i * transition_i.delta
else:
    z_knn = sum_i w_i * transition_i.z_next

d_min = min_i d_i
lambda = lambda_max * sigmoid(alpha * (delta0 - d_min))
z_pred = (1 - lambda) * parametric + lambda * z_knn
```

The conditioner validates that retrieved values are transitions, shapes match, distances are finite, and required `ctx.current_latent` exists for delta mode. It returns the same backend as `parametric` where possible. Torch paths use inference mode and move retrieved values to the parametric device explicitly.

`CondCtx` includes:

```python
@dataclass(frozen=True)
class CondCtx:
    current_latent: Latent | None
    step: int | None = None
    goal_latent: Latent | None = None
    metadata: Mapping[str, Any] | None = None
```

## Alternatives Considered

- Always trust nearest neighbor successor: simple, but unstable when the nearest neighbor is misleading.
- Concatenate retrieved latents as context tokens for v0.1: simple for transformer predictors, but not model-agnostic.
- Train a gate immediately: likely better, but violates the zero-training wedge.
- Use uniform neighbor averaging: less sensitive to distance calibration, but discards useful similarity information.

## Drawbacks

- Gate parameters require calibration for serious use.
- The corrector helps mainly where store coverage is dense.
- Delta arithmetic assumes the latent space supports meaningful residuals.
- Poor summary keys can retrieve misleading transitions.

## Migration / Rollout

v0.1 exposes the corrector as the default conditioner. Later releases may add learned gate calibration, but the untrained defaults remain for reproducible baselines. Any parameter default change requires a changelog entry and evaluation report.

## Testing Strategy

- Empty retrieval returns parametric exactly.
- Far neighbors produce gate near zero.
- Near neighbors produce nonzero blending.
- Delta and absolute modes match hand-computed examples.
- Non-finite distances fail.
- Torch and NumPy paths produce equivalent values on simple fixtures.
- Fixture evaluation writes gate-behavior report.

## Open Questions

- OPEN QUESTION: Default gate parameter calibration for the first non-fixture benchmark. Owner: maintainer. Target: v0.2 evaluation planning.

## References

- [Public API](../spec/02-public-api.md#numerical-contracts)
- [Testing Strategy](../spec/07-testing-strategy.md#required-test-groups)
- [PRD Section 8.1](../../prd.md#81-nonparametric-corrector-default-training-free)
