# Mneme: Episodic Memory and Retrieval for Latent World Models

**Document type:** Product requirements and technical specification
**Status:** Draft v0.1 (for review)
**Author:** Abdelhamid Bakhta
**Date:** 2026-06-24
**Interface designation:** WM-RFC-0002, Episodic Memory and Retrieval Interface (extends WMCP / [WM-RFC-0001](https://github.com/AbdelStark/wm-rfcs/blob/main/rfcs/WM-RFC-0001-wmcp.md))
**Reference implementation:** `mneme` (Python package, PyTorch + FAISS)

> Names (`Mneme`, package `mneme`, `WM-RFC-0002`) are working labels and can change. The design does not depend on them.

---

## 1. Summary

Mneme is a model-agnostic episodic memory and retrieval layer for latent (JEPA-style) world models. It stores realized trajectories as latent transitions in a content-addressed store, retrieves the most relevant past transitions at rollout time, and conditions a frozen world model's predictor on them so that imagined futures stay anchored to states the system has actually observed.

The target failure mode is long-horizon drift. Latent world models predict the next representation cheaply and plan well over short horizons, then degrade as compounding prediction error and distribution shift accumulate over a rollout. Retrieval is the standard fix for the analogous problem in language models, where a parametric model is grounded against a nonparametric datastore. Mneme ports that pattern to latent dynamics.

Two properties separate Mneme from existing research prototypes. First, it is a reusable primitive with a stable interface rather than an architecture baked into one model, so any latent world model can adopt it through the same API. Second, its store is content-addressed and committed, so a planner can prove which episodes conditioned a given decision. That second property is the verifiability angle: tamper-evident, auditable, reproducible episodic memory for systems that act in the physical world.

Mneme ships in two modes under one interface: a training-free nonparametric corrector that works against any frozen predictor out of the box, and a trained cross-attention adapter that improves accuracy at the cost of a small offline training step.

The initial release is Python, built on PyTorch and FAISS. That choice prioritizes reach into the ML community over minimal footprint, so the people most likely to adopt the primitive can install it in the stack they already use. It integrates with PyTorch latent world models (for example V-JEPA 2, the LeJEPA reference implementation, and DINO-WM) for encoding and prediction, with WorldForge as an orchestration component, and with WMCP for cross-implementation and shared or remote stores. A native acceleration core is a later option once the API and semantics are validated, and the interface is designed so that swap costs nothing at the API level (Section 11).

---

## 2. Background and motivation

### 2.1 Latent world models and the long-horizon failure

A latent world model encodes observations into a representation space and predicts future representations rather than future pixels. V-JEPA 2 is the canonical recent instance: a frozen Vision Transformer encoder (over one billion parameters) produces patch-token latents, and a separate action-conditioned predictor (V-JEPA 2-AC, roughly 300M parameters, 24 transformer layers, 16 heads, 1024 hidden units, block-causal attention, 3D rotary position embeddings for patches and 1D temporal embeddings for action and state tokens) predicts the next-frame representation given the current state, an action, and the end-effector state. Planning runs as model-predictive control: the Cross-Entropy Method searches action sequences, the energy of a sequence is the L1 distance between its predicted future representation and an encoded goal image, the first action executes, and the loop replans. LeJEPA (Balestriero and LeCun, 2025) recently gave this family a principled training objective, replacing the older stack of collapse-prevention heuristics with a single regularizer that drives embeddings toward an isotropic Gaussian. The LeJEPA reference code is PyTorch, and `lewm-rs` reproduces a LeWorldModel of this kind in pure Rust; the Python package targets PyTorch models first and reaches `lewm-rs` through an export adapter (Section 12).

This design buys abstraction, speed, and sample efficiency. It also has a documented weak point. The V-JEPA 2 authors report that rollouts degrade in accuracy over long horizons, which limits planning reliability when multi-step prediction is required without intermediate subgoals. The broader literature describes the same pattern as compounding latent-state error and distribution shift that produce kinematic drift and structural violations such as objects disappearing, and characterizes memoryless world models as "dream machines" that hallucinate plausible continuations instead of reconstructing previously observed content. The model has no episodic grounding: at step t it predicts only from the current state and a short context window, so when it revisits a location or returns to an earlier configuration it does not reliably recall what it saw before.

### 2.2 Why retrieval, and the LLM mirror

Language models faced the same structural problem, that a fixed-weight parametric model cannot hold the long tail or stay current, and the field converged on retrieval. The relevant prior art gives Mneme its design vocabulary:

- **kNN-LM** (Khandelwal et al., 2020) corrects a parametric model's output by interpolating it with a distance-weighted nearest-neighbor distribution over a datastore of cached contexts. It needs no training and updates by swapping the index.
- **RETRO** (Borgeaud et al., 2022) augments a transformer with chunked cross-attention over retrieved neighbors, amortizing the cost of using a large external store.
- **RAG and REALM** (Lewis et al., 2020; Guu et al., 2020) fuse retrieved evidence by concatenation or end-to-end-trained retrieval.

The same idea has older roots in reinforcement learning under the name episodic control: Model-Free Episodic Control (Blundell et al., 2016) and Neural Episodic Control (Pritzel et al., 2017) act by reading a memory of past state-action-outcome tuples, and MERLIN (Wayne et al., 2018) couples a predictive model with an external memory. The complementary learning systems framing (McClelland et al., 1995) motivates pairing a slow parametric model with a fast nonparametric memory.

Mneme maps these directly onto latent dynamics. The nonparametric corrector is kNN-LM applied to next-latent prediction. The trained adapter is RETRO's chunked cross-attention applied to a frozen world-model predictor. The store is the episodic memory of episodic control, but content-addressed and committed.

### 2.3 What is missing today

Retrieval and memory augmentation of world models already exist in research, which this document treats honestly in Section 4. What does not exist is any of the following, which is what Mneme provides:

1. A reusable, model-agnostic memory-and-retrieval layer with a stable interface, rather than a mechanism welded into one model for one task.
2. A content-addressed, committed store that yields retrieval provenance, so a decision can be tied to the exact set of episodes that conditioned it and that set can be verified against a signed commitment.
3. A Python implementation in the PyTorch ecosystem, packaged as a maintained, reusable library that the ML community can install and adopt directly, rather than bespoke research code tied to one paper.
4. A single interface that covers both a training-free path and a trained path, so adoption costs nothing to start and improves with optional training.

---

## 3. Goals and non-goals

### Goals

- Provide an episodic memory and retrieval layer that reduces long-horizon drift and improves loop and revisit consistency for latent world models, measured against a no-memory baseline.
- Stay model-agnostic. Any encoder and predictor that expose the WM-RFC-0002 hooks can use Mneme without architectural changes.
- Ship a nonparametric path that requires zero training and a parametric path (a small adapter) that improves accuracy, both behind one interface.
- Make episodic memory content-addressed, tamper-evident, and auditable, with retrieval receipts that prove which committed episodes conditioned a decision.
- Run at control rates on a single workstation. The reference target is the author's Apple M4 Max with 48 GB. The heavy paths run in native libraries (FAISS search, PyTorch ops, BLAKE3), so per-step retrieval stays within typical manipulation control budgets. Minimal-footprint edge and embedded deployment is a later concern and may motivate a native core (Section 11).
- Launch as a Python package on PyPI so the broader ML community can adopt it in the stack they already use, and integrate cleanly with PyTorch latent world models, WorldForge, and WMCP.

### Non-goals

- Mneme is not a new world-model architecture or training objective. LeJEPA and V-JEPA own that layer; Mneme sits beside the predictor.
- Mneme is not a generative or video model. It stores and returns latents, not pixels.
- Mneme is not a benchmark suite. It consumes existing benchmarks (Section 13) rather than defining new ones.
- Mneme v0 does not provide confidentiality. The store is integrity-protected and auditable, not encrypted-by-default. Confidentiality is a later, optional tier.
- Mneme v0 does not prove that a retrieval returned the exact correct top-k. Membership of returned items in the committed store is proven; proving search correctness is a research frontier (Section 10, Tier 3).
- Mneme is not tied to one embodiment, simulator, or dataset.
- The initial release does not target minimal footprint or a native runtime. Edge, embedded, and a native core are later milestones, not launch goals.

---

## 4. Prior art and positioning

This section states the precedents plainly so the novelty claim is narrow and defensible.

**Memory-augmented world models.** UniWM (2025) builds a memory-augmented world model for visual navigation using similarity-based retrieval and temporal weighting, and reports navigation success improvements up to roughly thirty percent over memoryless baselines. ESWM (2025) builds spatial world models from a bank of sparse episodic transitions and plans long trajectories over them. WorldGPT (2024) attaches a knowledge-retrieval system over state transitions to an LLM-based world model to keep predictions temporally consistent. R-WoM (2025) is a retrieval-augmented world model for computer-use agents. A direct architecture study, "A comparison of memory mechanisms in world models" (2025), finds that augmenting a short-context world model with memory improves recall of past states and loop-closure modeling, that attention-based injection of memory (cross-attention) outperforms LoRA-style and state-space injections on latent error, and that online-updated memory weights can destabilize prediction. These findings inform the conditioning design and the risk section directly.

**Retrieval-augmented decision making.** RA-DT (2024), the Retrieval-Augmented Decision Transformer, equips a Decision Transformer with an external vector index and cross-attends over retrieved sub-trajectories to predict the next action, pairing a parametric policy with a nonparametric memory. This is the closest precedent in spirit. Mneme differs in target (latent world-model prediction and planning, not return-conditioned action prediction), in being a reusable layer rather than a model, and in the committed store.

**What Mneme adds over all of the above.** A stable, model-agnostic interface; a content-addressed and committed store with retrieval provenance; a maintained Python library in the standard ML stack rather than bespoke per-paper code; and one interface spanning the training-free and trained paths. None of the precedents is reusable infrastructure, and none treats the memory as a verifiable, auditable object. The verifiability layer (Section 10) has no precedent in this space.

---

## 5. System overview

Mneme has a write path that ingests realized experience and a read path that conditions prediction during a rollout. The world model's encoder and predictor are external and treated as frozen by default.

```text
                    +-----------------------------+
   observation o_t  |     Encoder (frozen)        |   z_t (latent)
  ----------------> |   PyTorch world model hook  | -----------+
                    +-----------------------------+            |
                                                               v
  WRITE PATH (after a real step)                     READ PATH (per planning step)
  --------------------------------                   ------------------------------
  build Transition{z_t, a_t, z_{t+1}, t, ep_id}      q = Summarize(z_t [, z_goal] [, a])
  key   = Summarize(z_t)                                       |
  cid   = BLAKE3(canonical(item))                              v
  append to log (MMR)                                +-------------------+
  insert key -> Index (FAISS HNSW)                   |  Index (ANN)      |  ids, distances
  update Merkle root R                               +-------------------+
            |                                                  |
            v                                                  v
  +-------------------+                              +--------------------------+
  |  Memory Store     |  <-- query(q,k,filter) --    |  Conditioner             |
  |  log + index +    |  -- items + receipt    -->   |  kNN-corrector  (default)|
  |  Merkle root      |                              |  or cross-attn adapter   |
  +-------------------+                              +--------------------------+
                                                                 |
                                          parametric pred z_hat  |  fused prediction z_pred
                                          from frozen predictor  v
                                                       +--------------------------+
                                                       |  Energy = ||z_pred - z_g|| |
                                                       |  CEM / MPC over actions   |
                                                       +--------------------------+
                                                                 |
                                                                 v   first action a*_t
                                                            (execute, observe, repeat)
```

Components:

- **Encoder hook.** Adapter to a PyTorch world model (V-JEPA 2, the LeJEPA reference, DINO-WM, or a custom predictor) that turns an observation into a latent and exposes an encoder fingerprint.
- **Summarizer.** Maps a (large) patch-grid latent to a compact key vector for indexing. Default is mean pooling plus L2 normalization; a learned summary token is optional.
- **Index.** Approximate nearest-neighbor structure over key vectors. Default is FAISS configured as HNSW.
- **Store.** Append-mostly log of items, the index, and a Merkle commitment over item content ids.
- **Conditioner.** Fuses retrieved items with the predictor's parametric output. Two implementations, selected by configuration.
- **Receipt builder.** Produces a signed retrieval receipt with Merkle inclusion proofs for the returned items.

---

## 6. Data model and core concepts

```text
Latent              opaque tensor in the encoder's representation space (NumPy array or torch tensor)
SummaryVec          compact f32 vector used as the index key (default dim 256 to 1024)
EncoderFingerprint  stable id of the encoder + summarizer that produced a key (hash of config + weights digest)

Transition {
  z_src:    Latent          # state at time t
  action:   ActionVec       # model-defined action encoding (e.g. 7D for the Franka setup)
  z_next:   Latent          # state at time t+1
  delta:    Latent          # z_next - z_src, stored for delta-mode correction
  reward:   Optional[float] # optional, not required
  t:        int             # step index within episode
  episode_id: UUID
}

Value = Transition | Frame{ z: Latent } | Window{ z: list[Latent] }

MemoryItem {
  content_id: Cid           # BLAKE3 over canonical(key, value, meta, encoder_fp)
  key:        SummaryVec
  value:      Value
  meta:       Meta          # episode_id, t, source tag, timestamp, custom fields
  encoder_fp: EncoderFingerprint
}

Retrieval {
  items:     list[MemoryItem]
  distances: list[float]
  receipt:   Optional[RetrievalReceipt]
}

RetrievalReceipt {
  root:    MerkleRoot       # store commitment at query time
  ids:     list[Cid]
  proofs:  list[InclusionProof]
  params:  QueryParams      # k, metric, ef, filters, summarizer id
  signer:  Optional[PublicKey]
  sig:     Optional[Signature]
}
```

Design notes:

- **Key and value are separate.** The index holds a compact key (the summary vector) for fast search and a low memory footprint. The value can be the full latent, the full transition, or a short window, which the conditioner needs but the index does not. This split keeps the index small while the values stay rich. Keys are float32 NumPy arrays so they feed FAISS directly; values are NumPy or torch tensors.
- **Delta mode.** Storing `delta = z_next - z_src` lets the nonparametric corrector predict a displacement to apply to the current state rather than an absolute next state. Displacement transfers better across nearby states, the same reasoning behind residual prediction.
- **Encoder fingerprint per item.** Keys are only comparable within one encoder plus summarizer. Every item records its fingerprint so the store can detect and reject or re-encode mismatched items (Section 9).
- **Content id is exact, the index is approximate.** This is the hinge of the verifiability design and is resolved in Section 10. The content id identifies an item by hash; the index selects which items to return by vector similarity. These are orthogonal.

---

## 7. Interface specification (WM-RFC-0002)

WM-RFC-0002 defines the abstract operations, the query semantics, the conditioning contract, and a JSON message schema for cross-implementation and remote stores. WMCP carries these messages; a local in-process store skips the wire and calls the same operations directly.

### 7.1 Abstract operations

```text
put(item: MemoryItem) -> Cid
put_batch(items: list[MemoryItem]) -> list[Cid]
query(q: QuerySpec) -> Retrieval
commit() -> MerkleRoot          # seal current state, return root
prove(ids: list[Cid]) -> list[InclusionProof]
root() -> MerkleRoot
stats() -> StoreStats           # size, memory, encoder fingerprints present
```

### 7.2 Query semantics

```text
QuerySpec {
  vector:        SummaryVec     # the query key
  k:             int            # number of neighbors
  metric:        Metric         # Cosine (default) | L2 | InnerProduct
  ef:            int            # HNSW search breadth, recall/latency knob (FAISS efSearch)
  filters:       Filters        # by episode_id, time range, source tag, meta predicates
  temporal_decay: Optional[float] # down-weight by age, applied to distances post-search
  with_receipt:  bool           # build inclusion proofs and sign
}
```

The query vector at planning step t is `Summarize(z_t)` by default. Two enrichments are supported:

- **Goal-aware retrieval.** Concatenate or average with the encoded goal `z_goal`, biasing retrieval toward transitions relevant to the current objective.
- **Action-aware retrieval.** Append a candidate action encoding, biasing toward transitions taken under similar actions. This is only used in the retrieve-per-imagined-step mode (Section 9), since it differs per candidate.

`temporal_decay` multiplies similarity by an exponential in item age, which lets recent experience dominate without deleting older items.

### 7.3 Conditioning contract

A `Retrieval` is the input to a `Conditioner`. The contract the predictor side must honor:

- The conditioner receives the parametric prediction `z_hat` (the predictor's own next-state output) and the `Retrieval`, and returns a fused prediction `z_pred` in the same space as `z_hat`.
- The fused prediction must reduce to `z_hat` when retrieval is empty or when the gate (Section 8) decides the neighbors are uninformative. This guarantees Mneme never degrades a model that is already correct in-distribution, up to the gate's calibration.
- The conditioner must not require gradients through the base model. Either it uses no training (the corrector) or it trains only its own parameters with the base model frozen (the adapter).

### 7.4 JSON message schema (abridged)

```json
// QueryRequest
{
  "type": "wmrfc2.query",
  "vector_b64": "<base64 f32le>",
  "dim": 512,
  "k": 16,
  "metric": "cosine",
  "ef": 128,
  "filters": { "time_range": [0, 100000], "source": ["robot_a"] },
  "temporal_decay": 0.001,
  "with_receipt": true
}

// QueryResponse
{
  "type": "wmrfc2.query_result",
  "items": [ { "cid": "...", "value_kind": "transition", "value_b64": "...", "meta": { } } ],
  "distances": [0.07, 0.09],
  "receipt": {
    "root": "...",
    "ids": ["...", "..."],
    "proofs": [ { "leaf_index": 41233, "siblings": ["...", "..."] } ],
    "params": { "k": 16, "metric": "cosine", "ef": 128, "summarizer": "meanpool-v1" },
    "sig": "..."
  }
}
```

---

## 8. Conditioning mechanisms

This is the core technical decision: how retrieved latents change the prediction. Mneme supports three mechanisms behind the conditioning contract. The first is the default because it needs no training and ships in v0.1. The second is the accuracy path. The third is the simplest attention option and a fallback.

### 8.1 Nonparametric corrector (default, training-free)

Port kNN-LM to next-latent prediction. Given the query `q = Summarize(z_t)`, retrieve neighbors with successor information. For neighbor i with distance `d_i` and stored successor displacement `delta_i` (delta mode) or successor latent `z_i_next` (absolute mode):

```text
w_i      = softmax(-d_i / tau)                      # distance-weighted, temperature tau
z_knn    = z_t + sum_i w_i * delta_i                # delta mode
         = sum_i w_i * z_i_next                      # absolute mode
lambda   = lambda_max * sigmoid(alpha * (delta0 - d_min))   # gate, d_min = nearest distance
z_pred   = (1 - lambda) * z_hat + lambda * z_knn
```

The gate `lambda` falls toward zero when the nearest neighbor is far (`d_min` large), so the prediction reverts to the parametric `z_hat` outside the store's coverage. The parameters `tau`, `lambda_max`, `alpha`, `delta0` are configuration in v0.1 and can be fit on held-out data later. This path is honest about its limit: it helps where the store densely covers the query neighborhood (revisited places, repeated manipulation configurations) and does nothing harmful where it does not, assuming the gate is calibrated. The whole corrector is a few lines of NumPy or torch over the retrieved arrays.

### 8.2 Parametric memory adapter (trained, accuracy path)

Insert cross-attention into the frozen predictor so it attends to retrieved value latents, in the style of RETRO's chunked cross-attention. Concretely, add a memory input head that projects retrieved value latents into the predictor's hidden width, mirroring the predictor's existing per-modality input heads (the V-JEPA 2-AC predictor already uses separate linear heads for patch, state, and action tokens). Add a cross-attention sublayer in a subset of predictor blocks whose queries are the predictor's hidden states and whose keys and values are the projected retrieved latents. The adapter is a `torch.nn.Module`. Train only the memory head and the cross-attention sublayers; keep the encoder and the original predictor weights frozen. Train on the same interaction data used to fit the action-conditioned predictor.

The architecture comparison study cited in Section 4 supports this choice: attention-based memory injection outperformed LoRA-style and state-space injections on latent error, and a cache of encoded states attended to by the predictor gave the largest simple improvement on reconstruction and loop closure. The adapter costs an offline training pass and a modest parameter count, and it can use larger k more gracefully than in-context concatenation because cross-attention does not grow the predictor's self-attention sequence.

### 8.3 In-context tokens (fallback)

Append the top-k retrieved value latents as extra tokens in the predictor's block-causal context. No new parameters, trivial to implement, useful as a baseline and for predictors where adding sublayers is awkward. The weakness is that attention cost grows with k and long retrieved context can dilute the signal, a known scaling concern from the retrieval-augmented language-model literature.

### 8.4 Comparison

| Mechanism | Training | Inference cost | Strengths | Weaknesses | LLM analog |
|---|---|---|---|---|---|
| Nonparametric corrector | None | One ANN query plus a weighted sum | Ships day one, swap index to update, reverts safely via gate | Helps only where store covers the query, sensitive to summarizer quality | kNN-LM |
| Cross-attention adapter | Offline, adapter only | One ANN query plus cross-attention | Best accuracy, scales with k, base model untouched | Needs a training step and a small parameter budget | RETRO |
| In-context tokens | None or light | Longer self-attention sequence | Trivial, no new params | Scales poorly in k, can dilute | RAG concatenation |

v0.1 ships 8.1. v0.2 adds 8.2. 8.3 is available throughout as a baseline.

---

## 9. Indexing, storage, and runtime

### 9.1 Index

The minimal-install default index is the flat exact backend, which serves small
stores and ground-truth recall measurement. FAISS HNSW is the first approximate
backend behind the `index` extra, configured over key vectors with cosine
similarity (inner product on L2-normalized keys) and recall tunable through the
`efSearch` parameter. Python call overhead adds to search time but stays within
typical manipulation control budgets for the per-real-step mode (Section 9.4).
Later backends can be added behind the same swappable index protocol without
changing the store contract.

### 9.2 Storage and footprint

Patch-grid latents are large, so the key and value split matters. The index stores only the summary key (a few hundred to about a thousand f32 values per item). Values store the full latent or transition, optionally product-quantized. A rough budget: at a 512-dimensional f32 key, one million keys occupy about two gigabytes in the index before quantization, which fits the reference workstation; values are kept on disk or as memory-mapped NumPy arrays and loaded on retrieval. The store is append-mostly. Retention policies (cap by count, by age, or by per-region density) bound growth without requiring deletes that would complicate the commitment.

### 9.3 Encoder versioning

Keys are only comparable within one encoder plus summarizer. Each item carries an `EncoderFingerprint`. On query, the store checks that the query fingerprint matches the indexed fingerprint and rejects or routes mismatches. When an encoder changes, the store supports lazy re-encoding (re-summarize values into a new index under the new fingerprint) so the value corpus is preserved while keys are rebuilt. This is the price of a frozen-encoder assumption that is later relaxed, and it is stated rather than hidden.

### 9.4 Runtime modes

- **Retrieve-per-real-step (default).** Query once per real environment step on `z_t`, then reuse the retrieved set across all CEM candidate rollouts at that step. One FAISS query plus the Python glue per control step fits comfortably inside manipulation control loops that run at roughly five to fifty hertz.
- **Retrieve-per-imagined-step (opt-in).** Query at each imagined step of each candidate rollout, using action-aware queries. This is the expensive mode, since CEM evaluates hundreds of candidate sequences over a horizon, and Python plus the GIL make it more sensitive to per-call overhead. It is supported with batched FAISS queries for users who need it, it is off by default, and it is a candidate for the native core if it becomes a bottleneck.

---

## 10. Verifiability layer

This is the property no precedent provides and the reason Mneme fits a high-assurance posture. The goal is that a planner's memory is tamper-evident and auditable, and that a decision can be tied to the exact episodes that conditioned it.

### 10.1 Content addressing and commitment

Every item has a content id `cid = BLAKE3(canonical(key, value, meta, encoder_fp))`, using the `blake3` package. The store maintains a Merkle Mountain Range over item content ids in append order; the MMR peak set hashes to a single root `R`, the store commitment. The root is cheap to update on append and supports inclusion proofs without rebuilding. The root can be signed (Ed25519 via `cryptography` or `pynacl`) by the store operator and published or logged.

### 10.2 Retrieval receipts

A query with `with_receipt = True` returns the root `R` at query time, the returned content ids, and a Merkle inclusion proof for each returned id. A verifier with `R` can check that every returned item is a committed member of the store at that root, and can re-derive each returned `cid` from the returned value to confirm the value was not altered. Given the same committed items and the same `QuerySpec`, the conditioner is deterministic, so a decision is reproducible.

### 10.3 The approximate-versus-exact resolution

The index is approximate, which seems to conflict with exact commitments. It does not, because the two layers answer different questions. The commitment proves membership: the returned items belong to the committed store and were not tampered with. The index decides selection: which committed members to return. v0 proves membership and integrity, which is what audit, provenance, and reproducibility require. v0 does not prove that the returned set is the true top-k under the metric, since the index is approximate by construction. Proving search correctness is a separate problem, addressed only as a research tier below.

### 10.4 Tiers

| Tier | Guarantee | Mechanism | Cost | Status |
|---|---|---|---|---|
| T1 | Returned items are committed members and unaltered; decision is reproducible | MMR commitment, inclusion proofs, signed root, deterministic conditioner | Low, proof size logarithmic in store size | v0.3 target |
| T2 | T1 plus binding of items to a specific encoder and an append-only history | Encoder fingerprint in the leaf, append-only log proofs | Low | v0.3 to v0.4 |
| T3 | The returned set is the correct top-k over the committed set under the stated metric | Verifiable search, for example a STARK over the selection, tied to Stwo / Circle-STARK and the pLeWM line | High, and partly open since the index is approximate | Research, not committed |

T3 is marked research deliberately. The author's pLeWM and ProvableWorldModel work shows the cost of STARK-proving model inference; proving search correctness over an approximate index is harder still and may require proving against exact nearest neighbors or against a verifiable approximate scheme. The choice of a Python launch does not change this; T3, if pursued, would call an external prover regardless of the host language. Promising T3 in v0 would be overclaiming.

### 10.5 Threat model

T1 and T2 defend against an operator or an attacker who silently injects, deletes, or alters episodes to steer a planner, since any such change moves the root and breaks inclusion proofs against a previously published root. They support post-incident audit, since the exact conditioning set for any logged decision can be recovered and verified. They do not provide confidentiality; an observer who can read the store sees its contents. Confidentiality (encryption at rest, private retrieval) is out of scope for v0 and noted as a later option, which is relevant given the author's privacy-first stance and worth a deliberate design pass rather than a default.

---

## 11. Reference implementation: `mneme` (Python)

### 11.1 Package layout

```text
mneme.core       types, dataclasses, protocols, errors, canonical serialization
mneme.index      Index protocol, flat exact backend, optional FAISS HNSW backend
mneme.store      MemoryStore, MMR commitment, receipts, retention, persistence
mneme.condition  Conditioner protocol, KnnCorrector, CrossAttnAdapter, InContext
mneme.encode     Encoder protocol, PyTorch adapters (V-JEPA 2, LeJEPA, DINO-WM), Summarizer
mneme.remote     schema-versioned messages, validation, HTTP client/server adapters
```

### 11.2 Core interfaces (typed Python)

Interfaces are `typing.Protocol` so any conforming object works without inheritance. Data carriers are dataclasses in `mneme.core`.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
from uuid import UUID
import numpy as np
import torch

Latent = torch.Tensor | np.ndarray      # opaque to Mneme except for shape
SummaryVec = np.ndarray                  # float32, the index key
Cid = bytes                              # BLAKE3 digest

class Encoder(Protocol):
    def encode(self, obs: object) -> Latent: ...
    def fingerprint(self) -> "EncoderFingerprint": ...

class Summarizer(Protocol):
    def summarize(self, z: Latent) -> SummaryVec: ...
    @property
    def id(self) -> str: ...

class Index(Protocol):
    def add(self, cid: Cid, key: SummaryVec) -> None: ...
    def search(self, q: SummaryVec, k: int, ef: int, metric: "Metric"
               ) -> list[tuple[Cid, float]]: ...
    def __len__(self) -> int: ...

class MemoryStore(Protocol):
    def put(self, item: "MemoryItem") -> Cid: ...
    def put_batch(self, items: Sequence["MemoryItem"]) -> list[Cid]: ...
    def query(self, spec: "QuerySpec") -> "Retrieval": ...
    def commit(self) -> "MerkleRoot": ...
    def prove(self, ids: Sequence[Cid]) -> list["InclusionProof"]: ...
    def root(self) -> "MerkleRoot": ...
    def stats(self) -> "StoreStats": ...

class Conditioner(Protocol):
    # Must reduce to `parametric` when retrieval is empty or uninformative.
    def condition(self, parametric: Latent, retrieval: "Retrieval",
                  ctx: "CondCtx") -> Latent: ...

@dataclass
class Transition:
    z_src: Latent
    action: np.ndarray
    z_next: Latent
    delta: Latent
    t: int
    episode_id: UUID
    reward: Optional[float] = None

@dataclass
class MemoryItem:
    content_id: Cid
    key: SummaryVec
    value: object            # Transition | Frame | Window
    meta: dict
    encoder_fp: "EncoderFingerprint"

@dataclass
class KnnCorrector:          # training-free: tau, lambda_max, alpha, delta0, mode
    tau: float = 0.1
    lambda_max: float = 0.5
    alpha: float = 10.0
    delta0: float = 0.2
    mode: str = "delta"
    def condition(self, parametric, retrieval, ctx): ...

class CrossAttnAdapter(torch.nn.Module):   # trained: projects retrieved latents + cross-attends
    def forward(self, hidden, retrieved): ...

@dataclass
class InContext:             # appends value latents as extra tokens
    def condition(self, parametric, retrieval, ctx): ...
```

The world-model predictor stays external. `mneme.encode` provides PyTorch adapters that implement `Encoder` and expose the predictor so the conditioner can produce `z_hat`. Keys are float32 NumPy arrays for FAISS; values and the trained adapter operate on torch tensors.

### 11.3 Packaging, tooling, and a native core later

The package installs from PyPI with optional extras: `mneme[index]` for FAISS,
`mneme[ml]` for torch-backed adapter paths, `mneme[receipts]` for optional
cryptographic receipt helpers, and `mneme[remote]` for serving the HTTP ASGI
adapter. It is developed with uv and linted with ruff, and ships type hints
throughout. The launch is pure Python on top of native libraries (FAISS,
PyTorch, BLAKE3) that carry the hot paths. If profiling later shows the Python
control-loop glue or the commitment path is a bottleneck, a native acceleration
core (for example a small Rust extension exposed through PyO3, reusing the
author's Rust work) can replace those hot paths without changing the public API,
since the protocols above are the contract. That optimization is explicitly out
of scope for the initial release.

---

## 12. Integration

- **PyTorch world models.** The launch integrates with PyTorch latent world models through the `Encoder` and `Conditioner` protocols: V-JEPA 2 (Meta's released checkpoints and predictor), the LeJEPA reference implementation, DINO-WM, and any custom PyTorch predictor. This is the broad-audience entry point and the reason for the Python-first release.
- **`jepa-rs` / `lewm-rs`.** The author's Rust models are reachable through an export adapter: load exported weights into an equivalent PyTorch module, or call the Rust inference through bindings. This keeps the existing ecosystem connected without making the Python launch depend on it.
- **WorldForge.** Mneme registers as a memory component in a WorldForge graph. A planning chain becomes encode, retrieve, condition, predict, plan, with Mneme owning the retrieve and condition steps. This is the concrete realization of the world-model orchestration picture.
- **WMCP / WM-RFC-0002.** The message types let a model talk to a remote or shared Mneme store over WMCP, so several agents or robots can read from and write to one store, with each retrieval carrying a receipt.
- **Lensemble (later).** Federated and sovereign pooling of episodic memory across nodes, where each node contributes committed experience and retrievals carry provenance across the federation. A v0.5 direction, reusing the commitment layer as the trust substrate.

---

## 13. Evaluation plan

The central claim to validate is that Mneme reduces long-horizon drift and improves task outcomes relative to a no-memory baseline, at acceptable latency.

### 13.1 Datasets and environments

- **Loop and revisit consistency.** The Minecraft loop-navigation setting used by the memory-aided spatial-consistency benchmark (revisit the same locations, measure consistency), which directly exercises episodic recall.
- **Embodied world-model quality.** EWMBench for visual scene consistency, motion correctness, and semantic alignment on robotic manipulation scenarios.
- **Navigation.** The datasets used by comparable memory-augmented navigation work (for example Go Stanford, ReCon, SCAND, HuRoN, and the 1X humanoid set) for downstream success rates.
- **Manipulation planning.** A V-JEPA 2-AC-style pick-and-place and reach setup, in simulation first, to measure planning success with and without memory.

### 13.2 Metrics

| Metric | Definition | Baseline to beat | Target direction |
|---|---|---|---|
| Latent rollout error vs horizon | L1 or L2 distance between imagined and realized latents at horizons 1, 2, 4, 8, 16 | Memoryless predictor's degradation curve | Lower at long horizons |
| Loop-closure consistency | Agreement of predicted latent on revisiting a location | Memoryless predictor | Higher |
| Task success uplift | Success rate on manipulation or navigation with vs without memory | No-memory rate | Higher |
| Retrieval recall@k, MRR | Against an exact-index ground truth | Flat-index reference | Near exact at chosen ef |
| Gate behavior | lambda as a function of nearest-neighbor distance and out-of-distribution inputs | n/a | lambda toward 0 when neighbors far |
| Write throughput | Items per second on the reference workstation | n/a | Sufficient for real-time logging |
| Query latency p50 / p99 | At store sizes 1e5, 1e6, 1e7 | n/a | Within control-loop budget |
| Memory footprint per item | Raw vs PQ-compressed | n/a | Fits 48 GB at target size |

### 13.3 Baselines and ablations

- No-memory baseline (the frozen predictor alone).
- Nonparametric corrector vs trained cross-attention adapter vs in-context tokens.
- FAISS HNSW recall vs latency sweep (vary `efSearch`).
- Store coverage: dense vs sparse coverage of the query region, to characterize where the corrector helps.
- Delta mode vs absolute mode.
- With vs without temporal decay.
- Encoder-fingerprint mismatch handling (reject vs lazy re-encode).
- Receipt overhead: latency and proof size as a function of store size, to confirm the verifiability layer is cheap at T1.

### 13.4 Success criteria for v0

Mneme reduces latent rollout error at horizon sixteen by a clear margin over the no-memory baseline on at least the loop and manipulation settings, the gate demonstrably reverts to the parametric prediction on out-of-distribution inputs (no harm where memory does not help), and a retrieval at one million items returns within the per-step control budget with a T1 receipt whose proof is logarithmic in store size.

---

## 14. Milestones

Each phase has an exit criterion. Phases are buildable solo or by a small team.

**v0.1, the wedge (training-free).** `mneme.core`, `mneme.index` (FAISS HNSW plus flat), `mneme.store` without commitments yet, `mneme.encode` with a V-JEPA 2 (and LeJEPA reference) adapter, and the `KnnCorrector`. Reproduce the no-memory long-horizon degradation curve, then show the corrector lowers latent rollout error at long horizons on the loop benchmark with zero training. Exit: a measurable drift reduction and a clean API, with the gate reverting on out-of-distribution inputs.

**v0.2, accuracy path.** `CrossAttnAdapter` (a `torch.nn.Module`) and the memory input head, trained offline on interaction data with the base model frozen. Compare against the corrector and the in-context baseline, run the ablations in 13.3. Exit: the adapter beats the corrector on long-horizon error and on at least one downstream task, base model untouched.

**v0.3, verifiability T1 and T2.** MMR commitments, signed roots, retrieval receipts with inclusion proofs, encoder-fingerprint binding, the reproducibility harness that re-runs a logged decision against a committed root. Exit: a verifier reconstructs and validates the exact conditioning set for a logged decision, and receipt overhead is logarithmic and within budget.

**v0.4, integration and release.** WorldForge memory component, WM-RFC-0002 over WMCP for a remote or shared store, and a documented public release: PyPI package, worked examples, and a Colab notebook that runs the corrector against a V-JEPA 2 checkpoint end to end. Exit: a WorldForge planning chain uses a remote Mneme store with per-step receipts, and a new user reproduces the v0.1 result from the notebook without local setup.

**v0.5, sharing and federation.** Cross-embodiment memory-transfer experiments and the Lensemble tie-in for federated, sovereign pooling with provenance across nodes. Exit: experience pooled across at least two sources improves a downstream metric on a third, with verifiable provenance.

**Frontier, not committed.** T3 verifiable search (STARK over selection, Stwo / Circle-STARK, pLeWM line), online or streaming memory with stability safeguards given the documented instability of online-updated memory weights, and a native acceleration core if profiling demands it.

---

## 15. Risks and open questions

- **Does retrieval reduce drift or only add latency.** It helps when the failure is memory-limited and the store covers the query region, and does little when the failure is dynamics-limited or the region is uncovered. The gate prevents harm in the latter case only if it is calibrated. Mitigation: distance-gated blending, a learned gate, and the coverage ablation to map where the gain lives.
- **The corrector can hurt.** kNN-LM-style interpolation can degrade prediction when neighbors are misleading. Mitigation: the gate, temporal decay, provenance to inspect what was retrieved, and the option to fall back to the adapter.
- **Encoder drift.** Keys depend on a frozen encoder. A changed encoder invalidates keys. Mitigation: per-item fingerprints, compatibility checks, lazy re-encoding. Stated as a cost, not hidden.
- **Approximate retrieval versus exact provenance.** Resolved by layering (Section 10.3). Exact top-k proofs remain open and are scoped to T3.
- **Latency under CEM.** Per-imagined-step retrieval is expensive, and Python plus the GIL make the batched mode more sensitive to per-call overhead. Default is per-real-step with shared neighbors; per-imagined-step is opt-in and batched, and is a candidate for the native core if needed.
- **Python performance ceiling.** The launch prioritizes ecosystem reach over minimal footprint. Native libraries (FAISS, PyTorch, BLAKE3) carry the hot paths, but the pure-Python orchestration between native calls, and the GIL, bound very high control rates and concurrency-heavy modes, and the package is heavier than a native build. Mitigation: keep the protocols as the contract so a native core can replace hot paths later without an API change; treat edge and embedded as a later milestone, not a launch goal.
- **Memory footprint.** Full patch-grid latents are large. Mitigation: key and value split, PQ on values, retention policies.
- **Stale or conflicting memories injecting noise.** A known retrieval-augmentation failure. Mitigation: gating, recency weighting, provenance, and retention.
- **Online updates destabilize.** Online-updated memory weights can collapse prediction. Mitigation: v0 store is append-mostly with an offline-trained adapter; online adaptation is deferred to the frontier with explicit safeguards.
- **Confidentiality.** v0 is integrity-protected, not confidential. Mitigation: scope confidentiality to a later tier and design it deliberately.

---

## 16. Security and privacy considerations

The commitment layer protects integrity and provenance, not confidentiality. Operators should treat the store as readable by anyone who can access it unless a later confidentiality tier is enabled. Signed roots should be published or logged to a tamper-evident location so that after-the-fact tampering is detectable against a prior root. Episodic memory of a real environment can contain sensitive data, so retention, access control, and an optional encryption-at-rest path are first-class concerns for any deployment that records human environments, and they are called out here rather than left implicit.

---

## 17. References

- Assran et al. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv:2506.09985, 2025. (Frozen ViT-g encoder; ~300M action-conditioned predictor, block-causal attention, 3D-RoPE; L1 feature-prediction loss; CEM and MPC with an L1 goal energy; documented long-horizon degradation.)
- Balestriero and LeCun. LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544, 2025. (SIGReg; isotropic Gaussian embeddings.)
- Khandelwal et al. Generalization through Memorization: Nearest Neighbor Language Models (kNN-LM). arXiv:1911.00172, 2020.
- Borgeaud et al. Improving Language Models by Retrieving from Trillions of Tokens (RETRO). arXiv:2112.04426, 2022.
- Guu et al. REALM: Retrieval-Augmented Language Model Pre-Training. 2020. Lewis et al. Retrieval-Augmented Generation (RAG). 2020.
- Blundell et al. Model-Free Episodic Control. arXiv:1606.04460, 2016. Pritzel et al. Neural Episodic Control. arXiv:1703.01988, 2017. Wayne et al. MERLIN. 2018.
- Paischer et al. Retrieval-Augmented Decision Transformer: External Memory for In-context RL. arXiv:2410.07071, 2024.
- UniWM: Unified World Models, Memory-Augmented Planning and Foresight for Visual Navigation. arXiv:2510.08713, 2025.
- ESWM: Building Spatial World Models from Sparse Transitional Episodic Memories. arXiv:2505.13696, 2025.
- R-WoM: Retrieval-Augmented World Model for Computer-Use Agents. arXiv:2510.11892, 2025.
- WorldGPT: Empowering LLM as Multimodal World Model. arXiv:2404.18202, 2024.
- A Comparison of Memory Mechanisms in World Models. arXiv:2512.06983, 2025. (Attention-based memory injection beats LoRA and SSM injections on latent error; online-updated memory can destabilize; cached encoded states help loop closure.)
- Toward Memory-Aided World Models: Benchmarking via Spatial Consistency (LOOPNAV). arXiv:2505.22976, 2025.
- EWMBench: Embodied World Model Benchmark. 2025.
- Ha and Schmidhuber. World Models. 2018. Hafner et al. Dreamer / DreamerV3.
- Johnson, Douze, Jégou. Billion-scale similarity search with GPUs (FAISS). arXiv:1702.08734, 2017.
- Malkov and Yashunin. Hierarchical Navigable Small World graphs (HNSW). 2018. Jégou et al. Product Quantization for Nearest Neighbor Search. 2011.
- NVIDIA. Cosmos 3 world foundation models for physical AI. 2026. (Context for the generative branch; not a dependency.)

---

## Appendix A. Notation and glossary

- **Latent.** A representation produced by the encoder, opaque to Mneme except for its shape (a NumPy array or a torch tensor).
- **Summary vector / key.** A compact float32 vector derived from a latent, used only for indexing.
- **Transition.** A stored `(z_src, action, z_next)` triple, optionally with a precomputed delta and reward.
- **Parametric prediction `z_hat`.** The world-model predictor's own next-state output, before conditioning.
- **Fused prediction `z_pred`.** The conditioner's output, used by the planner.
- **Gate `lambda`.** A scalar in [0, 1] controlling how much the nonparametric estimate contributes.
- **Receipt.** A signed object proving the returned items are committed members of the store at a given root.
- **Encoder fingerprint.** A stable id of the encoder plus summarizer that produced a key.

## Appendix B. End-to-end planning loop (pseudocode)

```python
while True:                                          # each control step
    z_t = encoder.encode(observe())
    q   = summarize(z_t)                             # optionally fuse with goal latent
    R   = store.query(QuerySpec(q, k, metric, ef, filters, with_receipt=True))
    log(receipt=R.receipt)                           # provenance for this step

    def score(action_seq):
        z, energy = z_t, 0.0
        for a in action_seq:
            z_hat = predictor.predict(z, a)          # frozen base model
            z     = conditioner.condition(z_hat, R, ctx)   # reverts to z_hat if gate ~ 0
            energy += l1(z, z_goal)
        return energy

    best = cem.optimize(score)
    execute(best.first())

    # after the real step settles
    z_next = encoder.encode(observe())
    store.put(MemoryItem.from_transition(z_t, best.first(), z_next, fp))
    store.commit()                                   # update root
```
