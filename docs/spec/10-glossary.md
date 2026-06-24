# Glossary

- Status: Accepted
- Created: 2026-06-24

## Terms

Action vector
: Model-defined numeric action encoding associated with a transition.

Adapter
: Code that lets an external encoder, predictor, index, or remote service satisfy a Mneme protocol.

Canonical serialization
: Deterministic byte encoding used to compute content ids and verify persisted objects.

Conditioner
: Component that fuses a parametric prediction with retrieved memory items.

Content id
: Digest of canonical memory item bytes.

Delta mode
: Conditioning mode that applies a weighted average of stored successor displacements to the current latent.

Encoder
: External model or adapter that converts an observation into a latent.

Encoder fingerprint
: Stable identifier for the encoder, summarizer, weights digest, and configuration digest that produced a key.

Gate
: Scalar that controls how much retrieved memory affects the parametric prediction.

Index
: Search structure mapping summary vectors to content ids.

Latent
: Opaque representation produced by an encoder.

Memory item
: Stored unit containing key, value, metadata, encoder fingerprint, schema version, and content id.

Merkle root
: Commitment root over appended content ids.

Parametric prediction
: The base predictor output before memory conditioning.

Receipt
: Object containing root, returned ids, inclusion proofs, query parameters, and optional signature.

Retrieval
: Query result containing items, distances, and optional receipt.

Summary vector
: Compact `float32` key derived from a latent for nearest-neighbor search.

Transition
: Stored `(z_src, action, z_next, delta, t, episode_id)` value.

Value log
: Append-mostly persisted storage for memory values.
