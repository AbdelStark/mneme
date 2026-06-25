# Mneme

Mneme is an episodic memory layer for latent world models. It stores compact
transition memories, retrieves nearby transitions, and conditions model
predictions through typed, inspectable contracts.

The 0.1.0 release is useful as a local research and infrastructure library:

- schema-versioned memory items, queries, reports, receipts, and remote messages;
- a durable local store with validation, retention, rebuild, and typed errors;
- exact flat retrieval by default and optional FAISS HNSW behind an extra;
- training-free kNN and in-context conditioning baselines;
- receipt-backed replay and fixture-scale evaluation reports;
- a uv-first CLI, package build, strict tests, and release artifact checks.

Mneme does not claim external task success, broad benchmark improvement, private
retrieval, encrypted storage, or production remote-store security. Public claims
must stay tied to generated artifacts and reproducible commands.

Stores are not confidential by default. Treat store directories, manifests,
value logs, reports, and run outputs as sensitive when they contain real
environment data.

## Start Here

- [Getting Started](getting-started.md)
- [CLI](cli.md)
- [Examples](examples.md)
- [Evaluation](evaluation.md)
- [Security](spec/06-security.md)
- [Public API](spec/02-public-api.md)

## Local Documentation Build

```bash
uv sync --locked --group docs
uv run mkdocs build --strict
```
