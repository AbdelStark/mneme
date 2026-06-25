# Mneme

[![CI](https://github.com/AbdelStark/mneme/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/mneme/actions/workflows/ci.yml)
[![Docs](https://github.com/AbdelStark/mneme/actions/workflows/docs.yml/badge.svg)](https://github.com/AbdelStark/mneme/actions/workflows/docs.yml)

Mneme is a Python 0.1.0 library for episodic memory around latent world models.
It gives an existing encoder or predictor a clean memory layer: schema-versioned
latent transitions, content-addressed storage, exact retrieval, training-free
kNN conditioning, receipt-backed replay, remote-store protocol fixtures,
redacted events, and JSON evaluation reports.

Mneme is not a new world-model architecture. It does not claim external task
success, broad benchmark improvement, private retrieval, encrypted storage, or
production remote-store security. The included reports are fixture-scale release
evidence for API, packaging, protocol, and claim discipline.

## Why Use It

- Model-agnostic memory objects for latent transitions and summary keys.
- A durable local store with manifest validation, retention, rebuild, and
  typed failure modes.
- Exact flat search by default; FAISS HNSW is optional behind the `index` extra.
- Training-free `KnnCorrector` and in-context retrieved-token conditioning
  baselines.
- Retrieval receipts and replay utilities for committed local-store evidence.
- A modern `mneme` CLI for store, query, receipt, and evaluation tasks.
- uv-first packaging, locked development environments, strict CI, and a
  GitHub Pages documentation site.

## Install

From a checkout:

```bash
uv sync --locked --group dev
uv run mneme --help
uv run python -c "import mneme; print(mneme.__version__)"
```

Once Mneme is published to an index, add it to another uv-managed project with:

```bash
uv add mneme
mneme --help
```

To test a local wheel artifact instead:

```bash
uv pip install dist/mneme-0.1.0-py3-none-any.whl
mneme --help
```

Optional runtime extras stay out of the core import path:

```bash
uv sync --locked --extra index      # FAISS HNSW approximate index
uv sync --locked --extra ml         # torch-backed adapter components
uv sync --locked --extra receipts   # optional cryptography integrations
uv sync --locked --extra remote     # ASGI serving dependencies
```

Documentation tooling is a dependency group, not a runtime extra:

```bash
uv sync --locked --group docs
uv run mkdocs serve
```

## Quickstart

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np

from mneme.core import EncoderFingerprint, Metric, QuerySpec, Transition, build_item
from mneme.store import init_store, verify_store

fingerprint = EncoderFingerprint(
    encoder_id="example.encoder",
    summarizer_id="meanpool-v1",
    weights_digest=None,
    config_digest="blake3:example-config",
)

transition = Transition(
    z_src=np.array([1.0, 0.0], dtype=np.float32),
    action=np.array([0.1], dtype=np.float32),
    z_next=np.array([2.0, 0.0], dtype=np.float32),
    delta=np.array([1.0, 0.0], dtype=np.float32),
    t=0,
    episode_id=uuid4(),
)

item = build_item(
    transition,
    key=np.array([1.0, 0.0], dtype=np.float32),
    encoder_fp=fingerprint,
    meta={"safe_source": "readme-fixture"},
)

with TemporaryDirectory() as tmp:
    store = init_store(Path(tmp) / "store", active_fingerprints=[fingerprint])
    store.put(item)
    retrieval = store.query(
        QuerySpec(
            vector=np.array([1.0, 0.0], dtype=np.float32),
            k=1,
            metric=Metric.L2,
            encoder_fp=fingerprint,
        )
    )
    assert retrieval.items[0].content_id == item.content_id
    assert verify_store(store.path).ok
```

## CLI Tasks

All maintainer tasks run through uv:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=mneme --cov-report=term-missing
uv run mypy src/mneme
uv build --out-dir dist --clear --no-build-logs
```

Common Mneme commands:

```bash
uv run mneme store init .artifacts/demo-store
uv run mneme store verify .artifacts/demo-store
uv run mneme eval fixtures --out .artifacts/fixtures.json
uv run mneme eval remote-conformance --out .artifacts/remote-conformance.json
uv run mneme eval cross-source --out .artifacts/cross-source.json
```

Opt-in external benchmark runner plumbing:

```bash
uv run mneme eval benchmark --dry-run \
  --dataset dataset.json \
  --checkpoint CHECKPOINT \
  --out reports/benchmark.json
```

The built-in dry-run runner checks envelope plumbing only. It is not benchmark
evidence.

## Examples

```bash
uv run python examples/local_corrector.py
uv run python examples/remote_shared_store.py
```

The examples print JSON success signals and use synthetic fixtures. See
[Examples](examples/README.md) for prerequisites, expected output, generated
report paths, and security boundaries.

## Documentation

The documentation site is published from `main` with GitHub Pages:

- [Documentation site](https://abdelstark.github.io/mneme/)
- [Getting started](docs/getting-started.md)
- [CLI reference](docs/cli.md)
- [Examples](docs/examples.md)
- [Public API](docs/spec/02-public-api.md)
- [Security](docs/spec/06-security.md)
- [Release checklist](docs/release/RELEASE_CHECKLIST.md)

Build it locally with:

```bash
uv run --group docs mkdocs build --strict
```

## Security And Privacy

Mneme stores are not confidential by default. Treat readable store directories,
value logs, manifests, fixture reports, and run outputs as sensitive when they
contain real environment data. Remote/shared deployments require authenticated
transport, network policy, credential management, backup controls, and external
confidentiality protections when memories are sensitive.

Read [SECURITY.md](SECURITY.md) and the
[security spec](docs/spec/06-security.md) before using real environment data.

## Current Limits

- No encryption at rest.
- No private retrieval.
- No hosted authentication service for remote stores.
- No production trained-adapter checkpoint or external trained-adapter report.
- No external benchmark, task-success, or drift-improvement claim without a
  separate generated report.
- Retrieval receipts verify committed membership and canonical item bytes; they
  do not prove private retrieval, exact approximate-search optimality, or signed
  provenance.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Public docs, examples, and reports must
stay within the evidence produced by tests, fixture reports, and linked release
artifacts.
