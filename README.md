# Mneme

Mneme is a pre-1.0 Python package for episodic memory infrastructure around
latent world models. It focuses on the parts around a model: schema-versioned
memory items, local persistence, exact retrieval, training-free conditioning,
in-context retrieved-token baselines, fixture-scale evaluation reports, and
redacted structured events.

Mneme is not a new world-model architecture and does not claim external task
success, broad benchmark improvement, private retrieval, or encrypted storage.
Fixture reports are useful for CI and claim discipline only; external benchmark
claims require separate reports.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -c "import mneme; print(mneme.__version__)"
```

Development tools:

```bash
python3 -m pip install -e ".[dev]"
```

Optional extras are declared for subsystem work and should be installed only
when that subsystem is needed:

```bash
python3 -m pip install -e ".[index]"
python3 -m pip install -e ".[ml]"
python3 -m pip install -e ".[receipts]"
python3 -m pip install -e ".[docs]"
```

The minimal package install keeps optional ML, approximate-index, receipt, and
remote dependencies out of the core import path.

## Minimal Usage

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

Fixture evaluation report:

```bash
python -m mneme.eval.fixtures --out .artifacts/fixtures.json
```

Opt-in external benchmark dry-run:

```bash
python -m mneme.cli eval benchmark --dry-run --dataset dataset.json --checkpoint CHECKPOINT --out reports/benchmark.json
```

The benchmark command writes a valid `mneme.eval_report.v1` envelope for runner
plumbing and claim review. The built-in dry-run runner is not benchmark
evidence.

## Security And Privacy

Mneme stores are not confidential by default. Treat readable store directories,
value logs, manifests, fixture reports, and run outputs as sensitive when they
contain real environment data. See [SECURITY.md](SECURITY.md) and
[Security](docs/spec/06-security.md).

## Limitations

- No encryption at rest.
- No private retrieval.
- No remote authentication.
- No production trained-adapter checkpoint or external trained-adapter report
  yet.
- The in-context conditioner is a baseline for compatible predictor wrappers;
  its attention cost scales with retrieved `k`.
- Local retrieval receipts verify committed membership and canonical item bytes,
  but signing, verifiable search, and private retrieval are not complete yet.
- No external benchmark or drift-improvement claim without an external report.

## Documentation

- [Product requirements](prd.md)
- [Accepted specification](SPEC.md)
- [Architecture](docs/spec/01-architecture.md)
- [Public API](docs/spec/02-public-api.md)
- [Observability](docs/spec/05-observability.md)
- [Security](docs/spec/06-security.md)
- [Testing strategy](docs/spec/07-testing-strategy.md)
- [Release and versioning](docs/spec/09-release-and-versioning.md)
- [Release checklist](docs/release/RELEASE_CHECKLIST.md)
- [Implementation roadmap](docs/roadmap/IMPLEMENTATION.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Public docs, examples, and reports must
stay within the evidence currently produced by tests and fixture reports.
