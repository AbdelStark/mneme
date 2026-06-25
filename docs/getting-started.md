# Getting Started

## Environment

Mneme uses uv for dependency management, local tasks, examples, and release
checks.

```bash
uv sync --locked --group dev
uv run mneme --help
```

Optional runtime extras are installed only when needed:

```bash
uv sync --locked --extra index
uv sync --locked --extra ml
uv sync --locked --extra receipts
uv sync --locked --extra remote
```

## Minimal Store Query

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

## Quality Gate

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=mneme --cov-report=term-missing --cov-fail-under=80
uv run mypy src/mneme
uv run --group docs mkdocs build --strict
uv build --out-dir dist --clear --no-build-logs
```
