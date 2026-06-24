# Mneme

Mneme is an early Python package scaffold for episodic memory and retrieval
around latent world models. The current repository defines the accepted
specification and RFC corpus; implementation work is landing issue by issue.

The package does not yet provide retrieval, storage, conditioning, or benchmark
claims. Those capabilities are tracked in the GitHub issue queue and must be
validated before they are described as working behavior.

## Install From Source

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -c "import mneme; print(mneme.__version__)"
```

Optional extras are declared but implemented only as their subsystem issues land:

```bash
python3 -m pip install -e ".[index]"
python3 -m pip install -e ".[ml]"
python3 -m pip install -e ".[receipts]"
python3 -m pip install -e ".[docs]"
python3 -m pip install -e ".[dev]"
```

The minimal package install keeps optional ML, index, receipt, and remote
dependencies out of the core import path.

## Specification

- [Product requirements](prd.md)
- [Accepted specification](SPEC.md)
- [Architecture](docs/spec/01-architecture.md)
- [Public API](docs/spec/02-public-api.md)
- [Release and versioning](docs/spec/09-release-and-versioning.md)
- [Implementation roadmap](docs/roadmap/IMPLEMENTATION.md)

## Development

```bash
python3 -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
python3 -m build
```
