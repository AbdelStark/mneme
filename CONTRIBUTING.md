# Contributing

Mneme 0.1.0 is a typed Python ML infrastructure library. Keep changes small,
evidence-bound, and easy to review.

## Setup

Use uv as the project environment and task runner:

```bash
uv sync --locked --group dev
uv run mneme --help
```

Docs contributors also need the docs group:

```bash
uv sync --locked --group docs
uv run mkdocs serve
```

Optional runtime extras remain host-owned and lazy:

```bash
uv sync --locked --extra index
uv sync --locked --extra ml
uv sync --locked --extra receipts
uv sync --locked --extra remote
```

## Local Gates

Run focused tests while editing, then run the full local gate before opening a
pull request:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=mneme --cov-report=term-missing --cov-fail-under=80
uv run mypy src/mneme
uv run --group docs mkdocs build --strict
uv build --out-dir dist --clear --no-build-logs
```

When touching evaluation or release evidence, also run:

```bash
uv run mneme eval fixtures --out .artifacts/fixtures.json
uv run mneme eval remote-conformance --out .artifacts/remote-conformance.json
uv run mneme eval cross-source --out .artifacts/cross-source.json
```

## CI Reproduction

Hosted CI runs on pull requests and pushes to `main`. To reproduce the hosted
release gate locally from a clean checkout:

```bash
uv sync --locked --group dev
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=mneme --cov-report=term-missing --cov-fail-under=80
uv run mypy src/mneme
rm -rf dist .artifacts/ci .ci-install
uv build --out-dir dist --clear --no-build-logs
uv venv .ci-install
uv pip install --python .ci-install/bin/python dist/*.whl
.ci-install/bin/python -c "import mneme; print(mneme.__version__)"
.ci-install/bin/mneme eval fixtures --out .artifacts/ci/fixtures.json
.ci-install/bin/python -m mneme.release.validate_artifacts --dist dist --fixture-report .artifacts/ci/fixtures.json --out .artifacts/ci/release-artifacts.json
```

To reproduce the docs workflow:

```bash
uv sync --locked --group docs
uv run mkdocs build --strict
```

## Pull Request Discipline

- Use one branch and one pull request per issue.
- Keep optional dependencies behind their extras.
- Preserve typed failures for invalid schemas, stores, filters, and reports.
- Do not add benchmark, drift-improvement, privacy, receipt, or remote-store
  claims without a generated report or implemented validation path.
- Update docs and changelog entries for user-visible changes.

## Security And Privacy

Do not commit raw stores, private datasets, run outputs, secrets, local
credentials, generated coverage HTML, or local caches. Store directories are not
confidential by default; see [SECURITY.md](SECURITY.md).
