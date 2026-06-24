# Contributing

Mneme is pre-1.0. Keep changes small, evidence-bound, and easy to review.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

## Local Gates

Run the relevant focused tests while editing, then run the full local gate before
opening a pull request:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
.venv/bin/mypy src/mneme
.venv/bin/python -m build
```

When touching evaluation or release evidence, also run:

```bash
.venv/bin/python -m mneme.eval.fixtures --out .artifacts/fixtures.json
```

## CI Reproduction

Hosted CI runs on pull requests and pushes to `main`. To reproduce the hosted
release gate locally from a clean checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
.venv/bin/mypy src/mneme
rm -rf dist .artifacts/ci .ci-install
.venv/bin/python -m build
python3 -m venv .ci-install
.ci-install/bin/python -m pip install --upgrade pip
.ci-install/bin/python -m pip install dist/*.whl
.ci-install/bin/python -c "import mneme; print(mneme.__version__)"
.ci-install/bin/python -m mneme.cli eval fixtures --out .artifacts/ci/fixtures.json
.ci-install/bin/python -m mneme.release.validate_artifacts --dist dist --fixture-report .artifacts/ci/fixtures.json --out .artifacts/ci/release-artifacts.json
```

## Pull Request Discipline

- Use one branch and one pull request per issue.
- Keep optional dependencies behind their extras.
- Preserve typed failures for invalid schemas, stores, filters, and reports.
- Do not add benchmark, drift-improvement, privacy, receipt, or remote-store
  claims without a generated report or implemented validation path.
- Update docs and changelog entries for user-visible changes.

## Security And Privacy

Do not commit raw stores, private datasets, run outputs, secrets, or local
credentials. Store directories are not confidential by default; see
[SECURITY.md](SECURITY.md).
