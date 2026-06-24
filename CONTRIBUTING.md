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
