# CLI

The installed console script is `mneme`.

```bash
uv run mneme --help
```

## Store Tasks

```bash
uv run mneme store init .artifacts/demo-store
uv run mneme store stats .artifacts/demo-store --json
uv run mneme store verify .artifacts/demo-store
uv run mneme store commit-init .artifacts/demo-store
uv run mneme index rebuild .artifacts/demo-store
```

## Evaluation Tasks

```bash
uv run mneme eval fixtures --out .artifacts/fixtures.json
uv run mneme eval profile --store STORE --out .artifacts/profile.json
uv run mneme eval receipts --store STORE --out .artifacts/receipts.json
uv run mneme eval remote-conformance --out .artifacts/remote-conformance.json
uv run mneme eval cross-source --out .artifacts/cross-source.json
```

The fixture, remote-conformance, and cross-source commands are deterministic
release evidence. They are not external benchmark evidence.

## Release Artifact Validation

```bash
uv build --out-dir dist --clear --no-build-logs
uv run mneme eval fixtures --out .artifacts/release/fixtures.json
uv run python -m mneme.release.validate_artifacts \
  --dist dist \
  --fixture-report .artifacts/release/fixtures.json \
  --out .artifacts/release/release-artifacts.json
```

The validator checks built artifacts, package metadata, source contents, wheel
contents, and the fixture report envelope.
