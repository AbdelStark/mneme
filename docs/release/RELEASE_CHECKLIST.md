# Release Checklist

Use this checklist before tagging a Mneme release candidate. The checklist is a
release gate, not a release announcement.

## Required Gates

- [ ] Confirm the target issue or release tracker links the intended scope.
- [ ] Run `ruff check .`.
- [ ] Run `ruff format --check .`.
- [ ] Run `pytest`.
- [ ] Run `mypy src/mneme`.
- [ ] Build source and wheel artifacts with `python -m build`.
- [ ] Install the built wheel in a clean environment.
- [ ] Import the installed package and record `mneme.__version__`.
- [ ] Generate a fixture report from the installed package.
- [ ] Validate artifacts and fixture evidence with the release validation command.
- [ ] Confirm README, SPEC links, CONTRIBUTING, SECURITY, CHANGELOG, LICENSE, and
  this checklist are present in the source artifact.
- [ ] Confirm the changelog has a user-visible entry for the release.
- [ ] Confirm public docs do not claim external task success, broad benchmark
  improvement, private retrieval, encrypted storage, or receipt verification
  unless matching evidence artifacts exist.
- [ ] Confirm the security boundary still states that stores are not
  confidential by default.
- [ ] Confirm optional extras that are implemented for the release have at least
  one install/import check.

## Artifact Validation Command

After building artifacts and writing the fixture report, run:

```bash
python -m mneme.release.validate_artifacts \
  --dist dist \
  --fixture-report .artifacts/release/fixtures.json \
  --out .artifacts/release/release-artifacts.json
```

The command prints a `mneme.release_artifact_report.v1` JSON report and exits
with code 0 only when the wheel, source artifact, installed package metadata,
public release docs, and fixture report satisfy the release contract.

## Release Notes Template

```markdown
# Mneme VERSION

## Scope

- Summarize the implemented user-visible changes.
- Link the closing issues and pull requests.

## Evidence

- Local and hosted CI gate run links.
- Artifact validation report path or attachment.
- Fixture report path or attachment.

## Claim Boundary

This release does not claim external task success, broad benchmark improvement,
private retrieval, encrypted storage, remote-store security, or receipt
verification unless those claims are backed by linked release artifacts.

Stores are not confidential by default. Treat value logs, manifests, reports,
and run outputs as sensitive when they contain real environment data.

## Compatibility

- Python versions tested.
- Optional extras tested.
- Known migration notes or breaking changes.

## Security

- Security-boundary changes.
- Known limitations and deployment requirements.
```
