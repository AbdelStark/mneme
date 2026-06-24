# API Compatibility Gate

The compatibility suite pins the current pre-v1 public API surface in
`tests/fixtures/compat/public_api_snapshot.json`. The snapshot covers public
`__all__` exports, documented constructor and function signatures, protocol
importability, and schema-version constants.

Snapshot digest: `33e4c51879fa40e64910838003b44c6bbdc3717b27e93a7735312f8b7322b5bd`

When a public export, signature, or persisted schema version changes, update the
snapshot intentionally and include a migration note or changelog entry that
explains the change. Before v1.0, breaking changes are allowed only with that
review note. After v1.0, the deprecation policy in
`docs/spec/02-public-api.md#deprecation-policy` and
`docs/spec/09-release-and-versioning.md#deprecation` applies: deprecate for at
least one minor release before removal unless a security issue requires faster
action.
