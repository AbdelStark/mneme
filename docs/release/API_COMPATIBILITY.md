# API Compatibility Gate

The compatibility suite pins the current pre-v1 public API surface in
`tests/fixtures/compat/public_api_snapshot.json`. The snapshot covers public
`__all__` exports, documented constructor and function signatures, protocol
importability, and schema-version constants.

Snapshot digest: `25309d88be3bc45036a9a5642f704a48c8e67c9adb2f944c0784a10a85e35224`

When a public export, signature, or persisted schema version changes, update the
snapshot intentionally and include a migration note or changelog entry that
explains the change. Before v1.0, breaking changes are allowed only with that
review note. After v1.0, the deprecation policy in
`docs/spec/02-public-api.md#deprecation-policy` and
`docs/spec/09-release-and-versioning.md#deprecation` applies: deprecate for at
least one minor release before removal unless a security issue requires faster
action.
