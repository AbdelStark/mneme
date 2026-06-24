# Error Model

- Status: Accepted
- Created: 2026-06-24
- Source: [../../prd.md](../../prd.md#15-risks-and-open-questions)

## Principles

Errors must be typed, actionable, and stable enough for callers and command-line tools to handle. Public APIs do not expose backend-specific exceptions directly unless they are attached as `__cause__`.

## Error Taxonomy

```python
class MnemeError(Exception): ...
class ValidationError(MnemeError): ...
class SchemaVersionError(ValidationError): ...
class FingerprintMismatchError(ValidationError): ...
class ShapeError(ValidationError): ...
class DTypeError(ValidationError): ...
class StoreError(MnemeError): ...
class StoreCorruptionError(StoreError): ...
class TransactionError(StoreError): ...
class IndexError(MnemeError): ...
class IndexUnavailableError(IndexError): ...
class QueryError(MnemeError): ...
class EmptyStoreError(QueryError): ...
class UnsupportedOperationError(MnemeError): ...
class ReceiptVerificationError(MnemeError): ...
class OptionalDependencyError(MnemeError): ...
class EvaluationError(MnemeError): ...
```

## Public Operation Responses

`put`
: Validates schema, key shape, fingerprint, content id, and metadata. Raises `ValidationError` for invalid input, `TransactionError` for interrupted writes, and `StoreCorruptionError` for manifest/log mismatch.

`query`
: Validates query vector, metric, k, filters, fingerprint, and receipt support. Empty stores return an empty `Retrieval` unless the caller set a strict mode; invalid query parameters raise `QueryError`.

`condition`
: Empty retrieval returns `parametric`. Shape or device mismatches raise `ShapeError` or `DTypeError`. Numerical non-finite values raise `ValidationError`.

`commit` and `prove`
: Unsupported stores raise `UnsupportedOperationError`. Invalid proofs raise `ReceiptVerificationError`.

CLI
: Maps typed errors to stable exit codes defined in [Public API](02-public-api.md#command-line-surface).

## Failure Modes

- FM-001: query fingerprint does not match store fingerprint. Response: raise `FingerprintMismatchError`, include both fingerprints, never search mixed keys by default.
- FM-002: content id does not match canonical item bytes. Response: reject item on write; quarantine item on store verification.
- FM-003: index contains an id missing from the value log. Response: `store verify` reports corruption; query skips missing value only in repair mode.
- FM-004: value log contains items absent from index. Response: `index rebuild` can restore index from log.
- FM-005: optional backend not installed. Response: `OptionalDependencyError` with package extra name and fallback options.
- FM-006: approximate index returns duplicate ids. Response: de-duplicate with stable ordering and emit a warning metric.
- FM-007: non-finite latent, key, distance, or gate value. Response: reject and raise `ValidationError`.
- FM-008: receipt root differs from verifier root. Response: fail verification and include root mismatch details.
- FM-009: unsupported schema major version. Response: fail closed with `SchemaVersionError`.
- FM-010: partial transaction after process interruption. Response: manifest recovery either completes or rolls back the transaction before accepting writes.

## Recovery Commands

```text
mneme store verify PATH
mneme store repair PATH --mode index-only
mneme index rebuild PATH
mneme receipts verify RECEIPT_FILE --root ROOT_HEX
```

Repair commands must write a report before changing the store and must never delete value logs unless an explicit destructive flag is implemented later.

## Resolved Bootstrap Decisions

- Strict query mode belongs in store configuration, not `QuerySpec`. `QuerySpec` describes the semantic query; store policy decides whether empty stores, filtered-underflow results, and receipt-unavailable cases return empty results or raise typed errors.
