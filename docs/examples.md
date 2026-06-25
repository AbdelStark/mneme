# Examples

The examples are synthetic rehearsals for public API and CLI behavior. They are
kept small enough to run in CI and are not external benchmark evidence.

```bash
uv run python examples/local_corrector.py
uv run python examples/remote_shared_store.py
```

Expected success signals:

- `local_corrector.py` prints JSON with `"ok": true` and
  `corrected_l2 < no_memory_l2`.
- `remote_shared_store.py` prints JSON with `"ok": true` and
  `"receipt_verified": true`.

Generated report examples:

```bash
mkdir -p .artifacts/examples
uv run mneme eval fixtures --out .artifacts/examples/fixtures.json
uv run mneme eval remote-conformance --out .artifacts/examples/remote-conformance.json
uv run mneme eval cross-source --out .artifacts/examples/cross-source.json
```

Remote/shared examples still require deployment controls outside Mneme:
authenticated transport, network policy, credential management, backup controls,
and external confidentiality protections when memories are sensitive.
