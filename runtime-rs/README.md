# runtime-rs

Rust workspace for hot-path experiments.

This lane is intentionally separate from the Python harness.

Targets:
- Rust-backed Python AST/source indexing for fast preflight scans
- raw frame writer
- sparse residual decode
- ROI patch application
- future CPU inflator core

Current Python AST indexer status:
- `crates/python-ast-indexer` parses Python files with `rustpython-parser`
  and emits stable JSON for top-level functions, classes, imports, assigned
  names, line numbers, and function arguments.
- This is a first-stage replacement path for slow Python-side preflight AST
  scans. It is not score-bearing and does not mutate repository/provider state.
- Example: `cargo run -q -p python-ast-indexer -- ../src/tac/preflight.py`.

Current QMA status:
- QMA-family semantic mask parser/decode prototypes are still planned, but no
  QMA Rust crate is currently present in this workspace.

Current STBM1BR status:
- STBM1BR top-band semantic mask decode acceleration is still planned, but no
  STBM1BR Rust crate is currently present in this workspace.
