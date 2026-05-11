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

Current tac-packet-compiler status:
- `crates/tac-packet-compiler` is the SCAFFOLD for the first native PacketIR
  proof against the committed Python golden vectors at
  `src/tac/packet_compiler/golden_vectors/`. Every public function is
  `unimplemented!()` and returns `PacketCompilerError::NotImplemented`. The
  parity test harness (`cargo test -p tac-packet-compiler`) currently
  asserts the scaffold contract and passes 11/11. Promotion gate: each
  function flips its parity test to `assert_sha256_parity` once an impl
  matches its committed golden vector byte-for-byte. See the crate's
  `README.md` and the operator-decision packet in `.omx/research/` for
  ownership + effort.
