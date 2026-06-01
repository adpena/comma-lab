# Codex Findings: PR101 U32 Runtime Tree Materializer

UTC: 2026-06-01T17:22:21Z
Author: Codex
Axis: `[receiver-runtime-tree-materialization-only]`

## Landing

The PR101 grouped grammar lane can now emit a self-contained receiver runtime tree for `u32_decoder_len_adapter` archives:

- `inflate.sh`
- `inflate.py`
- `pr101_u32_adapter.py`
- `src/codec.py`
- `src/model.py`

The runtime tree vendors the override-aware in-repo PR101 codec implementation, keeps the PR101 model source from a supplied runtime root, and wires `inflate.py` through the generated u32 parser adapter. This closes the previous "adapter source exists but is not wired into a runtime tree" gap without claiming full-frame replay or exact eval authority.

The CLI refuses to materialize into a non-empty runtime output directory, so reruns do not silently overwrite runtime evidence.

## Local Proof

Focused test:

```text
uv run pytest src/tac/tests/test_pr101_per_tensor_grammar_solver.py -q
16 passed in 2.59s
```

Lint:

```text
uv run ruff check src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py tools/pr101_per_tensor_grammar_solver.py src/tac/tests/test_pr101_per_tensor_grammar_solver.py
All checks passed!
```

CLI smoke used `/Volumes/VertigoDataTier/pact/pr101_runtime_tree_cli_smoke_20260601T172221Z`, emitted report/grouped report/runtime tree/proof, byte-compiled generated `inflate.py`, `pr101_u32_adapter.py`, vendored `src/codec.py`, and `src/model.py`, printed proof summary, and deleted the scratch tree.

Smoke result:

```text
grouped selected bytes=198163; current grouped bytes=198508; saved=345; runtime=tac_decode_decoder_compact_with_overrides_required
schema pr101_u32_runtime_tree_materialization.v1
file_count 5
runtime_status u32_receiver_runtime_tree_materialized
blockers full_frame_inflate_parity_missing,contest_cpu_cuda_exact_eval_not_executed
files inflate.py,inflate.sh,pr101_u32_adapter.py,src/codec.py,src/model.py
```

## Verdict

The grouped grammar lane now reaches archive bytes plus a runtime tree. The remaining exact-readiness gate is no longer parser/runtime scaffolding; it is real `inflate.sh` full-frame replay against a byte-closed candidate archive, followed by CPU/CUDA auth only if local replay wins.
