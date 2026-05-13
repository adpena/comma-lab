# PacketIR Exact Closure V2 Runtime Binding - 2026-05-13

## Purpose

Close the PR106/R2 PacketIR parse/reemit loop without creating false score
authority. The closure artifact is diagnostic only: it records an already-run
contest-CUDA result, proves the archive/runtime/full-frame identity chain, and
blocks redispatch of the same exact-eval packet.

## Landing

- Code: `src/tac/packetir_exact_closure.py`
- Builder: `tools/build_pr106_r2_packetir_exact_closure.py`
- Queue guard: `src/tac/optimizer/exact_ready_audit.py`
- Artifact: `experiments/results/pr106_r2_packetir_exact_closure_20260513_codex/closure.json`
- Report: `experiments/results/pr106_r2_packetir_exact_closure_20260513_codex/closure.md`

## V2 Evidence Requirements

The closure now fails closed unless all of these hold:

1. PacketIR static result remains non-claiming and byte-accounted.
2. Runtime-consumption proof binds the candidate archive and has a non-empty
   score-affecting section map, all consumed by runtime, with exact set
   equality against the PacketIR static score-affecting section inventory.
3. Same-runtime full-frame parity binds source and candidate archives, complete
   pair coverage equal to the exact-eval sample count, equal streaming output
   SHA, and total frame bytes equal to the exact-eval inflated-output manifest.
4. Candidate CUDA eval is `[contest-CUDA]`, component-recomputed, and archive
   identity matched.
5. PacketIR source CUDA eval is supplied and archive matched.
6. Current-best CUDA reference is supplied and valid.
7. Runtime-consumption proof, full-frame parity proof, and exact CUDA eval all
   bind to the same `inflate.py` SHA; every runtime source file named by the
   consumption proof must match the exact CUDA runtime manifest file SHA.
8. Exact-eval duplicate key is emitted as
   `archive_sha256:runtime_content_tree_sha256:score_axis`.

## Fresh-Eye Review Fix

An adversarial subagent review found that the first V2 draft still accepted an
old runtime-consumption proof with `inflate.py` SHA
`5fa58d34dbf195e960d9a3db6370bf238c1e4e459de6cf3f11487b0a1f4b272f`, while the
exact CUDA auth eval and full-frame parity proof used
`60055bced3ab608d0e93ba83e18fa5bc662746cfa273ad50d5960c34028d1fb3`.

The fix was code-level, not documentary:

- `runtime_consumption_proof` now surfaces `runtime_inflate_py_sha256`,
  `runtime_source_tree_sha256`, and per-file runtime source SHAs.
- `runtime_identity_matches_cuda_eval_runtime` compares those file SHAs against
  the exact CUDA runtime manifest.
- The old proof now fails the closure until regenerated against the recovered
  exact-CUDA runtime.
- A recovered exact-CUDA runtime was transiently materialized from already-present
  custody surfaces whose file hashes match the exact CUDA manifest, used to
  regenerate the proof, then removed so ignored source files do not perturb the
  untracked-source baseline.
- `runtime_consumption.json` was regenerated from that recovered runtime and now
  binds to `inflate.py`
  `60055bced3ab608d0e93ba83e18fa5bc662746cfa273ad50d5960c34028d1fb3`.

Rebuild recipe for that transient runtime:

1. `inflate.py` and `inflate.sh` from
   `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/`
2. `src/codec.py` from `submissions/pr106_latent_sidecar_r2/`
3. `src/model.py` and `src/pr101_grammar.py` from
   `submissions/pr106_latent_sidecar_r2_pr101_grammar/`
4. Rerun `tools/prove_pr106_sidecar_runtime_consumption.py` against the PR106/R2
   candidate archive and overwrite the ignored
   `experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/runtime_consumption.json`
   proof.

## Result

- classification: `exact_measured_not_current_frontier`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- blockers: none
- duplicate key:
  `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d:c2ed377bdb6d8964601ec6646d72f2016b9488a130c03c58f62190f809786516:contest_cuda`

Interpretation: PR106/R2 PacketIR identity parse/reemit is an exact-measured
negative against current frontier, not a dispatch candidate. It stays useful as
a compiler/byte-closure control and as a guard against re-spending GPU on the
same archive/runtime/axis packet.

## Verification

- `ruff check src/tac/packetir_exact_closure.py src/tac/optimizer/exact_ready_audit.py tools/build_pr106_r2_packetir_exact_closure.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_optimizer_exact_ready_audit.py`
- `.venv/bin/python -m pytest src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_optimizer_exact_ready_audit.py::test_audit_blocks_ready_row_closed_by_packetir_exact_closure -q`
- `.venv/bin/python -m pytest src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_optimizer_exact_ready_audit.py -q`
- `.venv/bin/python tools/build_pr106_r2_packetir_exact_closure.py`
- `.venv/bin/python tools/audit_exact_ready_queues.py --suppression-manifest .omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json --format markdown`
- `.venv/bin/python tools/all_lanes_preflight.py`

The live exact-ready queue audit passed with 6 queues, 0 unresolved stale-ready
rows, and 5 manifest-suppressed/retracted rows.
