# Format0C Full-Stack Canonicalization - 2026-05-15

## Scope

Operator directive: make PR106/PacketIR/format0C score-table, proof, dispatch,
materialization, and submission-runtime surfaces canonical, contest-compliant,
and axis-safe. This is implementation evidence, not a score claim.

## Landed Fixes

- `experiments/build_pr106_latent_score_table.py`
  - Removed legacy `submissions/pr106_latent_sidecar` import binding.
  - Added explicit `--runtime-dir` and `--archive-member`.
  - Auto-detects `0.bin`, `x`, or a single member.
  - Uses the selected submission `inflate.py` runtime to parse sidecar packets.
  - Applies source sidecar corrections before scoring latent perturbations.
  - Emits member SHA, source payload kind, sidecar format id, runtime dir, and
    score-claim false custody.

- `src/tac/packet_compiler/pr106_runtime_consumption.py`
  - Added 0x0C exact-radix runtime decode support via
    `decode_pr101_exact_radix_fixed_meta_noop_rank_elided_sidecar`.

- `scripts/remote_lane_pr106_latent_sidecar.sh`
  - Added `PR106_RUNTIME_DIR` and `PR106_ARCHIVE_MEMBER` knobs.
  - Threads runtime/member into score-table generation.
  - Replaced legacy inline `0.bin` parser sanity with
    `tools/prove_pr106_sidecar_runtime_consumption.py`.
  - Binds auth eval to `$PR106_RUNTIME_DIR/inflate.sh`.

- `src/tac/deploy/kaggle/pr106_latent_score_table.py`
  - Bundles `submissions/pr106_latent_sidecar_r2_pr101_grammar`.
  - Bundles runtime-consumption proof tool used by the remote script.

- `src/tac/packet_compiler/pr106_latent_sidecar_selection.py` and
  `tools/materialize_pr106_latent_score_table_candidate.py`
  - Promote member-neutral custody: `source_archive_member_name` and
    `source_archive_member_sha256`.
  - Keep `source_zero_bin_sha256` only as a legacy fallback.
  - Fail on member-name mismatches.

- `tools/sub_0192_viability_guard.py`
  - Forced `contest_cuda` no longer falls back to `contest_cpu_score`.
  - Paired records select the axis-specific score.

- `experiments/train_substrate_stack_of_stacks.py`
  - Emits single-member `x`, matching its generated runtime.

- `tools/local_pre_deploy_check.py`
  - Replaced text grep with AST call-site detection so artifact globs like
    `contest_auth_eval*.json` no longer satisfy auth-eval reachability.

- `tools/all_lanes_preflight.py`
  - Gate now includes representative format0C `x` archive runtime-consumption
    proof.

## Generated Evidence

- Runtime proof:
  `experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/runtime_consumption_format0c.json`
  - format id: `0x0C`
  - member: `x`
  - runtime decode/apply: `true`
  - runtime-consumed sections: `pr106_payload=true`, `sidecar_payload=true`,
    `framing_meta=null`
  - score claim: `false`

- Score-table builder probe:
  `experiments/results/pr106_format0c_scoretable_builder_runtime_probe_20260515_codex/score_table_manifest.json`
  - member: `x`
  - source payload kind: `pr106_sidecar_packet`
  - sidecar format id: `0x0C`
  - runtime dir: `submissions/pr106_latent_sidecar_r2_pr101_grammar`
  - score claim: `false`

## Verification

- `90 passed`:
  - `src/tac/tests/test_pr106_latent_score_table.py`
  - `src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py`
  - `src/tac/tests/test_sub_0192_viability_guard.py`
  - `src/tac/tests/test_stack_of_stacks_dispatch_blocked.py`
  - `src/tac/tests/test_local_pre_deploy_check.py`
  - `src/tac/tests/test_materialize_pr106_latent_score_table_candidate.py`
  - `src/tac/tests/test_pr106_latent_sidecar.py`
  - `src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py`

- Ruff passed on all edited Python surfaces.
- `bash -n scripts/remote_lane_pr106_latent_sidecar.sh` passed.

## Remaining Explicit Blocker

`experiments/build_pr106_latent_sidecar.py` is still the legacy 0x01 materializer.
For format0C source packets, a future score-bearing materializer must compose
source sidecar corrections plus new score-table corrections into a byte-closed
PacketIR candidate instead of re-wrapping raw inner PR106 bytes through the old
0x01 grammar. Current audits now preserve runtime/member custody and fail closed
against member mismatches, so this is no longer silently conflated with a
dispatch-ready format0C materializer.

