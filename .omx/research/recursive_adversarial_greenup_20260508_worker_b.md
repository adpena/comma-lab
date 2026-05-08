# Recursive Adversarial Greenup - Worker B - 2026-05-08

Scope: review latest roadmap/dispatch/codec changes at `main` commit
`0cf90d2b` for score-claim leakage, CPU-build dispatchability, exact/proxy
confusion, malformed archive/ZIP/runtime custody, and review-gate/preflight
regressions.

## Findings

1. PR106 UNIWARD CPU-build remains non-promotable by manifest and by dispatch
   fanout guard.
   - Artifact inspected:
     `experiments/results/pr106_uniward_runtime_packet_20260508_codex_smoke/build_manifest.json`
   - Archive: `150511` bytes, SHA-256
     `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b`.
   - Manifest flags: `score_claim=false`, `promotion_eligible=false`,
     `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`,
     `dispatch_attempted=false`, `evidence_grade="[CPU-build]"`.
   - A live spoof test forced `ready_for_exact_eval_dispatch=true` around the
     same manifest and `tools/parallel_dispatch_top_k.py --dry-run` still
     refused it via `blocked_evidence_semantics`.

2. Real bug fixed: `tools/parallel_dispatch_top_k.py` treated any readable
   non-empty ZIP as custody-sufficient. A ready row with exact SHA/byte metadata
   but unsafe member names, duplicate members, or local/central header name
   mismatch could pass local fanout validation before the deeper contest
   compliance checker. This was a malformed archive custody gap in the paid
   dispatch actuator.

3. Real bug fixed: `src/tac/codec/dual_layer_stc_av1_codec.py` accepted
   unknown wire-format flag bits during decode. The current format defines only
   `FLAG_EMPTY_MAGNITUDE`; accepting future/garbage bits was not fail-closed.

4. Exact/proxy roadmap state is still logically separated. Worker A's ledger
   and `roadmap_state_reconciliation_20260508_codex.md` already mark `0.20454`
   as unsupported without matching exact CUDA JSON and keep CPU-prep rows
   `ready_for_exact_eval_dispatch=false`. I did not find a current PR106
   UNIWARD manifest flipping that flag.

## Patches

- `tools/parallel_dispatch_top_k.py`
  - Added ZIP member safety checks before fanout:
    unsafe names, zip-slip paths, hidden/resource-fork sidecars, duplicate
    members, and local-header/central-header name mismatches now become
    `archive_custody:*` blockers.
- `src/tac/codec/dual_layer_stc_av1_codec.py`
  - Added `KNOWN_FLAGS_MASK` and fail-closed decode rejection for unsupported
    flag bits.
- `src/tac/tests/test_dispatch_command_builder_shapes.py`
  - Added regression tests for zip-slip and duplicate-member dispatch
    candidates.
- `src/tac/tests/test_dual_layer_stc_av1_codec.py`
  - Added regression test for unknown flag-bit rejection.

## Verification

Passed:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_dispatch_command_builder_shapes.py \
  src/tac/tests/test_dual_layer_stc_av1_codec.py
# 50 passed
```

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_build_pr106_uniward_runtime_packet.py \
  src/tac/tests/test_all_lanes_preflight_timing_profile.py
# 11 passed
```

```bash
.venv/bin/python tools/verify_pr106_uniward_runtime_packet_sha256.py
# rebuilt archive matched expected size=150,511 B and SHA-256
# 0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b
```

```bash
.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
# pass
```

```bash
# Temporary spoofed ranked input using the PR106 UNIWARD CPU-build manifest,
# force-ready=true, and the real archive SHA/bytes:
.venv/bin/python tools/parallel_dispatch_top_k.py --ranked-input <tmp> --dry-run
# refused with blocked_evidence_semantics:
# pr106_uniward_lagrangian_runtime_packet_byte_closed_cpu_build_no_score [cpu-build]
```

## Residual Risks

- The PR106 UNIWARD archive is byte-closed and rebuildable, but it remains a
  CPU-build/proxy artifact until exact CUDA auth eval lands on the exact archive
  and runtime. It should not rank, promote, kill, or update the frontier.
- `parallel_dispatch_top_k.py` now blocks common malformed ZIP custody hazards,
  but final release/upload readiness still requires
  `scripts/pre_submission_compliance_check.py --contest-final --strict` on the
  exact packet surface.
- The `0.20454` roadmap value remains a stale/unanchored formula projection
  unless a matching `contest_auth_eval.adjudicated.json` with archive/runtime
  custody is found.
