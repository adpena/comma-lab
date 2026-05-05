---
name: Checks 72 + 73 STRICT — lane failure bug classes A + B extinct
description: 2026-04-29 PM. Subagent H landed 2 new STRICT preflight checks + SegMapTrainer.train_epoch batch-chunking. Closes 2 bug classes that wasted ~$3 today on Modal failures (Lane MM dead-flag scan, SA/SC++/SO T4 OOM). 33 new tests; 80 STRICT checks total now.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Bug class A — INVENTED CLI flags in shell-script invocations of experiments/*.py**
Why: tac.preflight.preflight_arity scanned Python launchers (modal_train_lane.py etc.) but NOT remote_lane_*.sh shell scripts. Each lane added its own inline regex scanner; some scanners false-positive matched comment text and killed the lane (Lane MM dead-flag bug, rc=3 in 4s).
How fixed: Check 72 `preflight_shell_lane_arity` adds shell-side coverage. Walks all 85 remote_lane_*.sh, finds every `("$PYBIN"|python3?) -u? experiments/<file>.py` invocation across line continuations, opens the target's argparse via AST, validates the used flags are a subset. Live scan: 85 scripts × 173 invocations clean → 0 violations → STRICT.

**Bug class B — CUDA OOM at training start on T4 (unchunked train_epoch)**
Why: SegMapTrainer.train_epoch processed all 600 pairs in one forward; --batch-size CLI flag was parsed but never used inside the trainer. Tried to allocate 7.03 GiB on T4 (14.56 total, 11.66 already used, 2.90 free).
How fixed:
1. SegMapTrainer.train_epoch accepts `batch_size: int = 8` kwarg. Pairs are mini-batched; gradients accumulate via `(loss / N).backward()` with single optimizer.step() per epoch. EMA + grad-clip preserved. Frame indices preserve global per-pair offsets so the per-frame affine embedding stays aligned across mini-batches.
2. experiments/train_segmap.py main loop probes train_epoch signature; passes args.batch_size when supported; loud-warns on legacy unchunked path.
3. Check 73 `preflight_t4_oom_training_guard` requires `--batch-size <= 32` OR `export GPU_TIER_HINT=A10G` for any `train_segmap.py` / `train_renderer.py` invocation. Live scan: 8 invocations clean → STRICT.

VRAM math now: batch_size=8 → 8 pairs × T=2 frames = 16 frames × scorer_input → ~1 GiB activations. Fits T4 with margin.

**How to apply**:
- Future SegMap-paradigm lanes can run on T4 (cheaper than A10G) with --batch-size 8.
- A10G remains the choice for batch_size > 32 (per Check 73 threshold).
- 33 new tests cover positive + negative + edge cases (line-continuation, pipeline-bound, non-target invocations, eq-when-batch=B equivalence, chunked-runs-and-steps, chunks-correctly, rejects-invalid-bs).
- Subagent H marked all 6 modified .py files reviewed by both council + codex (2-distinct-approver protocol).

Cross-refs:
- project_selfcomp_v2_failures_council_killlist_20260429
- feedback_dead_flag_wiring_pattern (the original Python-side scanner that this complements)
