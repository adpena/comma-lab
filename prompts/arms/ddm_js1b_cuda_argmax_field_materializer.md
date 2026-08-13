# ddm_js1b — T4 argmax-field materializer: clear js1's BLOCKED_AXIS_MISMATCH with CUDA-custody fields

**Charter date:** 2026-08-13 · **Owner:** codex arm (xhigh) · **Consumer:** MAIN fires the T4 job (~$0.20–0.30, #381) → js1 stage-0 admission → V0–V5 ladder adjudication.
**Parents:** ddm_js1 stage-0 (`BLOCKED_AXIS_MISMATCH`, commit `3325810a9a`, memo `.omx/research/ddm_js1_stage0_per_edge_20260812.md`, receipt `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/STAGE0_RESULT.json`) + the PROVEN vd1 T4 infrastructure (dispatcher `experiments/ddm_vd1_modal_batch_event_validator.py` — lazy mounts, top-level module name, dual-path import, reserved-kwarg lessons ALL baked in; worker loader `experiments/ddm_vd1_batch_event_validator_worker.py` — cpr1 canonical decode + Brotli local-disk bootstrap, vd1b-proven). Read `docs/operating_manual_craft_handoff.md` + CLAUDE.md/AGENTS.md first.

## Why (the measured blocker)

js1's per-edge decomposition needs matched n600 SegNet argmax fields for BOTH promoted archives on the
CONTEST axis. The local Mac-CPU renderer fails both reference controls (CP135 local 50,395 flips vs
promoted 34,964; C1 target 17,927 vs batch-16 reference 17,926) — the batch-shape/axis instrument law
(et4/m23). js1's pre-registered fire trigger: "a working 1:1 T4 CUDA lane with artifact return becomes
available; retain both promoted argmax fields, recompute the matched n600 edge map and rho, then fire
V0–V5 only if Stage 0 is admitted." The vd1 chain IS that lane.

## The job

1. **T4 job (adapt the vd1 dispatcher+worker — reuse, never re-implement):** decode BOTH archives through
   their canonical receivers on Modal T4 and run the frozen SegNet argmax over all 600 pairs on CUDA:
   (a) cp135 composed archive `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip`
   (sha `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`, 186,252 B);
   (b) the T1R1/C1 archive js1 used — read its exact path+sha from STAGE0_RESULT.json (custody: do not guess).
   Persist BOTH full n600 argmax planes (uint8, 600×2×384×512 or the worker's native layout) + the GT argmax
   plane reference + per-plane sha256 to the retained volume. P0 KEEP THE PAYLOAD — planes, not just counts.
2. **Local post-step (build now, runs after MAIN's fire):** a `--from-argmax-fields` mode on js1's runner
   (`experiments/ddm_js1_stage0_per_edge.py`) that consumes the downloaded CUDA fields → matched n600 per-edge
   map + rho, replacing the refused local forward. Same-batch-shape discipline: the T4 forward must use the
   SAME batch shape as the promoted reference rows (batch 16 per the C1 control; verify from the js1 receipt).
3. **Admission gate carried forward:** stage-0 is ADMITTED only if the CUDA-custody CP135 flip count matches
   the promoted row's 34,964 (and C1 matches 17,926) — the same controls js1 refused on. If the T4 fields
   ALSO mismatch, that is a finding (promoted-row custody question), not a patch target — STOP and report.
4. **No Modal dispatch from the arm** (sandbox cannot reach Modal — js1's own dead-end). Land on main via
   serializer (post-edit working-tree shas) → final message = `READY_TO_FIRE` + pinned dispatch command +
   the pinned local post-step command + expected byte/time arithmetic (2 decodes ≈ 2×466 s + scorer passes
   vs the 1,800 s budget — state the K arithmetic).

## Custody / constraints
- Archives + adapted runtimes READ-ONLY. Single-flight + dual-ledger discipline stays with MAIN at fire time.
- No scorer authority claims; the arm runs NO scorer (advisory-free build). MPS NEVER.
- Falsifier: if the worker cannot decode the T1R1/C1 archive through its canonical receiver, report the exact
  seam (section, traceback) — do not hand-roll a parser (the vd1b lesson).
