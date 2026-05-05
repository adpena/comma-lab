---
name: PHASE 1 dispatch verdict — codex GREEN/YELLOW/RED per lane + dispatch order
description: 2026-04-29 PM. Tight Phase 1 verdict codex (PID 37021, 222 lines) — gives per-lane dispatch readiness + recommended order. Lane 1 KL-distill in flight is RED (let finish for sunk-cost data, no further dispatch); Lanes 2/3/5/6 are GREEN (ready or cheap to validate); Lanes 4/7 are YELLOW (one specific blocker each); Lane 8 is RED (current postfilter-only impl doesn't satisfy canonical inflate_renderer pipeline).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Per-lane verdict (codex 2026-04-29 PM)

| Lane | Verdict | One-line reason |
|---|---|---|
| **1 SC++/q_faithful base** | 🔴 RED | KL distill violates hard-learned failure rule; let running jobs finish only if sunk-cost auth data is useful, no further dispatch |
| **2 Archive diet pose-delta** | 🟢 GREEN | encode/decode wired (commit 2d913687 + 3 regression tests), roundtrip verified, large byte win |
| **3 STC hybrid CUDA validation** | 🟢 GREEN | cheap CUDA validation with explicit 380KB kill threshold; current CPU smoke is invalid but not disqualifying |
| **4 Ω-W water-filling** | 🟡 YELLOW | tests pass, but blocked on a trustworthy checkpoint; do not couple to KL-distill Lane 1 output |
| **5 LCT** | 🟢 GREEN | adversarial fixes landed (commits 8bd467a9 + bcf11f20 + 90398b7c), tests pass, tiny 10-byte payload, low integration risk |
| **6 GP rerun** | 🟢 GREEN | real root-cause bug fixed (commit 8746793e: fit_pose_gp:33 baseline_poses kwarg FINALLY landed); CUDA rerun is now high expected-value |
| **7 PSD standard** | 🟡 YELLOW | profile exists (PSD_STANDARD_ADAPTIVE in profiles.py:168), but dispatch script gap remains; green after adapting known remote template |
| **8 Multi-pass inflate** | 🔴 RED | current implementation targets legacy postfilter path (trick_stack._stage_multi_pass), not canonical inflate_renderer; convergence/scorer constraints unresolved |

## Dispatch order (codex recommended)

**2, 5, 3, 6, 7, 4, 1, 8**

- Dispatch byte-safe, already-validated bolt-ons FIRST: Lane 2 then Lane 5
- In parallel, run cheap Lane 3 CUDA validation + Lane 6 fixed GP rerun (both clear uncertainty quickly)
- Lane 7 follows after script adaptation
- Lane 4 waits for a non-KL trustworthy checkpoint
- Lane 1 is monitor-only (no new dispatch)
- Lane 8 should NOT consume Phase 1 resources until canonical-path generalization exists

## Resource allocation (next 24h)

- **Modal: $15** (covers STC CUDA validation, GP rerun, PSD dispatch, short auth/eval jobs)
- **Vast.ai: $0** (avoid until a lane needs sustained multi-GPU training)

## Critical context: Lane 1 RED reason

Codex says KL distill violates "hard-learned failure rule". CLAUDE.md confirms: "KL distill caused PoseNet collapse as primary loss." But Quantizr uses `kl_on_logits(T=2.0)` for SegNet during specific training phases at weight=0.002. The Modal apps running 4-7h are mostly KL-distill profiles — let them complete and use whatever auth data emerges, but stop dispatching new KL-distill training.

## Gap to fix before dispatch wave

For each YELLOW lane:
- **Lane 4**: needs a non-KL base checkpoint. Could use Lane G v3 (1.05 baseline) or an earlier dilated_h64 checkpoint. Verify checkpoint provenance before bolting Ω-W onto it.
- **Lane 7**: needs `scripts/remote_lane_psd_standard.sh`. Adapt from `scripts/remote_lane_d_halfframe_retrain.sh` (357 lines) — replace `--profile dilated_h64_half_frame` with `--profile psd_standard_adaptive` + adjust output paths + heartbeat.

For Lane 8 RED:
- Need a NEW design: extend trick_stack._stage_multi_pass to wrap inflate_renderer.py canonical path
- Add convergence criterion (score plateau, max iterations, wall-clock cap)
- Strict-scorer-rule: compress-time iteration with score-feedback; deploy-inflate stays single-pass + scorer-free
- Defer to Phase 2 / 3 since Phase 1 has higher-EV work

## Cross-refs

- /tmp/codex_runs/phase1_verdict_tight.log (the verdict source)
- project_phase1_dispatch_state_corrections_20260429.md (Lane 7+8 partial-impl discovery)
- project_phases_2_3_4_design_implementation_math_provenance_20260429.md (Phase 2-4 lanes)
- project_6month_strategic_plan_20260429.md (high-level plan)
- feedback_silent_default_bug_class_findings_20260429.md (KL-distill bug class — STRICT preflight check 81 dc44b305)
