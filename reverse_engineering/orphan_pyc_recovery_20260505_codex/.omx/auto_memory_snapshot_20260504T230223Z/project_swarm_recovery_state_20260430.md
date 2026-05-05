---
name: 2026-04-30 ~05:00 CDT — Swarm recovery state checkpoint after API quota incident
description: Captures complete state of the 7-agent swarm + their findings AFTER quota cap hit at ~3am CDT. Multiple subagent commits stuck in lock-timeout / commit_failed but their WORK is on disk. Critical findings to preserve: Ω-W-V2 stack = 1.07 contest-CUDA REGRESSION; Lane GP v4 unanimous KILL; Lane Maturity Harness LANDED; Round 11 Q4A+Q4B LANDED.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Quota incident summary

User quota hit at ~3:30am CDT during the 7-agent swarm. 6 of 7 agents returned "You're out of extra usage · resets 4:50am" — meaning they exited mid-flight. Their WORK is on disk in many cases but commits are unconsolidated. Quota reset at 4:50am CDT.

## What ACTUALLY committed to git (verified via git log)

| SHA | What | Source agent |
|---|---|---|
| 98d8e17f | Lane maturity CLI (tools/lane_maturity.py audit/mark/unmark/validate/report/add-lane) | Maturity Harness (#279) |
| d8572047 | Lane G v3 + Ω-W-V2 stack [contest-CUDA] = **1.07** REGRESSION vs 1.05 | OWV2 dispatch (#272) |
| 4e505a0e | Lane maturity registry seed (23 lanes: 1 L3, 4 L2, 11 L1, 7 L0) | Maturity Harness (#279) |
| 8ce8b19e | Council unified Phase 1-4 battleplan | #278 |
| 02ce3a2c | Joint-ADMM Q4B adaptive rho_init Council #271 | Round 11 (#276) |
| 2192d960 | Joint-ADMM Q4A use final lam Council #271 | Round 11 (#276) |
| 9fef3382 | Lane Ω-W-V2 stack remote dispatch script | OWV2 dispatch (#272) |
| eff3020c | Lane G v3 + Ω-W-V2 stacked archive build script | OWV2 dispatch (#272) |
| 232b24ec | OWV2 inflate handler + 11 round-trip tests | OWV2 dispatch (#272) |

## Critical findings to preserve (NO SIGNAL LOSS)

### Finding 1: Lane Ω-W-V2 stack = 1.07 [contest-CUDA] — HARD REGRESSION
- Score breakdown: 100×seg = +0.003, **√(10×pose) = +0.052** (KILLER), 25×rate = −0.034
- PoseNet distortion went from 0.003455 → 0.005644 (+63.4%) due to OWV2 conv-weight perturbation
- Rate term saved exactly as predicted (−0.034) but PoseNet sensitivity overwhelmed
- The `40.98%` byte savings citation in earlier memory was conv-only-eligible-subset; full renderer is 20.59%
- **Lane G v3 = 1.05 [contest-CUDA] remains the best frontier**
- Hard kill rule fired (>1.05 = breaking score-relevant renderer behavior)
- Future revival paths: tighter `bit_budget_ratio=0.85`, PoseNet-sensitivity-weighted layer protection, real per-channel Hessian
- Memory cross-ref: feedback_owv2_savings_correction_conv_vs_full_renderer_20260430.md

### Finding 2: Lane GP v4 = UNANIMOUS KILL VERDICT
- Council reviewed 3 candidates (B-spline, DCT, natural cubic spline) over 4 rounds, 3/3 clean-pass
- **Empirical re-analysis** of actual Lane G v3 baseline `optimized_poses.pt` (600×6 fp32):
  - Pose trajectory in dims 1-5 is **white-noise** (`diff_std/signal_std ≈ 1.35` ≈ √2)
  - DCT energy fraction in top-40 of 600 bins is only 12-43% for dims 1-5 (uniformly distributed)
  - All 3 candidate bases plateau at avg RMSE ≈ 1.15-1.59 (near signal std 1.5-2.3)
  - Per-dim convergence is O(1/K^0.3) (sub-linear) — confirms non-smooth signal
  - To reach PoseNet noise floor (RMSE 0.01) requires K ≈ 500 (~6 KB), same order as raw fp16 (7 KB)
- **Previous Runge-phenomenon explanation was WRONG** — real cause is white-noise content
- **Lane PFP16 (fp32→fp16 cast, 7.0KB, ZERO distortion) DOMINATES** all 3 candidates
- Council documents on disk at `.omx/research/council_lane_gp_v4_design_20260430.md` + 4 round files
- Files NOT YET COMMITTED but present:
  - `.omx/research/council_lane_gp_v4_design_20260430.md`
  - `.omx/research/council_lane_gp_v4_round{1,2,3,4}_20260430.md`
  - `src/tac/tests/test_check_pose_basis_fit_kill.py` (14/14 tests pass)
  - `src/tac/preflight.py` (Check 91 added)
  - `experiments/fit_pose_gp.py` (kill marker added)
  - `src/tac/pose_gaussian_process.py` (kill marker in module docstring)
- Lane registration: `/tmp/lane_registration_lane_gp_v4.json`

### Finding 3: Lane 7 PSD = UNANIMOUS 10/10 REJECT (NO GPU SPENT)
- Council ran adversarial review BEFORE any dispatch (per user gating mandate)
- Decisive evidence: PSD historical = 1.49 (5× PoseNet regression vs dilated 1.33), already on `competition_state.killed_techniques`
- Files on disk uncommitted:
  - `.omx/research/council_lane_7_psd_dispatch_review_20260430.md`
  - `.omx/research/lane_7_psd_kill_memo_20260430.md`
  - `/tmp/lane_registration_lane_7_psd.json`
- Reactivation criteria: PSD-LumaSkip variant designed AND separate council approves AND current floor < 0.50

### Finding 4: Lane 12 NeRV — Level 1 → Level 2 LANDED (94.4% bytes saving)
- From serializer log: "Lane 12 (NeRV mask codec) Level 1 → Level 2: trainer + tests + dispatch + 94.4%"
- 94.4% bytes saving on the mask sequence (vs AV1 baseline 421KB → ~24KB target?)
- Files on disk: src/tac/nerv_mask_codec.py + experiments/train_nerv_mask.py + scripts/remote_lane_nerv.sh + tests
- Commit was attempted but blocked by lock timeout — commit text shows it WAS prepared

### Finding 5: Lane 8 multi-pass — LANDED with codec + integration + STRICT Check 92
- From serializer log: "Lane 8 multi-pass compress LANDED — codec + integration + STRICT Check 92"
- Commit was prepared but failed due to lock contention
- Files on disk: src/tac/multipass_compressor.py + tests + integration

### Finding 6: Lane Maturity Harness — LANDED (commits 4e505a0e + 98d8e17f)
- 23 lanes seeded: 1 L3 (Lane G v3), 4 L2, 11 L1, 7 L0
- CLI works: `python tools/lane_maturity.py audit/mark/unmark/validate/report/add-lane`
- Check 90 (preflight) status unclear — may be in PID 333 lock_timeout commit (still pending)

### Finding 7: Round 11 Joint-ADMM Q4A+Q4B — LANDED (commits 2192d960 + 02ce3a2c)
- 44 joint-ADMM tests pass
- Q4A: use final lam (not lam_avg) when converged
- Q4B: adaptive rho_init on iter 1
- Q4C deliberately skipped (Council #271 ruled MISDIAGNOSED)

## Pending commits (uncommitted work on disk)

1. **Preflight gate unblock** (parent's Check 83 false-positive fix + Check 79 subcommand-aware + 5 .sh waivers + smoke proofs) — 7 files staged, blocked by lock contention earlier
2. **Lane GP v4 KILL verdict** — 9 files (4 council reports + impl/test/preflight/markers)
3. **Lane 7 PSD KILL memo** — 2 files (council review + kill memo)
4. **Lane 12 NeRV codec** — multiple files (codec + tests + script)
5. **Lane 8 multi-pass codec** — multiple files (impl + tests + integration + Check 92)
6. **Maturity Harness Check 90** — preflight check + tests (PID 333 lock_timeout)
7. **Lane GP v4 lane registration** — `/tmp/lane_registration_lane_gp_v4.json`
8. **Lane 7 PSD lane registration** — `/tmp/lane_registration_lane_7_psd.json`
9. **OWV2 stack contest-CUDA result** — `experiments/results/lane_g_v3_omega_w_v2_stack_landed/{provenance.json,contest_auth_eval.json}` (committed at d8572047 — verify)

## Lane 17 IMP, 19 logit-margin, 20 Ballé — quota hit BEFORE producing artifacts

These 3 swarm subagents were killed by quota mid-flight. Need to:
- Check what's on disk (likely partial implementations)
- Re-spawn with explicit "continue from where you stopped" guidance
- All 3 have $≤$1.50 expected Vast.ai cost — proceed without re-budget approval
- Lane 17 specifically had a $25 budget gate — agent should have stopped at pre-dispatch memo (verify)

## Vast.ai instance state (verify clean)

Per OWV2 dispatch report (#272), 4 zombies were destroyed:
- 35886131 destroyed
- 35886299 destroyed
- 35886307 auto-destroyed by phase2-extract on driver-803 mismatch
- 35886609 destroyed after harvest

**ACTION ITEM**: run `vastai show instances` to verify zero leaked instances burning $$.

## Outstanding overnight GPU dispatches (status unclear)

The 6 overnight experiments dispatched earlier (HM-S, Ω-W-V2 stack, SC++, STC, SA-v2, SO):
- Ω-W-V2 stack: LANDED with 1.07 regression
- HM-S: dispatched on Vast.ai 35885106 (need to check)
- SC++, STC, SA-v2, SO: status unclear — need to check vastai/Modal dashboards

## Recovery actions (in priority order)

1. **Commit my preflight unblock bundle** — gate for everything else (8 files)
2. **Verify Vast.ai zero-zombie state** + reconcile
3. **Spawn quick-recovery agent for Lane 7 PSD commit** (just 2 files — council + kill memo)
4. **Spawn quick-recovery agent for Lane GP v4 commit** (9 files — KILL verdict + Check 91)
5. **Spawn investigation agent for Lane 12 NeRV / Lane 8 / Maturity Check 90 / Lane 17 / Lane 19 / Lane 20 commit recovery** — find their work on disk + commit each
6. **Re-spawn Lane 17/19/20 if their work is incomplete** (small Vast.ai dispatches if needed)
7. **Save Ω-W-V2 1.07 regression learning to memory** (if not already)
8. **Update CLAUDE.md if any new non-negotiable lessons** (e.g., Lane PFP16 fp32→fp16 dominance for pose stream)

## Cost summary tonight (approximate)

- Vast.ai OWV2 stack dispatch: ~$0.05 (50s GPU)
- Vast.ai zombies destroyed: ~$0.00 each
- HM-S in flight on Vast.ai: ~$1-2 (running 6h)
- Modal A10G overnight: SA-v2, SO ~$5-10 each
- Total tonight estimate: $15-25 of $100 cap (well under)

## Cross-refs

- feedback_production_hardened_standard_definition_20260430.md (Levels 0-3 + Level 3 7-gate checklist)
- project_session_state_checkpoint_20260430.md
- feedback_owv2_savings_correction_conv_vs_full_renderer_20260430.md
- project_lane_gp_v3_landed_runge_phenomenon_20260429.md (the SUPERSEDED Runge explanation)
- feedback_bash_run_in_background_kills_vastai_dispatch_20260430.md (SIGURG-144 trap that hit my own commit)
