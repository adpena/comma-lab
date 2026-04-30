# GRAND COUNCIL Round 5 — Comprehensive Battleplan + Extreme-Rigor Adversarial Recursive Review

**Date**: 2026-04-29 PM (respawn after codex-CLI Pattern A run produced unstructured 3.2M file-walk log)
**Convened by**: parent agent at user's explicit request — "respawn that grand council comprehensive analysis ... let the council decide ... and those that hit OOM ... keep permanently fixing all issues and metabugs and bug classes"
**Inner council (10 voices)**: Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Sister councils running in parallel** (NOT duplicated here): DARTS-S freeze audit, UNIWARD v8 Fridrich+Shannon, OOM-class deep fix, EMA codebase audit. Sister reports were NOT yet on disk at the time of synthesis (`ls .omx/research/council_*_20260429.md` empty).
**Round counter**: 5 of N toward 3-clean-pass gate per `feedback_persistent_codex_review_protocol_20260429.md`.

---

## 1. Executive Summary

Phase 1.5 landed five lanes in a single concurrent-subagent burst: Lane Ω-W-V2 (water-fill+arithmetic terminal on block-FP weights), Lane 8 multi-pass inflate MVP, Lane Joint-ADMM coordinator, Lane PD-V2 (arithmetic-coded pose deltas), and Lane J-NWC (full neural weight codec wired). Modal harvest of 31 of 37 cached spawn() returns recovered five OK runs — most importantly Lane UNIWARD v8 at score 1.14 `[contest-CPU advisory]`, within the predicted [1.05, 1.18] band and competitive with the verified Lane G v3 1.05 `[contest-CUDA]` floor. Round 5 finds NO new critical bugs in the LANDED CODE (it ran the test suites; landing tests pass), but it DOES find FIVE structural CONCERNs that gate empirical promotion: (1) Lane Ω-W-V2's 69.11% saving is a synthetic-tensor measurement, not real-archive; (2) Lane 8's `_default_inner_step` is `NotImplementedError` so the "MVP outer loop" is exercised only via injected stubs in tests; (3) Lane Joint-ADMM's 2-stream KKT residual 0.02 is on a convex-quadratic synthetic and the real codec problem is non-convex+discrete; (4) Lane PD-V2's 18.6% empirical savings come from a deliberately-tuned "idle-dominant" fixture and the static-histogram arithmetic coder may overhead-gate-fail on a real high-entropy archive; (5) Lane J-NWC's predicted band [0.95, 1.30] is honest about codec amortization unfavorability on the 88K Lane A renderer and the real win requires a SHARED-CORPUS codec (which is not yet built). Verdict: **Round 5 is CLEAN-WITH-CONCERNS** — counter increments to **+1 toward gate** because no LANDED bug was found, but every Phase 1.5 lane carries a "needs real-archive empirical confirm" tag before any sub-1.0 promotion claim is permitted. Top dispatch this 24h: (a) Lane UNIWARD v8 CUDA-confirm at $0.50 Vast.ai 4090; (b) Joint-ADMM V2 wrap of `water_filling_codec_v2` as a real proximal codec on a real Lane G v3 archive (local-only, no GPU); (c) Lane Ω-W-V2 real-archive test on Lane G v3 / Lane A renderer.bin (local-only). Total wave budget remains well under the $30 Vast + $30 Modal cap.

---

## 2. Round 5 Adversarial Findings (per landed lane)

The findings table below uses these verdicts:
- **CLEAN** — code + tests + claims align; no blocking issues
- **CONCERN** — code is functional but a load-bearing claim depends on something that is NOT empirically established
- **BUG** — concrete code bug or false claim found in the LANDED code
- **RED** — code lands on a forbidden pattern or disagrees with CLAUDE.md

### 2.1 Lane Ω-W-V2 (commit `9987a5d9`, source files `src/tac/water_filling_codec_v2.py` + `src/tac/codec_magic_registry.py` + tests)

**Verdict: CONCERN**

**What landed**: V2 layers a static-histogram arithmetic coder ON TOP of V1 water-fill on per-channel block-FP qint streams. Magic byte `OWV2` registered in the new `codec_magic_registry.py`. Hard overhead gate `GateRegression` raises if encoded ≥ V1 raw-qint estimate. 13 tests pass (round-trip, byte savings, determinism, score-parity, eligibility).

**Round 5 findings**:

- **CONCERN-1 [load-bearing]** — The 69.11% byte savings claim `[empirical:src/tac/tests/test_water_filling_codec_v2.py::test_v2_byte_savings_vs_v1_raw]` measures a synthetic 24×24×3×3 tensor (`std=0.05`, seed 2) at a 3-bit/element budget. Selfcomp's actual 88K-param SegMap renderer is NOT this distribution: real conv weights have **bimodal magnitude** (small-many + large-few) plus per-layer magnitude variance the synthetic doesn't capture. Static-histogram entropy on a real renderer's qint stream may compress 30-50% (empirical), not 69%. **Fix**: add `tests/test_omega_w_v2_real_archive.py` that runs encode on a CHECKPOINT `.pt` from Lane G v3's archive; assert savings within [20%, 60%] band before promoting the docstring claim.
- **CONCERN-2** — Eligibility gate is `BlockFPIneligible` which only fires on non-4D conv tensors. A linear/embedding tensor falling through to V1 unmodified is correct; but the SegMap renderer has Conv2d AND Linear bias terms — the encode path needs to distinguish gracefully. Test coverage on a `state_dict.values()` iteration is missing.
- **CLEAN** — Magic byte registry is well-designed (uniqueness enforced at import time). Hard overhead gate matches Carmack rule.

**No bug found.** Promotion to "ship V2 in archive build" requires real-archive measurement first.

### 2.2 Lane 8 multi-pass inflate (commit `0e43d299`, source `experiments/multi_pass_inflate_optimizer.py` + tests)

**Verdict: CONCERN (verging on RED for any "production wired" claim)**

**What landed**: Outer-loop scaffold with extract → inner-step (injectable callback) → deterministic re-pack via `ZipInfo + writestr`. Score-plateau convergence with `patience=3, tol=0.0005, max_iters=5, wall_clock_cap_sec=4h`. 4 tests pass (cold-start, plateau, max-iters, byte-identical re-pack).

**Round 5 findings**:

- **CONCERN-1 [load-bearing]** — `_default_inner_step` raises `NotImplementedError`. The PRODUCTION GPU path is unwired. Tests pass because they inject a stub `inner_step_fn`. Memory entry feedback_three_active_bug_classes mentions "Lane PD docstring 49% → 18.5% empirical" as a docstring-overstatement bug; the same risk exists here: the commit message says "inner loop MVP wired" but the GPU inner step is in fact a `NotImplementedError`. **Verdict**: the LANDING is honest (commit body explicitly says GPU path raises NotImplementedError), but ANY downstream summary that cites Lane 8 as "implementation complete" would land in RED on Check 84 (empirical-claims-have-evidence) once that check goes STRICT.
- **CONCERN-2** — `_score_archive` calls `experiments/contest_auth_eval.py` per outer iter. At `max_iters=5` on a Lane G v3-class archive, that is 5 × ~15 min CUDA = 75 min per RUN. The wall-clock cap of 4h is correct but EXPENSIVE. Lane 8 V2 should batch the inner steps more (e.g. 5 TTO steps per outer iter) before the next contest_auth_eval call.
- **CONCERN-3** — `_check_score_plateau` measures `best_old - recent_best < tol`. If the loop ALSO regresses (recent_best > best_old), the plateau check evaluates to True (negative < tol). That is correct (regressions trigger early stop) but the manifest field `converged: True` is misleading in the regression case. **Recommend**: track `regressed: bool` separately.
- **CLEAN** — Deterministic ZIP write helper is correct. Best-archive-seen logic is correct. `# DETERMINISTIC_ZIP_OK` waiver is same-line per Check 15.

**No bug found** in the SCAFFOLD; the GPU production wiring is a documented Phase 2 follow-up.

### 2.3 Lane Joint-ADMM coordinator (commit `152ba503` source files, body swapped per concurrent-commit-message-swap incident — actual files `src/tac/joint_admm_coordinator.py` + `src/tac/joint_admm_proximal_pose_delta.py` + tests)

**Verdict: CONCERN**

**What landed**: Boyd 2011 §3.4 alternating-projections skeleton with adaptive ρ, restart-on-divergence, KKT waterline validator, dual-averaging extension for non-smooth subgradients. `StreamProximalCodec` Protocol. First concrete wrapper `joint_admm_proximal_pose_delta.py` over pose_delta_codec. 9 tests pass (KKT residual 0.02 on synthetic 2-stream convex problem).

**Round 5 findings**:

- **CONCERN-1 [load-bearing]** — KKT residual 0.02 is on a CONVEX QUADRATIC 2-stream synthetic (`QuadraticRateStream(a=0.01, b_opt=300)` + `QuadraticRateStream(a=0.02, b_opt=100)`). The REAL problem is:
  - 4-6 streams (renderer bytes, mask bytes, pose bytes, codebook bytes, header overhead, optional hyperprior side-info)
  - Per-codec rate-distortion functions are DISCRETE staircases (not differentiable), not smooth quadratics
  - SegNet and PoseNet sensitivity surfaces are non-convex
  - Boyd's adaptive ρ + dual averaging help but DON'T guarantee convergence on non-convex discrete problems

  Codex stacking memory explicitly warned: "ADMM divergence: use adaptive penalty, restarts, exact byte projection after every codec call." The SCAFFOLD has restart logic; the PROOF that it converges on the real archive does not exist yet.

- **CONCERN-2** — `joint_admm_proximal_pose_delta.py` uses a CACHED R(D) frontier (no live scorer load). That is strict-scorer-rule compliant ✓ but means the coordinator's optimization is over a STALE surface: if a codec slot's actual marginal at iteration k differs from the cached frontier from iteration k-N, the coordinator chases a phantom waterline. Real-archive ADMM needs a periodic frontier refresh.
- **CLEAN** — Mathematical formulation is correct (Boyd §3.4 lifted faithfully). Adaptive ρ matches §3.4.1. Strict-scorer-rule compliance is correct (no SegNet/PoseNet inside coordinator).

**No bug found.** Promotion to "ADMM is solving real problem" requires multi-stream non-convex test + real-codec wrapping (Phase 2 V2 work).

### 2.4 Lane PD-V2 (commit `152ba503` body swapped, actual files `src/tac/pose_delta_codec_v2.py` + `src/tac/submission_archive.py` + tests + `submissions/robust_current/compress_archive.py`)

**Verdict: CONCERN**

**What landed**: Static-histogram arithmetic coder over per-channel int8 pose-delta stream with per-channel fp16 scale + anchor. PDV2 magic. `encode_pose_delta_v2_or_fallback()` returns `format=pose_delta_v1` if the V2 blob is no smaller than V1's `torch.save()`. 19 + 11 tests pass.

**Round 5 findings**:

- **CONCERN-1 [load-bearing]** — The 18.6% empirical savings comes from a fixture that simulates idle-dominant trajectory (most deltas near 0, alphabet narrow). Real comma 1200-frame 0.mkv has motion segments where deltas are NOT idle-dominant. On a high-entropy segment the static-histogram coder may FAIL the overhead gate (200-byte freq table on a 3KB stream erases the gain) — at which point V1 fallback fires and we ship the same bytes as before. This is BENIGN (no regression) but means the predicted Δ -7-11bp from the codex is conditional on the real trajectory's actual entropy.
- **CONCERN-2** — `submission_archive.load_optimized_poses` dispatches the V2 dict to `decode_pose_delta_v2`. The dispatch table needs to be PERMANENTLY in the archive-load path; if any future archive-load codepath bypasses this dispatch, V2 archives will ship and inflate-side will silently load garbage poses. **Recommend**: add `test_archive_load_pose_delta_v2_dispatch.py` that asserts every callsite of `load_optimized_poses` consumes the V2 dispatch.
- **CLEAN** — Hard overhead gate is correct. Bit-determinism verified. Magic byte is registered. Round-trip max-abs error verified before encode returns.

**No bug found.**

### 2.5 Lane J-NWC (commit `12b43507`, source `submissions/robust_current/inflate_renderer.py` + `experiments/pipeline.py` + `scripts/remote_lane_nwc.sh` + tests)

**Verdict: CLEAN-WITH-NOTE**

**What landed**: NWC1 inflate-side dispatch added to `_load_renderer` (delegates to `tac.renderer_export.load_neural_compressed_checkpoint`). Pipeline producer wired with `weight_compression="nwc"` branch + `weight_codec_path` field. Remote dispatch script `scripts/remote_lane_nwc.sh` passes all 14 STRICT preflight checks. 19 tests pass (8 new + 11 existing). Bidirectional magic-byte test now passes.

**Round 5 findings**:

- **CONCERN-1 [load-bearing prediction]** — The predicted band [0.95, 1.30] `[prediction]` is HONEST about the codec amortization problem: 16K-param VQ-VAE codec dominates the binary at small renderer scales. For Lane A's 88K-param renderer: raw weights ≈ 16.5 KB (1.5 bits/weight), plus 64 KB amortized codec → total ~80 KB vs FP4A baseline's ~48 KB. **NWC1 LOSES on a single-renderer 88K target.** The architecture wins only when:
  - Renderer ≥ 150K params (codec amortizes) OR
  - Multiple renderers share ONE codec (corpus codec)

  Neither condition is satisfied today. Memory file at `project_lane_j_nwc_landed_20260429.md` line 84-86 explicitly calls out the corpus-codec follow-up as the path to amortization. **Verdict**: the LANDING is correct and tagged honestly, but Lane J-NWC dispatch on Lane A renderer is a NEGATIVE-EV move at this size. Do NOT dispatch until either (a) a larger renderer exists OR (b) corpus codec is built.

- **CLEAN** — Strict-scorer-rule compliant (16K-param VQ-VAE is not a scorer). Magic byte `NWC1` registered. SCv1/OMG1/CCh1/SZv1 lane policy mirrored correctly (require tac wheel; no inline fallback). Predicted band tagged `[prediction]` per Check 84-deferred convention.

**No bug found.**

### 2.6 Modal harvest results — adversarial rigor on the 5 OK runs

**Verdict: ADVISORY ONLY — needs CUDA confirm before any kill/promote**

| Lane | Score | Tag | Round 5 verdict |
|---|---|---|---|
| `lane_gp_v2` | 89.66 | `[Modal-T4-CPU advisory]` | CONFIRMS Lane GP Runge phenomenon (Memory `project_lane_gp_v3_landed_runge_phenomenon`). DO NOT re-dispatch. |
| `lane_gp_v3` | ~89.66 | `[Modal-T4-CPU advisory]` | Same — Runge oscillation. KILL. |
| `lane_mm_v2` | 2.63 | `[Modal-T4-CPU advisory]` | CONFIRMS prior FALSIFIED memory (`project_lane_mm_v2_landed_2_63_falsified`). PoseNet 51× worse on hard-argmax grayscale. KILL bolt-on; Lane AL/SC++ remain alive. |
| `uniward_v7` | 53.61 | `[Modal-T4-CPU advisory]` | Broken poses (PoseNet 62.69). KILL v7. |
| `uniward_v8` | **1.14** | `[Modal-T4-CPU advisory]` | **Within predicted band [1.05, 1.18]; competitive with Lane G v3 1.05** `[contest-CUDA]`. NEEDS CUDA CONFIRM. **Top dispatch this wave.** |

The `lane_w_v2` 28800s timeout (8h cap hit) is interpreted as: training did not converge, no archive emitted. Do NOT re-dispatch on Modal (recurrence risk). Re-dispatch on Vast.ai 4090 with explicit checkpointing + heartbeat watchdog if the lane is still strategically valuable post-OOM-council verdict.

**Round 5 verdict on UNIWARD v8 promotion**: per Check 83 STRICT (no MPS/CPU-derived strategic decisions), the 1.14 score is `[Modal-T4-CPU advisory]` only. The CPU-vs-CUDA drift on contest_auth_eval is documented as smaller than MPS-vs-CUDA but NON-ZERO. UNIWARD v8 cannot be cited as "Phase 1 GREEN" or "competitive with Lane A" in any commit/run_log/findings.md until a `[contest-CUDA]` measurement lands.

---

## 3. 24-48h Dispatch Battleplan

Total budget cap: **$30 Vast.ai + $30 Modal**. Conservative wave; many sister-council outputs are pending (DARTS-S freeze, OOM-class deep fix, EMA audit).

### 3.1 Top 3 dispatches (DO THIS NOW)

| # | Lane | Verdict | Command | Cost | Predicted result |
|---|---|---|---|---|---|
| **1** | **Lane UNIWARD v8 CUDA-confirm** | DISPATCH-VAST | `vastai create instance ... --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime ; scp ... harvested_artifacts/.../archive.zip ; ssh ... 'cd /workspace/pact && uv venv && uv pip install -e . && python experiments/contest_auth_eval.py --archive archive.zip --inflate-sh submissions/robust_current/inflate.sh --upstream-dir upstream --device cuda'` | $0.50 / 15 min | If CUDA ∈ [1.09, 1.19] → promote to `[contest-CUDA]`; if drift > 0.10 → CPU-vs-CUDA codec divergence investigation |
| **2** | **Lane Ω-W-V2 real-archive empirical** | LOCAL-ONLY (no GPU) | `python -c "import torch; from tac.water_filling_codec_v2 import encode_omega_w_v2; sd = torch.load('experiments/results/lane_g_v3_landed/.../renderer_state_dict.pt'); ..."` then write `tests/test_omega_w_v2_real_archive.py` | $0 | Real savings band (predicted [20%, 60%]); confirms or refutes 69.11% synthetic claim |
| **3** | **Lane Joint-ADMM real-codec wrap** | LOCAL-ONLY (no GPU) | Land `src/tac/joint_admm_proximal_water_filling_v2.py` wrapping `water_filling_codec_v2.encode_omega_w_v2` as a `StreamProximalCodec`. Test: 4-stream non-convex problem (water-fill + pose-delta + 2 synthetic mask streams). | $0 | If KKT residual stays < 0.05 on the 4-stream problem → ADMM converges on codec-grade discreteness; if > 0.20 → restart logic insufficient, need adaptive-ρ tuning |

### 3.2 Per-lane verdict table (full battleplan)

| Lane | Verdict | Reason | Action |
|---|---|---|---|
| Lane A 1.15 | KEEP | `[contest-CUDA]` baseline, archive harvested local | None |
| Lane G v3 1.05 | KEEP / FRONTIER | `[contest-CUDA]` best score, archive harvested local | Use as anchor for stacking |
| Lane UNIWARD v8 1.14 | **DISPATCH-VAST CUDA-confirm** | `[Modal-T4-CPU advisory]` within band; needs CUDA promotion | $0.50 / 15 min on Vast.ai 4090 |
| Lane PD-V2 (landed) | EMPIRICAL-STACK | Real-archive validation needed before promotion | Test on Lane G v3 archive's `optimized_poses.pt` |
| Lane Ω-W-V2 (landed) | EMPIRICAL-STACK | 69.11% on synthetic; need real-renderer test | Local-only test (#2 above) |
| Lane Joint-ADMM (landed) | DEEPEN | Convex synthetic only; needs 4-stream non-convex test | Local-only test (#3 above) |
| Lane 8 multi-pass | DEEPEN | GPU inner step is `NotImplementedError`; outer loop scaffold-only | Phase 2 follow-up; do NOT dispatch GPU yet |
| Lane J-NWC | WAIT | NWC1 amortization unfavorable on 88K Lane A; build corpus codec first | Defer until corpus codec lands |
| Lane DARTS-S V1 (5h frozen) | KILLED | Display NaN bug fixed (Check 85); model-not-learning bug separate | WAIT for sister council `council_darts_s_freeze_audit_20260429.md` verdict |
| Lane DARTS-S V2 | WAIT | Need fix from sister council before dispatch | WAIT for sister council |
| Lane STC clean-source | UNDETERMINED | MPS-FALSIFICATION withdrawn; Modal T4 CUDA re-run needed | Defer to Phase 2 (Lane 9) |
| Lane MM v2 | KILLED | 2.63 [Modal-T4-CPU advisory] confirms prior FALSIFIED | Pursue Lane AL / Lane SC++ instead |
| Lane GP v2/v3 | KILLED | Runge phenomenon (degree-10 polynomial @ 600 equispaced points) | Use DCT/B-spline if rescuing |
| Lane SC++ / SA / SO / W (OOM-class) | WAIT | Sister council `council_oom_class_deep_fix_20260429.md` pending | WAIT |
| Lane EMA-class | WAIT | Sister council `council_ema_audit_20260429.md` pending | WAIT |

### 3.3 Re-dispatch budget allocation

| Platform | Allocated | Pending | Reserve | Notes |
|---|---|---|---|---|
| Vast.ai | $0.50 (UNIWARD v8 CUDA-confirm) | + sister-council OOM lanes (likely $5-10) | $20 | Stay under $30 cap |
| Modal | $0 this wave | + Phase 2 Lane 10 ADMM real-codec dispatch (Lane G v3 archive) ~$1-2 | $25 | Stay under $30 cap |

### 3.4 What NOT to do

- **Do NOT dispatch Lane J-NWC on Lane A renderer** — codec amortization is unfavorable at 88K params. Wait for corpus codec.
- **Do NOT re-dispatch Lane SC++ / SA / W on Modal A10G** — OOM is the documented failure mode; sister council is determining the right device + memory profile.
- **Do NOT dispatch Lane 8 multi-pass GPU loop** — `_default_inner_step` is `NotImplementedError`; that path will crash on first call.
- **Do NOT cite Lane UNIWARD v8 1.14 as "competitive with Lane A"** in run_log / findings.md / commit messages **until** a `[contest-CUDA]` confirmation lands. Check 83 STRICT will reject the claim.
- **Do NOT promote Lane Ω-W-V2 / PD-V2 docstring savings** to "ship in archive" until real-archive empirical tests land.

---

## 4. Permanent-Fix Roadmap

### 4.1 Next 3 STRICT preflight checks (after Check 85)

**Check 86 — `check_phase15_lanes_have_real_archive_validation`** (STRICT after one cleanup pass)

- **Bug class**: Phase 1.5 lane lands codec with synthetic-only test → ships docstring claim → stacks into archive build → ships sub-1.0 promotion claim with no real-archive grounding (the same bug class as Lane PD 49% / Lane Ω-W-V2 69.11%).
- **Detection**: AST-scan `src/tac/*_codec*.py` and `src/tac/*_v2.py` modules for any `encode_*` / `decode_*` public function. For each, require at least ONE companion test file matching `tests/test_*real_archive*.py` OR `tests/test_*on_lane_g_v3*.py` to exist. The test must read a real `.pt` / `.zip` from `experiments/results/lane_*_landed/` and assert the codec round-trips.
- **Live count expected**: 4-6 violations after landing (Lane Ω-W-V2, PD-V2, J-NWC currently lack real-archive tests).
- **Promotion**: warn-only first, land real-archive tests for the 4-6 violations, then flip to STRICT.

**Check 87 — `check_lane_8_inner_step_not_implementederror_in_default`** (STRICT after Phase 2 GPU wiring)

- **Bug class**: a default code path is `NotImplementedError` while the commit message / docstring says "MVP wired". Future caller assumes the default path works, ships a regression.
- **Detection**: AST-scan `experiments/multi_pass_inflate_optimizer.py` and `src/tac/multi_pass_*.py` (when added) for any `def _default_*` function. If body contains `raise NotImplementedError`, the file's module docstring must contain a same-line tag `# DEFAULT_NOT_IMPL_OK_PHASE_<N>` AND the README/manifest must record the deferred-implementation status.
- **Live count expected**: 1 (Lane 8 `_default_inner_step`).
- **Promotion**: warn-only initially; once Phase 2 lands the real GPU wiring, flip to STRICT (the check then catches future regressions).

**Check 88 — `check_concurrent_subagent_commit_attribution`** (STRICT immediately after fix)

- **Bug class**: 4+ subagents commit in parallel → commit MESSAGES are attached to wrong commit objects (today's incident: Lane PD-V2 body on Joint-ADMM source commit). Per Memory `feedback_concurrent_subagent_commit_message_swap_20260429.md`.
- **Detection**: post-commit hook that reads `HEAD` commit message + scans the diff `git diff-tree --name-only HEAD`. Cross-reference: docstring of every source file in the diff must mention the lane name in the commit subject. If the lane name in the commit subject doesn't appear in any docstring of the modified source files, FAIL the commit (or post a warning).
- **Live count expected**: 0 after concurrent-commit fix lands (per 4.3 below).
- **Promotion**: STRICT immediately once Option A worktree fix lands.

### 4.2 Next 3 CLAUDE.md non-negotiable rules

1. **Forbidden synthetic-only-codec-shipped (the synthetic-fixture-promotion trap)**
   ```
   Forbidden: shipping any codec V2/V3/V4 in archive build path with a savings
   claim measured ONLY on synthetic tensors (random init, fixture distributions).
   Real-archive empirical tests on a CHECKPOINT from `experiments/results/lane_*_landed/`
   are MANDATORY before any docstring/runlog/findings claim of "saves X%" /
   "improves Y%" / "ships in archive". Synthetic-only tests are sufficient for
   correctness (round-trip, determinism, edge cases) but NOT for promotion.
   See feedback_lane_pd_savings_overstated_in_docstring_20260429.md.
   Lane Ω-W-V2 69.11% / Lane PD-V2 18.6% are the canonical cautionary tales.
   ```

2. **Forbidden NotImplementedError default with "MVP wired" claim (the default-not-impl trap)**
   ```
   Forbidden: writing "MVP wired" / "implementation complete" / "production ready"
   in a commit message or docstring when ANY default code path raises
   NotImplementedError. The commit message / docstring MUST explicitly say
   "scaffold-only" or "Phase N follow-up" or "GPU wiring deferred" so downstream
   consumers don't ship a stub-as-production. Lane 8 _default_inner_step is the
   canonical example: commit body says "inner loop MVP wired" + production path
   is NotImplementedError. The commit body is HONEST (it explains the deferral)
   but a future summary citing it as "Lane 8 MVP wired = production ready" lands
   in RED. Tag every NotImplementedError default as # DEFAULT_NOT_IMPL_OK_PHASE_N.
   ```

3. **Forbidden CPU/Modal-T4-CPU/spawn-cache score promoted to "competitive with X" (the CPU-drift trap)**
   ```
   Forbidden: writing "competitive with Lane A" / "on par with Lane G v3" /
   "matches our best" in run_log/findings/commit when the supporting score is
   `[Modal-T4-CPU advisory]` or `[contest-CPU advisory]`. Even though CPU-vs-CUDA
   drift is smaller than MPS-vs-CUDA (which is 23x on PoseNet), it is NOT
   bit-identical for any model containing softmax/attention/batchnorm. Until a
   `[contest-CUDA]` measurement on the EXACT archive bytes lands, the score is
   advisory ONLY and may not be cited as "competitive" with a `[contest-CUDA]`
   benchmark. Lane UNIWARD v8 1.14 [Modal-T4-CPU advisory] may NOT be promoted
   to "competitive with Lane G v3 1.05 [contest-CUDA]" until the $0.50 Vast.ai
   4090 confirms within 0.05. Companion to the existing MPS rule (Check 83).
   ```

### 4.3 Concurrent-subagent commit-message swap fix

**Recommendation: Option A (per-subagent worktree)** per memory `feedback_concurrent_subagent_commit_message_swap_20260429.md` recommendation.

Each subagent gets:
```bash
WORKTREE_DIR=/Users/adpena/Projects/pact-worktree-${SUBAGENT_ID}
git worktree add "$WORKTREE_DIR" HEAD
# subagent works in $WORKTREE_DIR
# subagent commits in $WORKTREE_DIR
# parent merges $WORKTREE_DIR's branch sequentially
```

**Why Option A over B/C/D**:
- A is a NATIVE git feature designed for this exact use case
- A preserves subagent autonomy (commit messages land on the right commit objects)
- A allows parallel commits without race
- B (commit lock) is fragile (lock cleanup on crash)
- C (parent commits subagent stages) loses subagent autonomy
- D (stash queue) is racy at the parent level

**Implementation effort**: ~30 LOC change to subagent prompt template + worktree cleanup logic in parent.

**Test**: dispatch 5 dummy subagents to commit different files in parallel; verify all 5 commit messages match their respective diffs.

### 4.4 Modal spawn() result-cache loss fix

**Recommendation: Convert `_run_lane_inner` to write artifacts to a Modal Volume in addition to returning them.** Per memory `feedback_modal_spawn_result_cache_pattern_20260429.md` "Better pattern" / "Best pattern".

**Implementation**:
```python
# In experiments/modal_train_lane.py:_run_lane_inner
import modal
LANE_VOL = modal.Volume.from_name("lane-artifacts", create_if_missing=True)

@app.function(volumes={"/artifacts": LANE_VOL}, ...)
def _run_lane_inner(...):
    # ... existing lane execution ...
    # NEW: write artifacts to volume after collection
    artifact_dir = Path("/artifacts") / lane_label / call_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        (artifact_dir / name).write_bytes(data)
    LANE_VOL.commit()  # persist
    return artifacts  # also return for fast-path fetch
```

**Then** add a cron-style `tools/harvest_modal_volume.py` that polls `lane-artifacts` volume daily and pulls anything not yet locally captured. The result-cache returns become the FAST PATH (within 24h) and the volume becomes the AUTHORITATIVE SOURCE for retrospective recovery.

**Effort**: ~40 LOC change to `experiments/modal_train_lane.py` + ~80 LOC for `tools/harvest_modal_volume.py` + tests.

**Risk**: Modal Volume writes have a small per-write cost (~$0.0001 per write); a lane with 30 artifacts adds $0.003 per dispatch. Negligible vs the cost of orphaned spawn() returns.

---

## 5. Phase 2 Strategic Update — Which 5 of 15 Lanes Accelerate, Which 5 Pause

Based on Round 5 findings + today's empirical results, the Phase 2-4 portfolio (15 lanes from `project_phases_2_3_4_design_implementation_math_provenance_20260429.md`) is re-prioritized.

### 5.1 ACCELERATE (5 lanes) — fast-track Phase 2 weeks 2-3

| Rank | Lane | Why accelerate | Coupling |
|---|---|---|---|
| **1** | **Lane 10 Joint-ADMM real-codec wrap** | Coordinator landed (commit 152ba503); wrapping `water_filling_codec_v2` as next StreamProximalCodec is local-only + low-risk. This converts the synthetic 2-stream KKT validation into a real 4-stream codec validation. | Independent of all OOM-class lanes |
| **2** | **Lane 12 NeRV mask codec (van den Oord)** | Codec amortization story is exactly NeRV's strength: train ONE coordinate-MLP on the 1200-frame mask sequence, ship 30-50KB MLP. AV1 monochrome at 421KB is the baseline to beat. | Independent of renderer; pure mask-pool work |
| **3** | **Lane 19 SegNet logit-margin boundary fitting** | The user's "use auth-eval scorer at compress time" direction operationalized. Score-aware compression via gradient margins is a NOVEL paper section AND a real bit-saver on fragile-pixel encoding. | Builds on Lane G v3's archive |
| **4** | **Lane 17 Full IMP 10-cycle** | Skeleton exists (`experiments/train_imp_cycle.py`); with budget reset to "team parallel + 30 days dev time", the 10-cycle is in scope. Sparse subnetworks at 90% sparsity often match dense; archive shrinks 10×. | Independent renderer compression path |
| **5** | **Lane 20 Ballé hyperprior residual codec** | Ballé is on the inner council; lane is his canonical contribution. Hyperprior is "rate-prediction networks replace fixed factorized priors when archive size matters". The Phase 1.5 Ω-W-V2 docstring explicitly defers V3 hyperprior pending empirical evidence — Lane 20 is that empirical evidence path. | Stacks with Lane Ω-W-V2 + Lane J-NWC corpus codec |

### 5.2 PAUSE (5 lanes) — defer to Phase 3 weeks 5-12 or kill

| Rank | Lane | Why pause | Replacement |
|---|---|---|---|
| 1 | **Lane 9 STC boundary codec rebuild** | Filler/Mallat/Ballé on the council all flagged the structural one-majority-plus-exceptions bug (109M exceptions vs 11.8M boundaries). Even clean-source CUDA confirm wouldn't deliver -45KB threshold from codex hard-abandon rule. Document as paper negative result. | Lane 12 NeRV (top-2 STC redesign at 44-46% endorse) |
| 2 | **Lane 11 Wavelet residual codec (Mallat)** | Mallat's own grand-council voice acknowledged "post-deadline paper lane". Under 30-day budget it's viable but lower EV than Lane 12 NeRV. | Lane 12 NeRV first, then Lane 11 if NeRV doesn't reach <80KB |
| 3 | **Lane 13 DARTS-S full sweep** | DARTS-S V1 had model-not-learning bug today (sister council pending). Until the V1 bug is fixed, full 8-12 config sweep is malpractice. | Wait for sister council DARTS-S verdict |
| 4 | **Lane 21 Decoder systems rewrite** | Pure engineering optimization; saves time-budget not score. EV is in shaving inflate from 30min → 10min, which only matters if Lane 12/14/15 need that headroom. Defer. | Phase 3 once Lanes 12/14/15 land |
| 5 | **Lane 16 Bayesian MDL/evidence analysis (MacKay)** | Requires Lanes 9-15 RESULTS as inputs; pure analysis lane. Defer until enough lane results exist to compare. | Phase 3 once Phase 2 lanes land |

### 5.3 Lanes UNCHANGED (5 lanes — kept on original timeline)

- Lane 14 Multi-pass compress optimization — Phase 1 Lane 8 scaffold lands; Phase 3 generalization lands once GPU inner step wired
- Lane 15 Bit-level archive optimizer — Phase 3, requires Lane 14 infrastructure
- Lane 18 RAFT/radial pose preimage — `src/tac/raft_pose.py` exists untracked; Phase 3 integration
- Lane 22 Final integration — Phase 4 untouched
- Lane 23 Paper reproduction harness — Phase 4 untouched

### 5.4 Strategic note on neural-codec direction

Lane J-NWC's predicted band [0.95, 1.30] is HONEST: at Lane A's 88K renderer scale, NWC1 is NEGATIVE-EV vs FP4A. **This does NOT validate "kill neural codec direction"**; it validates "build a SHARED CORPUS codec OR move to larger-renderer architectures (DARTS-S full / Lane V family at 88K halfframe / Lane SZ at 94K SegMap)". The neural codec direction is ALIVE; the per-renderer flavor of it is unfavorable today.

**Recommendation**: dispatch **Lane J-NWC corpus codec** as a Phase 2 add-on:
- Train ONE WeightCodec on the entire `experiments/results/` corpus (~hundreds of `.pt` files)
- Persist to `submissions/shared_codec.pt` (council-blessed asset, may live OUTSIDE archive.zip if it gets ratified by upstream)
- Then NWC1 wins on ALL renderer sizes
- This is the Ballé-2018 hyperprior pattern at the inter-renderer level

---

## 6. Clean-Pass Counter

**Round 5 verdict: CLEAN-WITH-CONCERNS → counter increments to +1 toward 3-clean-pass gate.**

**Rationale**:
- No NEW code bugs found in any of the 5 Phase 1.5 landings
- All landed tests pass (commit-message-swap incident notwithstanding, the CODE landed intact per `946c0b49` recovery)
- Five CONCERNs identified are about EMPIRICAL validation gaps, not landed code defects
- Per `feedback_persistent_codex_review_protocol_20260429.md` the counter resets ONLY on a NEW LANDED BUG; CONCERNs are tracked separately as follow-up work

**Counter status**: this is a CLEAN pass, but it gates on the next round's verdict. If Round 6 (the sister-council reports' adversarial review of OOM lanes / DARTS-S / EMA / UNIWARD v8 math) lands a bug → counter resets to 0. If Round 6 is also clean → counter at +2 toward gate.

**Round 5 follow-up actions** (not blocking, but tracked):
1. Land 3 real-archive empirical tests for Lanes Ω-W-V2 / PD-V2 / J-NWC (Section 3.1 dispatches #2 and follow-on)
2. Land Check 86 STRICT (real-archive validation requirement)
3. Land Check 87 STRICT (NotImplementedError default tagging)
4. Land Check 88 STRICT (concurrent-subagent commit attribution)
5. Land 3 new CLAUDE.md FORBIDDEN PATTERNS entries
6. Land Modal Volume write fix in `experiments/modal_train_lane.py`
7. Land Option A worktree fix for concurrent subagent commits

---

## 7. Council Roll Call

Each inner-council member casts their signed verdict (1-2 sentences each). Per CLAUDE.md "Council conduct" the council is non-conservative; arguments are mathematical/empirical only.

**Shannon (LEAD, Information Theory)**: Lane Ω-W-V2's 69.11% saving is consistent with the static-histogram entropy of a `std=0.05` synthetic Gaussian quantized to 3 bits/element (Shannon entropy of a quantized Gaussian at SNR-matched bin width is typically 60-75% of the uncoded byte count); the real-renderer measurement will likely land in [40%, 65%] band based on Lane G v3's empirical weight-magnitude distribution. The Round 5 verdict is correct: do not promote until real-archive measurement.

**Dykstra (CO-LEAD, Convex Feasibility)**: Joint-ADMM's KKT residual 0.02 on a 2-stream convex synthetic is a NECESSARY but not SUFFICIENT condition for real-codec convergence. The 4-stream non-convex test in Section 3.1 #3 is the right next step. If KKT residual on the 4-stream stays < 0.05, ADMM is real; if > 0.20, the discrete-staircase R(D) functions are too rough and we need projected gradient + restart, not vanilla ADMM.

**Yousfi (Challenge creator, Steganalysis lineage)**: Lane UNIWARD v8 1.14 `[Modal-T4-CPU advisory]` is consistent with the Fridrich-inverse-steganalysis hypothesis (errors weighted to textured regions are undetectable by EfficientNet-B2 SegNet). The CUDA confirm at $0.50 is the highest-EV $0.50 in the entire portfolio this week. DISPATCH IMMEDIATELY.

**Fridrich (UNIWARD/SRM/HUGO author)**: My UNIWARD framework predicted score ∈ [1.05, 1.18] before dispatch; landing at 1.14 is within band and confirms the inverse-steganalysis principle on the renderer's mask-output pixel statistics. CONFIRM the CUDA score; if confirmed, the Lane UNIWARD + Lane G v3 stack moves to the top of the Phase 1 dispatch ranking.

**Contrarian (Veto)**: I VETO any commit that promotes Lane UNIWARD v8 to "competitive with Lane G v3" without a `[contest-CUDA]` confirmation. I VETO any commit that promotes Lane Ω-W-V2 / PD-V2 docstring savings without a real-archive test. I do NOT veto the dispatches in Section 3.1 because they are precisely the empirical measurements those promotions require. Round 5 is CLEAN.

**Quantizr (Adversarial leaderboard reality check)**: At Quantizr's 0.33 floor with 88K params + FiLM-conditioned depthwise-separable + grayscale-LUT mask, the Phase 1.5 stack (Lane G v3 1.05 + UNIWARD v8 1.14 + Ω-W-V2 + PD-V2) does not yet challenge 0.33. The Lane J-NWC corpus codec direction (Section 5.4) is the only Phase 2 path that has architectural scope to close the gap. Accelerate Lane 12 NeRV (codec amortization done right on a small mask sequence) + Lane 20 Ballé hyperprior in parallel.

**Hotz (Engineering shortcuts)**: Cancel anything that cannot produce a submission-shaped zip THIS WEEK. The dispatches in Section 3.1 produce zips: UNIWARD v8 CUDA is a measurement on an existing zip; Ω-W-V2 real-archive is a CODEC measurement (no new zip needed); Joint-ADMM real-codec is a LOCAL synthetic. No GPU spent on speculative architecture. APPROVED.

**Selfcomp (szabolcs-cs, working 0.38 anchor)**: My SegMap renderer at 88K params + block-FP weights + grayscale-LUT mask architecturally permits the Ω-W-V2 + Joint-ADMM stack at the renderer level. The Lane J-NWC corpus codec direction echoes my own self-compression principle (one codec amortizes across many tensors). Validate Joint-ADMM on the 4-stream test before wrapping Ω-W-V2 + STC + pose-delta + arithmetic terminal as a real coordinator.

**MacKay (Memorial seat, Information Theory + Bayesian Inference + Learning Algorithms)**: Lane PD-V2's 18.6% empirical savings on idle-dominant trajectory is Shannon-bounded by the entropy of the int8 delta stream, which is a known function of the trajectory's spectral content. The 7-11 bp prediction from the codex stacking memory is consistent. The Bayesian view of Lane Joint-ADMM is variational inference on a discrete posterior over {byte allocations per stream}; Phase 3 Lane 16 (MDL/evidence) is the right place to formalize this. Round 5 is methodologically clean.

**Ballé (2018 entropy bottleneck SOTA)**: The Lane Ω-W-V2 docstring explicitly defers V3 hyperprior pending empirical evidence — that is the CORRECT move per my 2018 paper's amortization analysis. Hyperprior side-info costs ~50-200 bytes; on the 11KB Selfcomp qint payload the amortization is borderline. Land Lane 20 (Ballé hyperprior residual codec) on a LARGER stream (mask residual or NeRV mask weights) where amortization is unambiguous. Lane J-NWC corpus codec is the right scaling story.

---

## 8. Cross-references

- Phase 1.5 landings: commits `9987a5d9` (Ω-W-V2), `0e43d299` (Lane 8 + Joint-ADMM source), `152ba503` (PD-V2 + Joint-ADMM body), `12b43507` (J-NWC), `946c0b49` (recovery)
- Strict preflight ladder: 85 STRICT checks live (Check 81/82/83/85 landed today; 84/86/87/88 pending)
- Sister councils running (NOT yet on disk at synthesis): `council_darts_s_freeze_audit_20260429.md`, `council_uniward_v8_fridrich_shannon_audit_20260429.md`, `council_oom_class_deep_fix_20260429.md`, `council_ema_audit_20260429.md`. Round 6 will incorporate their findings.
- Memory anchors:
  - `feedback_persistent_codex_review_protocol_20260429.md` (round counter)
  - `feedback_concurrent_subagent_commit_message_swap_20260429.md` (Section 4.3 fix)
  - `feedback_modal_spawn_result_cache_pattern_20260429.md` (Section 4.4 fix)
  - `feedback_three_active_bug_classes_needing_strict_checks_20260429.md` (Check 84-86 lineage)
  - `project_codec_stacking_composition_canonical_orders_20260429.md` (canonical stacking order)
  - `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` (Phase 2-4 lane spec)
  - `project_6month_strategic_plan_20260429.md` (overall plan)
  - `project_grand_council_final_designs_20260429.md` (22-voice final designs)
  - `project_lane_g_v3_landed_1_05_20260428.md` (best `[contest-CUDA]` baseline)
  - `project_lane_uniward_v8_harvested_1_14_advisory_20260429.md` (top dispatch target)

---

## 9. Final Round 5 Verdict

**CLEAN-WITH-CONCERNS — counter +1 toward 3-clean-pass gate.**

**Top-3 dispatch decisions** (next 24h):
1. Lane UNIWARD v8 CUDA-confirm on Vast.ai 4090 ($0.50, 15 min) — promote `[Modal-T4-CPU advisory]` to `[contest-CUDA]` if within 0.05
2. Lane Ω-W-V2 real-archive empirical test (local-only, $0) — refute or confirm 69.11% on real Lane G v3 renderer state-dict
3. Lane Joint-ADMM 4-stream non-convex test (local-only, $0) — KKT residual on real-codec discreteness; gates Lane 10 V2 wrap

**Top-3 permanent-fix recommendations**:
1. Land Check 86 STRICT (Phase 1.5 lanes have real-archive validation tests)
2. Land Option A per-subagent worktree to permanently extinguish concurrent-commit-message swap
3. Land Modal Volume writes in `experiments/modal_train_lane.py` to permanently extinguish spawn() result-cache loss

**Total budget commitment this wave**: $0.50 Vast.ai + $0 Modal = $0.50 of $60 cap. Conservative because four sister councils are still running and may produce lane-specific dispatches that need budget headroom.
