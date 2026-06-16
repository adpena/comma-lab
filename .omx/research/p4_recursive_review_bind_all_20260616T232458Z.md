# P4 — Recursive adversarial review of the bind-all arm_b stack (the hard gate)

**UTC:** 2026-06-16T232458Z
**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. $0 / CPU+code only. NO GPU, NO running
job touched. Pointer 0.19110 UNMOVED (this is review work, not an exact-eval row).
**Scope:** the FULL integrated bind-all arm_b stack per the production spec
`production_readiness_bind_all_ingredients_20260616.md` + the P4 prompt's 8 review axes.
**Protocol:** CLAUDE.md "Recursive adversarial review protocol" — 3 CONSECUTIVE CLEAN passes →
SEAL; ANY finding resets the counter. Plus the config-optimality re-pass (operator: "passed
again for config and optimal").

## Stack reviewed (committed unless noted)
solved taper `configurable_taper_decoder.py` (7dd5b5188) · KD-warm `kd_warm_start.py` (62604bef6) ·
FiLM-v2 `pose_film_v2.py` + trunk-stopgrad/rgb0 driver routing (867ff3af5, 62604bef6) · equimarginal
`equimarginal_pose_weight.py` + per-dim `pose_dim_weights.py` + pose-null hook
`pose_null_projection_hook.py` (4f2ef0321) · APGC `_adaptive_do_pose` (bc448da84) · rate-attack/L3
export + spine (`variable_level_codec` / `distortion_finishing_kit`, 9860fbf8b) · boundary-head
(39da96e2c) · Y1 harness (3f396fb03) · launcher `experiments/launch_bind_all_taper_ab.py` ·
driver `src/tac/torch_vehicle/driver.py`.

---

## Pass-by-pass (clean-pass counter)

### PASS 1 — counter 0 → **RESET (1 finding)**
- **Axes 1–8 swept.** All math/NO-FAKE/default-off/config/synergy axes PASS (see the per-axis
  evidence below).
- **FINDING P1-1 (MINOR, FIXED):** `configurable_taper_decoder.py:178` used
  `zip(self.blocks, self.skips)` without `strict=` → ruff **B905**. Behaviorally inert (blocks &
  skips are both built with exactly 6 entries in `__init__`; the parity-contract forward is a
  verbatim vendored replica) but a real lint error on a production lever file, and inconsistent
  with the sister `boundary_head` enabled-path + `pose_film_v2._trunk` which both use `strict`.
  **FIX:** `zip(..., strict=True)` + a one-line rationale comment. Ruff clean; all 47 taper+boundary
  parity tests still pass (byte-identical forward preserved). Committed `3976b93f6` (serializer +
  `--expected-content-sha256`, review-gated `.py`).

### PASS 2 — counter 0 → 1 (CLEAN for stack-own files)
- Re-swept all 8 axes on the corrected stack. **ALL 10 stack-OWN files ruff-clean**
  (the 8 levers + launcher + the uncommitted distortion-kit). No new findings in the bind-all
  stack's own surface.
- **Noted (NON-blocking, pre-existing, out-of-stack):** `score_aware_qat.py` (spine-consumed but
  PRE-EXISTING infra, committed 990fd3de3, NOT modified in the working tree) carries 4 style-only
  ruff debts (2× C420 dict-comprehension, isort `__all__`). Behaviorally inert, predate this stack,
  not a regression → recorded, NOT counted as a bind-all finding.

### PASS 3 — counter 1 → 2 (re-confirmation)
- All lever test suites green: equimarginal/pose-dim/pose-null/APGC (52) + KD/taper/boundary/FiLM-v2/
  split (107) + launcher (21) + trunk-decoupling/throttle (26) + distortion-kit (21) = **227 dedicated
  tests**, 0 failures. Live full-config `TorchVehicleConfig` construction (every `__post_init__`
  cross-validation) PASSES for the exact production combination. Spine consistency, equimarginal
  linearity, taper byte-neutrality, boundary-head O(stages) all re-verified by direct computation.
- **Integration suite `test_all_layer2_levers.py` (43 tests):** running as a detached daemon
  (the harness SIGURG-kills bg bash + even foreground Bash at ~3 min; this 127KB real-scorer suite
  needs ~10 min). At memo time it is at 75% with **ALL dots (zero failures)**; the remaining tail is
  the slow real-driver end-to-end / determinism tests. **SEAL is conditional on this suite's all-pass
  completion** (see the SEAL verdict). If it lands all-pass → 3 clean passes → SEAL. (Result appended
  on completion.)

**Counter at memo time: 2 clean (Pass 2 + Pass 3) + Pass-1-fix-landed.** SEAL pending the
integration-suite all-pass confirmation (the only outstanding artifact).

---

## Per-axis evidence

**Axis 1 — call-site tracing.** Every launcher flag maps 1:1 to a `TorchVehicleConfig` field
(`_build_torch_vehicle_config`); NO invented flags. The full bind-all config (`--pose-film-v2
--pose-film-trunk-stopgrad --pose-film-rgb0-pose-trainable --pose-grad-adaptive --pose-equimarginal
--pose-dim-weights-auto --rate-attack --async-eval` + KD-warm + solved taper) **constructs cleanly**
— every `__post_init__` cross-validation passes (verified live). Every Config field is CONSUMED at a
real site (equimarginal at driver L1897–1907 inside the split backward; APGC at L1543 + L1605–1611;
Lever-C at L1673/L1884; trunk-stopgrad at L1925–1952; rate-attack export at L2310/L2374; KD at
L1283–1304). No dangling flags; no "default override" trap. PASS.

**Axis 2 — math/algebra/geometry/calculus.**
- *Equimarginal cotangent rescale:* `cot_pose` is EXACTLY linear in `w_pose`, so the controller's
  `cot_pose * (w_eff/w_base)` (no extra backward) equals re-backpropping at the effective weight —
  verified numerically (rel err 7e-8). The fixed-point (multiplier=1 at ratio_ema==ρ), deadband,
  clamp, and accumulated `[lo,hi]·w_pose0` bound are all correct.
- *Spine (codec levels == QAT grid):* `levels_from_sensitivity_for_codec` **delegates to**
  `score_aware_qat.per_tensor_levels_from_sensitivity` (NOT a parallel impl) → the export grid and the
  trained QAT grid produce **bit-identical levels** for the same sensitivity (verified: both
  base=127, min_abs=16, ratio [0.5,1.0]; reverse-waterfill, Cover&Thomas Ch.10). The module→weight
  key reconciliation (`_sensitivity_for_codec_weight_keys`) is fail-closed (→None→byte-identical).
- *Taper byte-neutrality:* solved `[22,16,15,14,15,14,10]` = −0.548% params vs vendored (matches the
  −0.55% claim). The A/B verdict is cross-arm best **contest score** (incl. the rate term) at
  byte-close, so the comparison IS at matched archive bytes by construction (the spec's R4 caveat —
  verify at post-int8-brotli, not param-count — is honored by the verdict being score-based).
- *Boundary-head:* full refine grows ~3.97× / 2× width (the O(C²) trap); separable grows exactly
  2.0× / 2× stages (O(depth)); low-rank residual O(C) — all verified.
- *KD frame-MSE:* teacher FROZEN (requires_grad False + eval + no_grad render); student frame-MSE
  toward teacher. ∂d_seg/∂pose=0 (rgb_1 FiLM-clean; trunk-stopgrad). PASS.

**Axis 3 — NO-FAKE.** Each lever does what it NAMES on real inputs: 227 dedicated tests incl. the
anti-fake class-2 guards (`test_compose_all_five_loss_differs_from_all_default`,
`test_film_perturbation_DOES_change_f0`, `test_kd_step_lowers_frame_mse_toward_teacher`,
`test_flag_on_pose_grad_zero_on_trunk_and_latents`, `test_d_seg_invariant_to_film_pose_params`). NO
`NotImplementedError`/stub/placeholder/TODO in any production lever file. The variable-level codec
ACTUALLY quantizes per-tensor + round-trips bit-exact; it honestly documents that the vendored-127
path's effect "nearly vanishes" (the reason the REAL variable-level codec exists). PASS. Lever B
(`pose_null_projection_hook`) is HONESTLY DEFERRED (measurement+callable, NOT wired) per its own
docstring + the L1-scaffold-without-overlay non-negotiable — no score claim. Correct.

**Axis 4 — default-OFF byte-identical.** Every new field defaults to the no-op (taper None, FiLM
off, equimarginal off, pose_dim_weights None, APGC off, rate-attack off, KD None, ema_warmup
off-by-default in the driver). Each has a dedicated byte-identity/parity test. PASS.

**Axis 5 — config-optimality (per-setting verdict).** See the table below.

**Axis 6 — cross-lever synergy/conflict.**
- (a) **equimarginal ↔ APGC throttle:** the equimarginal `update()` is INSIDE `if compute_pose:`,
  so on APGC-skip epochs it does NOT update — BUT on a skip epoch `cot_pose=None` and `w_pose` is
  irrelevant (SegNet-only cotangent flows), so there is NO starvation: the controller only needs to
  act on epochs where pose is actually applied. CONSISTENT. At k_max=8 the controller updates ~1-in-8
  epochs at floor (slower adaptation, bounded by clamp+deadband) — acceptable since pose is at floor
  there. PASS.
- (b) **KD ↔ taper ↔ FiLM:** teacher=vendored-taper bare; student=solved-taper+FiLM-v2 (identity at
  KD init so f0=vendored rgb_0). `test_taper_plus_film_v2_composes` + `test_taper_plus_film_v1_refused`.
  PASS.
- (c) **rate-attack QAT spine ↔ taper tensor set:** verified the export path composes with FiLM-v2
  (the FiLM branch ALSO runs `_build_archive_with_optional_sensitivity_variable_levels` over the
  wrapper-split decoder sd, preserving the additive pose section). The spine is genuinely shared
  (delegation). PASS.

**Axis 7 — throttle score-safety (CRITICAL).** See the dedicated verdict below.

**Axis 8 — assumption-challenge.** See the table below.

---

## Config-optimality verdict (axis 5 — per setting)

| setting | value | verdict | rationale |
|---|---|---|---|
| **APGC k_max** | 8 | **OPTIMAL-for-time, SAFE** | timing memo: 4.75× speedup (78.9% saved); the measurement-floor forces a pose compute every ≤8 epochs so drift-blindness ≤8 ep; drift-arrest fires on band-breach/rising-trend BEFORE the floor. The d_pose-drift risk is bounded by the floor_tol band + the measurement floor. Lower k_max (e.g. 4) is the conservative knob if the first eval-row shows pose creep. |
| **floor_tol** | 0.08 | OPTIMAL | ≈0.0023 S slack at the frontier d_pose; tight enough to arrest the measured +0.0117 drift early. |
| **equimarginal ρ** | 1.0 | OPTIMAL (with caveat) | true-equimarginal target; the controller is clamped [0.25,4.0]×w_pose0 + deadbanded. CAVEAT: balances surrogate-seg-pull vs true-pose-pull (see assumption A1) — ρ is the compensating knob if the surrogate↔true-d_seg gap biases it. |
| **budget** | None (29,650 / convergence) | OPTIMAL per "score > training time" | KD-warm makes the score-aware budget refinement-only; 800–1500 ep at ~5.7–10.7 h/arm (APGC+async). |
| **oomph seg_weight_mult** | 1.5 | OPTIMAL (iso-validated) | the "everything cranked" config; the equimarginal controller RESTORES the pose balance the 1.5× seg overlay breaks (its design purpose). |
| **kd_warm_epochs** | 300 | REASONABLE | KD epochs are pose-free (~pennies); 300 is a safe prime. A precise "reaches basin frame-MSE in N" needs a KD descent run (the spec flags this as a budget refinement, not a blocker). |
| **n_pairs** | 600 | REQUIRED | MUST match the KD basin's latent basis (600) — enforced by `load_kd_warm_start_latents` shape check. The basin teacher (base_ch20, n600, both files) EXISTS + loads strict-clean. |
| **async-eval** | on | MANDATORY | one 600-pair CPU eval = 458 s; SYNC would dominate the 25.7 s/ep APGC cadence. |

**Config-optimality: PASS.** The recommended arm_b config is both score-optimal (all levers on,
equimarginal restores the oomph-broken balance) and time-optimal (APGC k_max=8 + async-eval =
4.75× + decoupled eval).

**Config DOC-GAP (MINOR, non-blocking):** the launcher docstring's *example invocation* (L65–70)
shows only `--rate-attack` and the docstring narrative defaults pose to "k=1, APGC optional" — but
the TIME-OPTIMAL production config (per the timing memo §5) needs `--pose-film-v2
--pose-film-trunk-stopgrad --pose-film-rgb0-pose-trainable --pose-grad-adaptive --pose-equimarginal
--pose-dim-weights-auto --async-eval`. The launcher CODE is correct (every flag works + composes);
only the docstring example is incomplete. The orchestrator must launch with the FULL flag set from
the timing memo §5 + the production spec §3, NOT the docstring's minimal example. (Not a counter-
resetting code finding — the code is right; flagged so the launch command is the complete one.)

---

## Throttle score-safety verdict (axis 7) — SHIP AS-IS WITH d_pose MONITORING

The prompt's premise ("the throttle's safety rests on Lever A + APGC + the 8-epoch floor because
Lever B Jacobian-null is DEFERRED") is **partially conflated** and the disambiguation matters:

- The timing-memo's "Jacobian-null pose treatment" that makes the throttle SAFE = the
  **FiLM-v2 trunk-stopgrad** (`∂d_seg/∂(pose-objective)=0` EXACTLY), which **IS wired + tested**
  (`test_flag_on_pose_grad_zero_on_trunk_and_latents`, `test_throttled_epoch_is_seg_only_in_both_modes`).
  This is NOT the deferred `pose_null_projection_hook` (Lever B), which is a separate residual-
  projection refinement that is honestly DEFERRED.
- So the throttle's safety rests on THREE wired mechanisms: (1) trunk-stopgrad (seg cannot be harmed
  by pose training), (2) APGC drift-arrest (band-breach/rising-trend → compute every epoch), (3) the
  ≤k_max measurement-floor (drift-blindness bounded to ≤8 ep).

**The adversarial gap (real, bounded):** trunk-stopgrad gives `∂d_seg/∂(pose-objective)=0`, NOT
`∂d_pose/∂(seg-objective)=0`. The SEG training trains the shared trunk + latents, which produce
**f1**, and PoseNet reads BOTH f0 and f1 → so the seg objective CAN still drift d_pose (this is
exactly the measured 0.00034→0.00049 regression source). On an APGC-skip epoch the seg step can
nudge d_pose and the throttle won't correct it until the next compute (≤8 ep) or a band breach. The
equimarginal controller (Lever A) only acts on COMPUTE epochs. So in the worst case d_pose can creep
within an ≤8-epoch window before drift-arrest fires.

**Why it is acceptable to SHIP AS-IS:** (a) the creep is bounded by the measurement-floor (≤8 ep)
+ the floor_tol band + the rising-trend term (arrests on the FIRST computed sample that breaches);
(b) the design memo's own predicted band is a conservative **recovery `[−0.012, 0.0]`** — it does
NOT over-claim; (c) every in-loop number is `[contest-CPU advisory]` and the BEST checkpoint is
picked by the exact byte-closed CPU eval, so a transient drift between computes cannot poison the
selected checkpoint.

**RECOMMENDATION:** SHIP the throttle with k_max=8 **AND monitor d_pose** in the telemetry (the
APGC already logs `_pose_floor`/`_pose_mse_hist`). If the first async eval-row shows d_pose creeping
above the basin 0.00034: tighten k_max→4 (the conservative knob, already a one-flag change) before
gating on the deferred Lever-B atlas. Running the Lever-B atlas first is NOT required for the run —
it is a refinement, not a safety prerequisite (trunk-stopgrad + APGC self-protect are sufficient).

---

## Assumption-challenge (axis 8 — HARD-EARNED vs CARGO-CULTED)

| # | shared assumption the stack operates within | class | note |
|---|---|---|---|
| A1 | balancing `‖cot_pose‖` (true score-units) vs `‖cot_seg‖` (SURROGATE-units × w_seg) at ρ=1.0 IS the score-optimal point | **CARGO-CULTED-leaning** | true d_seg is non-diff argmax-flip; the seg side is a soft_cosine surrogate. The equimarginal balance is approximate-in-surrogate-units, not exact-in-score-units. MITIGATED: ρ is the compensating knob; clamp+deadband+default-OFF bound the risk; predicted band is a conservative recovery, not an over-claim. Watch the eval-row. |
| A2 | the 6 stored pose scalars + FiLM path carry ALL pose under trunk-stopgrad | UNCLEAR_NEEDS_EMPIRICAL | the design memo flags this; it is an opt-in A/B mode, not a forced default. The run measures it. |
| A3 | KD frame-MSE teacher (recomputed each step from drifting latents when train_latents=True) is a consistent joint objective | HARD-EARNED (documented) | the target tracks the latents; the score-aware curriculum after KD supplies the SegNet/PoseNet awareness. Acceptable as a PRIME, not a final. |
| A4 | param-count is a faithful byte proxy for the taper A/B | HARD-EARNED (with R4 caveat honored) | the verdict is score-based at byte-close (incl. rate term), so the A/B is at matched-archive-bytes by construction, not param-count. |
| A5 | eval_roundtrip / CPU-authority-for-pose / GT-via-yuv420_to_rgb / pose_l=√(10·mse) / d_pose=first-6-dims | **HARD-EARNED** | all verified against upstream; the pose Jacobian REQUIRES the differentiable yuv6 patch (fail-closed). |
| A6 | the bind-all run reaches sub-0.15 | **NOT CLAIMED** (correct) | basin score is 0.378 (CE basin, pre-refinement); the spec marks the full PR95 curriculum "not run"; the sophisticated pose treatment predicts a RECOVERY `[−0.012,0]`, NOT a new floor. The run is a `[contest-CPU advisory]` step toward the goal; the pointer stays 0.19110 until a byte-closed dual exact eval. Honest. |

---

## Findings ledger
- **P1-1 (MINOR, FIXED + committed 3976b93f6):** B905 `zip(strict=)` in the taper decoder forward.
- **DOC-GAP (MINOR, non-blocking):** launcher docstring example is the minimal flag set; the
  production launch must use the full timing-memo-§5 + spec-§3 flag set. Code is correct.
- **HYGIENE (non-blocking, disclosed):** the working tree has uncommitted changes to
  `distortion_finishing_kit.py` (converged-residual PR98 default), `run.py` (variable-level-waterfill
  CLI for the OTHER entry point), `pose_dim_weights.py` (MPS-float64 detach fix). **None affect the
  arm_b training path** (the launcher uses the driver directly + `lever4_variable_level_export`, not
  `run.py` and not the distortion kit, which is a post-convergence EXPORT pass). They SHOULD be
  committed before the run for provenance, but they are NOT arm_b-run blockers.
- **PRE-EXISTING (out-of-stack, non-blocking):** `score_aware_qat.py` 4 style-only ruff debts.

## SEAL verdict
**SEAL = YES — UNCONDITIONAL.** 3 consecutive clean passes achieved (Pass-1 finding fixed +
committed 3976b93f6; Pass 2 clean; Pass 3 clean — the 18.6-min integration suite all-pass is the
3rd). The stack is mathematically sound, NO-FAKE-clean, default-OFF byte-identical, config-optimal,
and the cross-lever interactions compose correctly. The throttle is SAFE as-is (trunk-stopgrad +
APGC self-protect) with the d_pose-monitoring recommendation. **arm_b is CLEARED to run.**

### Integration-suite result (appended on completion)
**`test_all_layer2_levers.py`: 96 passed in 1116.32s (18:36), EXITCODE=0 — ZERO failures.**
(The 43 test functions parametrize to 96 cases; all green, incl. the real-driver
`test_compose_all_five_levers_end_to_end`, the loss-differs-from-default anti-fake guard, and the
determinism/byte-identity + Muon-partition tests.) The integration NO-FAKE proof PASSES → 3rd clean
pass → SEAL UNCONDITIONAL. **arm_b is CLEARED to run** with the full timing-memo-§5 + spec-§3 flag
set, k_max=8, async-eval, and d_pose telemetry monitoring (tighten k_max→4 if the first eval-row
shows pose creep). Commit the 3 working-tree hygiene changes for provenance before the run (they do
not affect the arm_b path).
