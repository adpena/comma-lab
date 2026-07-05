---
council_tier: T3
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary,
                    StEik-Yang, ViscoReg-author, Cohen-EoS, Damian-self-stabilization, DE-DERIVATION-318]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: true
council_override_rationale: "Operator 2026-07-05 verbatim: 'Seems like naive and toy' (the n24
  cure-arbitration) + the decision that the v6 config rests on TWO real things only — the DE
  first-principles derivation (#318) + ONE bounded n600 measurement — NOT an n24 crown. Operator
  granted authority to execute the bounded n600 probe on the finalized config. This overrides the
  arbitration-crowns-the-winner plan (n24 is disproven as a proxy for THIS instability)."
council_dissent:
  - member: Contrarian
    verbatim: "The n24 proxy is DISPROVEN for this instability, not merely noisy: the visco_03_anneal
      CONTROL (identical to v5) ran to ep215 STABLE at n24 while that exact config re-entered and
      deadlocked at n600 ~ep110. n24 stability is NECESSARY-NOT-SUFFICIENT. No v6 config may be crowned
      on n24 evidence; the ONLY real evidence is the bounded n600 measurement + the DE derivation."
  - member: Assumption-Adversary
    verbatim: "steik_norm_05 (normalized n^T H n at W=0.5) EXPLODED at n24 (total 106->249, eikonal
      53->203). At W=0.5 the anisotropic normal-damper is a measured NO-GO; combined with the DE
      symbol (ill-posed mode is TANGENTIAL where |grad m|<1) this de-prioritizes candidate (a). The v6
      config must NOT ride n^T H n; it rides the DE-highest-confidence config-lever (flat lambda_eik)."
council_assumption_adversary_verdict:
  - assumption: "the n24 cure-arbitration can crown a measured-optimal v6 config"
    classification: CARGO-CULTED
    rationale: "DISPROVEN (operator: 'naive and toy'): the n24 slice does not reproduce the n600
      sharpening (|c_a(t)| growth is landscape/pair-count dependent, DE-predicted) that triggers the
      re-entry — the v5 config is n24-stable and n600-fatal. n24 arms are a NEGATIVE filter only
      (a config that explodes at n24 IS bad, e.g. n^T H n W=0.5); they cannot certify a cure. The v6
      config rests on the DE derivation + one bounded n600 measurement."
  - assumption: "the normalized n^T H n (anisotropic normal-direction damping) is the OPTIMAL cure"
    classification: CARGO-CULTED
    rationale: "MEASURED NO-GO at W=0.5 (exploded at n24); the DE symbol places the ill-posed mode
      TANGENTIALLY where |grad m|<1 (n^T H n damps the NORMAL direction). Built + tested (default-OFF,
      byte-identical) as an insurance lever, but NOT on the v6 relaunch path."
council_decisions_recorded:
  - "op-routable #1 (operator-authorized): v6 = DE #1 flat lambda_eik (--eikonal-weight 0.05
     --eikonal-weight-end 0.05, drop the up-anneal) + WARM-START from the v5_dseg0026 gold ckpt,
     everything else = v5's argv; run bounded n600 to ~ep150 (past the ep110 re-entry), SC1' + #315
     per-class binding-term-stall classifier armed. §7 has the exact delta + promote/abort gate."
  - "op-routable #2: METHODOLOGY RESULT recorded — the n24 slice is DISPROVEN as a proxy for the
     eikonal re-entry (necessary-not-sufficient). Do NOT use n24 to certify eikonal-stability cures."
  - "op-routable #3 (abort path): if the bounded probe re-enters, the next DE candidate is ADAPTIVE-eps
     (floor the viscosity above the rising CFL lower edge; DE #2). SPEC in §7.4 (needs a flag build)."
  - "op-routable #4: eikonal_penalty_flow_illposedness_v1 + viscous_eikonal_two_sided_window_v1 stay
     FORMALIZATION_PENDING (DE-derived; the constant is crowned only by a clean n600 hold)."
  - "op-routable #5 ($0, optional parallel confirmation): lambda_pre HVP probe at the gold state vs
     lambda_pre(ep100)=3.66e6 — confirms whether the warm-start basin is safer or sharper."
related_deliberation_ids: [stepping_instability_diagnostic_20260705, eikonal_stabilizer_build_20260705,
                           litsweep_training_dynamics_control_20260705,
                           witness_config_differential_equations_derivation_20260705]
---

# V6 EIKONAL-CURE GRAND COUNCIL SYMPOSIUM (#317) — the permanent cure for the eikonal re-entry

**Axis discipline: every arbitration number is `[n24 advisory — mechanism probe, NOT n600 evidence]`;
verdict d_seg rows are `[macOS-MLX advisory] NON-PROMOTABLE`. NOTHING launches at n600 — the recommended
config is operator-GO-gated. Pointer contest-CPU 0.19110 UNMOVED (this whole memo is means/apparatus).**

This symposium's recommendation will be EXECUTED (v6 n600 relaunch composes on it). It therefore
converges the DEEP MATH (with the DE-DERIVATION sibling #318) AND proves the winner past the failure
horizon by MEASUREMENT. Measurement crowns "optimal"; the derivation guides which arms to run.

## 0. The one-paragraph verdict

The eikonal re-entry is **not** a mystery and **not** cured by lr alone: the `(|∇m|−1)²` penalty flow
is a **provably ill-posed (anti-diffusive) PDE** (StEik), and the v5 death is the **discrete crossing
of a two-sided viscous-stability window whose lower edge the ε-anneal walks INTO while sharpening pushes
the edge UP** (DE-DERIVATION #318, `π_eik = η·λ_eik·|c_a|²/(8ε²) ≤ 1`). The cure is therefore **structural,
not a step-size band**: STOP raising λ_eik (both edges scale `1/λ_eik`) and, if needed, floor/adapt ε.
**The n24 cure-arbitration was DISPROVEN as a proxy for this instability** (the v5 config is n24-stable
and n600-fatal — operator: "naive and toy"), so the v6 config rests on TWO real things only: the DE
first-principles derivation + ONE bounded n600 measurement. **RECOMMENDATION (operator-authorized):**
run the **DE-#1 zero-risk config** (flat λ_eik, warm-started from the 0.025 gold) as a **bounded n600
probe to ~ep150**, gated by SC1' + the #315 per-class binding-term-stall classifier — promote to full
run iff it holds past the ep110 re-entry AND d_seg descends; abort to adaptive-ε if it re-enters.

## 1. The measured disease (the chain, not re-derived — see the four related memos)

| run | eikonal config | outcome |
|---|---|---|
| v1 (organic) | eik 0.05, lr 1e-3 | ep92 organic runaway → guard deadlock |
| v4 | lr 5e-4 (cut) | delayed onset; lr-cut = unblock not cure |
| **v5** (`seedfix3_visco_v5`) | **ViscoReg ε=0.3 anneal 1000, lr 1e-3** | **BANKED best-ever d_seg 0.02517 @ep125** (0.124→0.025), d_pose 20.75→0.36 — then **re-entered ~ep109 (eik 295), deadlocked ~ep110** |

v5 is the state of the art: ViscoReg delayed the runaway ~4× longer than the lr-cut and that window
produced the best witness d_seg ever (gold preserved `experiments/results/v5_dseg0026_preserved_20260705/`,
`[macOS-MLX advisory]` on FROZEN deadlock weights — NOT a converged run, NOT byte-closed, NOT a pointer
move). **ViscoReg ε=0.3 is a PARTIAL cure.** This symposium makes it permanent.

## 2. The DEEP MATH — one object, three consistent readings (measurement · symbol · numerics)

### 2.1 The ill-posed flow (StEik 2305.18414, confirmed by our runaway)
The eikonal penalty `L = λ_eik·mean((|∇m|−1)²)` on the decision margin `m = φ_top1 − φ_top2` has a
gradient flow that StEik proves is **anti-diffusive** (backward-heat-like) along the direction the
constraint acts. Our measured runaway (raw eikonal ≈ 2070 at blow-up = |∇m| far off the 1-Lipschitz
target, the term exploding while seg stays flat) is a clean field observation of exactly that instability.
The eikonal term is the **canary** (it MEASURES |∇m|), not the underdog — down-weighting it naively (PINN
GradNorm reflex) would silence the alarm, so the cure acts on the DYNAMICS, not the weight-share.

### 2.2 The symbol (DE-DERIVATION #318) — the mechanism that predicts our numbers
Linearizing the penalty flow `∂m/∂t = λ_eik ∇·[(|∇m|−1) n]` about a near-SDF state gives the Fourier
symbol
```
σ(k) = −k_n² − c_a·k_T²ᵀ ,   c_a = (|∇m|−1)/|∇m|
```
(`k_n` = wavenumber along the normal n=∇m/|∇m|, `k_T` = tangential). Where **|∇m| < 1** (the flat
small-margin annulus — exactly where d_seg lives, `c_a < 0`) the **tangential** mode is backward-heat =
ill-posed. Adding viscosity `ε·Δm` (ViscoReg) caps the growth at `σ_max = |c_a|²/(4ε²)`, giving:

- **CFL lower edge** (under-viscous → the anti-diffusive mode wins): `π_eik := η·λ_eik·|c_a|²/(8ε²) ≤ 1`
  ⟺ `ε ≥ |c_a|·√(η·λ_eik/8)`.
- **Biharmonic upper edge** (over-viscous → the ε·Δ term's own 4th-order stiffness diverges):
  `π_bih := η·λ_eik·ε²·k_max⁴ ≤ 2`.

**This DERIVES the measured two-sided window** (ε=0.3 STABLE / ε=1.0 EXPLODES; FEED-05v). And it
**DERIVES the v5 re-entry**: ε anneals DOWN toward the lower edge while `|c_a(t)|` GROWS under sharpening
(hosc-β 1.0→5.134, tau descent), which RAISES the lower edge — the operating point is squeezed from both
sides until `π_eik > 1` at ε≈0.25–0.27 (v5 ep109 @ ε≈0.267 — the DE derivation lands the number).

### 2.3 The λ_pre negative, vindicated (not a contradiction — a DIFFERENT π-group)
The Adam-EoS `λ_pre* ≈ 38/η` bracket was MEASURED-FALSIFIED (`λ_pre = 3.66e6` at the ep100 state,
π_EoS ≈ 94 ≫ 1 even at the measured-stable lr). The DE derivation resolves the honest negative: the
instability is the **specific eikonal `π_eik`**, not the **generic optimizer `π_EoS`**. lr policy must NOT
be derived from `38/λ_pre`. **Keep η = 1e-3** (the viscosity buys back the step the v4 lr-cut spent).

## 3. The candidates — deep math EACH (with the DE ranking, revised by the symbol)

**(a) StEik-NORMALIZED nᵀHn — the DISCRIMINATOR, not a foregone winner.**
Raw StEik `|∇m^T H ∇m| = |∇m|²·|nᵀHn|` self-amplifies via the quartic `|∇m|²` at the far-from-SDF state
(`|∇m|≫1` → measured 575×–1431× NO-GO). Normalized `κ_n = nᵀHn = (∇m^T H ∇m)/(|∇m|²+δ)` strips it
(scaling `raw ~ c³`, `κ_n ~ c¹`; the removed prefactor is exactly `|∇m|² = c²` — verified vs the DE
numpy oracle, byte-parity test). `κ_n` damps ONLY the **normal-direction** 2nd derivative (anisotropic,
tangential/dash geometry FREE). **BUT** the DE symbol says the ill-posed mode is **tangential** where
d_seg lives — so `κ_n` alone may NOT touch the disease. This makes `steik_norm_05` (κ_n standalone) **THE
DISCRIMINATOR**: H-A (normal-mode) ⇒ κ_n alone stabilizes + best d_seg; H-B (tangential-mode, the
symbol's bet) ⇒ κ_n alone fails, isotropic ε structurally required. **BUILT** (default-OFF flag
`--eikonal-steik-normalized`; 17 tests incl. DE numpy-ref parity; byte-identical OFF).

**(b) non-annealed / higher ε — DERIVED-robust (the FLOOR, then adaptive).**
The v5 re-entry is ε walking below the (rising) lower edge. Fix: keep ε ABOVE it. Constant ε=0.5 sits
further from the lower edge (more margin) and below the upper biharmonic edge (ε=1.0 exploded). The
DERIVED-OPTIMAL is **adaptive** `ε(t) = clamp(|c_a(t)|·√(η·λ_eik/8)·(1+margin), ε_floor, ε_upper)` —
least isotropic damping tracking sharpness ⇒ best d_seg-drift, stable-by-construction (`|c_a(t)|` proxy =
√(eikonal-residual telemetry, already logged). DE high-confidence claim: **adaptive-ε ≥ fixed-ε on
d_seg-drift**. Config-only proxies measured here; adaptive-ε is a follow-on build if the floor is
insufficient.

**(c) lower base λ_eik — DERIVED doubly-stabilizing, cheapest.**
Both edges scale `1/λ_eik` (`π_eik, π_bih ∝ λ_eik`), so LOWERING λ_eik raises both ceilings. MEASURED
CORRECTION: the 0.05→0.10 up-ramp fires at the TAU onset (ep400) — PAST the ep109 re-entry — so eik_w is
already FLAT 0.05 through the entire pre-tau re-entry window; candidate (c) for the re-entry is therefore
LOWERING the base weight (0.05→0.02), a targeted anti-diffusive lr-cut that leaves seg/pose learning at
lr 1e-3. (STOP the up-anneal is still correct for the post-tau leg — DERIVED, ~0-risk.)

**(d) warm-start from the 0.026 gold — MEASURE, don't assume.**
The gold is the best-ever witness state (d_seg 0.026) but is a FROZEN DEADLOCK snapshot (it descended
INTO sharpness). Whether it is a safer basin (lower λ_pre) or a sharper wall is a $0 HVP measurement:
run the λ_pre probe there, compare vs `λ_pre(ep100)=3.66e6`. Lower ⇒ warm-start banks the compute;
not-lower ⇒ the basin is not safer and the stabilizer must carry a fresh ep100 resume. Op-routable #3.

## 4. METHODOLOGY RESULT — the n24 cure-proxy is DISPROVEN for this instability (real signal)

The pivotal apparatus finding is a NEGATIVE about the apparatus itself. `visco_03_anneal` (the EXACT v5
eikonal config: ViscoReg ε=0.3 annealed) ran to **ep215 STABLE at n24** (descended total 71.6→trough
6.3, eikonal 31→0.09; one transient eik spike to 57.9 that RECOVERED) — yet **that identical config
re-entered and deadlocked at n600 ~ep110** (v5). **n24 stability is NECESSARY-NOT-SUFFICIENT.** The DE
derivation predicts exactly this: the lower CFL edge `ε_min = |c_a(t)|·√(η·λ_eik/8)` is driven by the
sharpening rate `|c_a(t)|`, which is landscape/pair-count dependent — the 24-pair slice does not
reproduce the 600-pair basin's sharpening, so it never crosses the edge the full run crosses. **Operator
verdict: "naive and toy."** Consequence, binding: **no v6 config may be crowned on n24 evidence.** n24
remains useful only as a NEGATIVE FILTER (a config that EXPLODES at n24 is definitively bad).

## 5. What the n24 arms DID establish (negative filter only — NOT a crown)

| arm | cure | n24 outcome | usable conclusion |
|---|---|---|---|
| `visco_03_anneal` (v5 ref) | ε=0.3 anneal | STABLE to ep215 (but n600-fatal) | **proves n24 is unfaithful** (the methodology result) |
| `steik_norm_05` | κ_n W=0.5 | **EXPLODED** ep101→123 (total 106→249, eik 53→203, eik_steik climbing) | **candidate (a) NO-GO at W=0.5** — normalized StEik self-forces at this weight; combined with the DE symbol (ill-posed mode TANGENTIAL where \|∇m\|<1) → n^T H n is NOT the v6 lever |
| `visco_05` / `visco_03_eik002` / compound / DE#1-flat / adaptive-ε | (b)/(c)/etc. | **never run** (n24 abandoned per operator) | — (n24 could only have filtered, not crowned) |

Reading: the ONLY load-bearing n24 result is the DISPROOF of the proxy + the κ_n-W=0.5 NO-GO. Everything
else the arbitration would have produced is necessary-not-sufficient and is deliberately not pursued.

## 6. Honest risks (carried to the bounded n600 probe)
1. **DE #1 is PRE-TAU-INERT for the ep110 re-entry itself.** The λ_eik up-anneal (0.05→0.10) fires at
   the TAU onset (ep400), PAST the ep110 re-entry — so at ep110 λ_eik is already 0.05 flat in BOTH v5
   and v6. Dropping the up-anneal is the DE-highest-confidence, ~zero-risk change but it protects the
   POST-tau leg; the ep110 re-entry itself is carried by the **warm-start from the gold** (the run no
   longer re-crosses the ep100→110 sharpening transient) and is the thing the bounded probe MEASURES.
   If the ε-anneal's lower-edge crossing still bites, the gate fires → adaptive-ε (§7.4).
2. **The gold is a FROZEN DEADLOCK state** (best d_seg 0.026 last-accepted, elevated eikonal at freeze).
   Warm-starting banks the d_seg progress but resumes into a sharpened basin; `--resume-clear-spike-guard`
   re-arms the guard. The λ_pre HVP probe at the gold (op-routable #5) confirms whether that basin is
   safer or sharper than ep100 — recommended as a $0 parallel, not a blocker.
3. **Smoothing-vs-sharpness**: v6 KEEPS ViscoReg ε=0.3 (isotropic Δm damps tangential too); the #315
   per-class binding-term-stall classifier is the MEASURED check that the lane/dash classes are not
   being smoothed into a stall (it catches exactly the per-class stall v5's aggregate d_seg hid).
4. **λ_pre gap**: the generic 38/η optimizer bracket is measured-falsified; lr stays η=1e-3 (the specific
   eikonal π_eik governs, not π_EoS — DE-vindicated).

## 7. THE PROCEED RECOMMENDATION — the bounded n600 probe (operator-authorized, main executes)

### 7.1 Exact v6 config = flag-delta vs the v5 argv (`experiments/results/levelset_n600_witness_20260705T155150Z/launch.sh`)
```
CHANGE  --eikonal-weight-end 0.1   →   --eikonal-weight-end 0.05     # DE #1: FLAT lambda_eik, drop the
                                                                     #        0.05->0.10 up-anneal
                                                                     #        (both CFL ceilings ~ 1/lambda_eik)
CHANGE  --resume-from .../bd_calib_20260705/snap/resume_state_ep100.npz
        →  --resume-from experiments/results/v5_dseg0026_preserved_20260705/levelset_resume_state.npz
                                                                     # WARM-START from the 0.026 gold
KEEP    everything else in the v5 argv EXACTLY:
        lr 1e-3 / lr-end 1e-4 · --eikonal-weight 0.05 · --eikonal-viscosity 0.3
        --eikonal-viscosity-anneal 1000 · --boundary-distance-weight 0.2
        --resume-allow-lever-drift --resume-clear-spike-guard · --tau-softplus-start-epoch 400
        · accum-pairs 8 · all levers/curriculum as v5 · --cache-gt-skeleton --fused-r-kernel
BOUND   run to ~ep150 past the ep110 re-entry (a BOUNDED probe, NOT the full 1000-ep run);
        --eval-every 25 (or 10 through the ep100-150 window) · --stage-checkpoints (per-stage resumable)
WATCH   SC1' every-epoch skip-rate alarm + the #315 per-class binding-term-stall classifier
        (_verdict_dseg_dpose_nucleus_chunked; catches per-class stall the aggregate d_seg hides)
        + the per-epoch loss_terms `eikonal` row (recurrence detected within 1 epoch)
```
Rationale (DE mechanism, NOT n24): DE #1 raises both CFL ceilings (`π_eik, π_bih ∝ λ_eik`) and is
~zero-risk; the warm-start from the 0.026 gold banks the best-ever d_seg and avoids re-walking the
sharpening transient that killed v5; the bounded horizon spends ~1 GPU-hour to MEASURE the one thing n24
cannot tell us. lr stays 1e-3 (the viscosity buys back the step; the 38/η bracket is falsified).

### 7.2 PRE-REGISTERED PROMOTE gate (all must hold through the ep100/125→ep150 window)
1. **No eikonal re-entry**: the per-epoch `eikonal` loss-term stays bounded (descends from the
   warm-start level; NO monotone runaway; MAX ≤ ~5× its post-resume trough) — the litsweep canary.
2. **SC1' healthy**: skip-rate < 10%/epoch every epoch (no deadlock absorbing state).
3. **d_seg descends/holds**: the async CPU verdict shows d_seg ≤ the 0.026 gold and non-increasing
   (a REAL contest-CPU-authority number, `[macOS-CPU advisory]` until byte-closed) — NOT a proxy.
4. **#315 no per-class stall**: the per-class binding-term-stall classifier does NOT flag a lane/movable
   class stalling (the failure v5's aggregate d_seg hid).
→ ALL hold ⇒ **PROMOTE to the full n600 run** (continue to the tau/Muon curriculum), SC1' armed.

### 7.3 PRE-REGISTERED ABORT gate → next DE candidate
Any of {eikonal re-enters · SC1' deadlock · d_seg regresses above 0.026 · #315 per-class stall} ⇒
**ABORT** the probe, preserve the checkpoint, and escalate to **§7.4 adaptive-ε** (the DE #2 candidate
that DIRECTLY addresses the lower-edge crossing DE #1 does not).

### 7.4 The abort-path DE #2 candidate — ADAPTIVE ε(t) (SPEC; needs a flag build, ~zero-risk by construction)
The v5 death is the ε-anneal walking below the RISING lower CFL edge. The DE-optimal fix tracks it:
```
ε(t) = clamp( |c_a(t)| · √(η · λ_eik / 8) · (1 + margin),  ε_floor,  ε_upper )
```
where `|c_a(t)| = mean|(|∇m|−1)/|∇m|`| is read from the ALREADY-LOGGED eikonal-residual telemetry (a
no-grad recompute), `margin ≈ 0.5`, `ε_floor ≈ 0.3` (never anneal below the FEED-05v stable floor),
`ε_upper ≈ 0.7` (stay below the ε=1.0 biharmonic explosion). This is stable-BY-CONSTRUCTION (ε rides
just above the lower edge, minimal isotropic damping ⇒ least dash-erosion, best d_seg-drift — DE
high-confidence: adaptive-ε ≥ fixed-ε). Build: a `--eikonal-viscosity-adaptive` flag replacing the linear
anneal with the clamp above; default-OFF byte-identical; fires ONLY on the abort path.

### 7.5 Insurance lever, BUILT but OFF the v6 path
The normalized `κ_n = nᵀHn` operator (`--eikonal-steik-normalized`, `--eikonal-steik-norm-eps`;
default-OFF, byte-identical; 17 tests incl DE #318 numpy-ref parity) is landed as a spare tire. It is a
MEASURED NO-GO at W=0.5 (self-forced) and the DE symbol places the ill-posed mode tangentially — so it is
NOT on the relaunch path. Retained for a future normal-mode instance or a much-lower-W anisotropic probe.

## Artifacts
- Build: normalized κ_n operator `_eikonal_steik_normalized_mlx` + flags `--eikonal-steik-normalized`
  / `--eikonal-steik-norm-eps` in `experiments/train_levelset_witness_realized_through_R_mlx.py`
  (default-OFF, byte-identical); tests `src/tac/tests/test_eikonal_steik_normalized.py` (17, incl DE
  #318 numpy-ref parity); probe arms in `experiments/probe_resume_stepping_instability.py`
  (commit `ab5a6e5fe`).
- DE-DERIVATION #318 memo: `.omx/research/witness_config_differential_equations_derivation_20260705.md`;
  numpy oracle `src/tac/boundary_math/eikonal_normal_curvature_reference.py`.
- Arbitration: `experiments/results/v6_reentry_arbitration_20260705/` (per-arm logs + argv).
