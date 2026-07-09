# The ξ→PoseNet Jacobian — conditioning, the pose-descent BASIN TRIGGER, and a default-on basin telemetry (formalization + design, 2026-07-09)

**Axis:** `[formalization / design — no run]` · **$0, no GPU/MLX/paid, no launch.** Pointer contest-CPU
**0.19110 UNMOVED — MEANS.** git `60ba8231f`. This is a FORMALIZATION + APPARATUS-DESIGN + telemetry-spec
task (operator directive 2026-07-09). It does NOT train, does NOT edit the trainer, does NOT re-seal v7.5.
Every claim is labeled **MEASURED** (cite artifact) / **DERIVED** (from the scorer's own structure) /
**ASSUMED** (flagged, owed-on-measurement). No over-claim: the Jacobian conditioning is the INSTRUMENT that
makes the pose-timing question measurable — it is NOT a proof that an earlier engage-point wins. That
optimality is a HYPOTHESIS to be MEASURED (the resume-A/B on the engage-point).

STORES CONSULTED (proactive recall — this is ~85% recall of settled anchors, ~15% new derivation):
`pose_legible_witness_aperture_design_20260708` (§1 = the Jacobian near-rank-deficiency argument this
memo formalizes) · `pose_aperture_probe_measured_20260708` (the aperture FALSIFICATION on the cheap fixed
carrier — the honesty bound this memo must respect) · `pose_solve_output_space_inverse_20260708` (§0a
refutation: a fixed-render post-hoc warp floors ~1.2–1.5; the reachable-set argument) · FEED-238resolved +
`TerminalPoseFinish` v7.5 D.9 (`curriculum_dsl.py:2247`; the CURRENT terminal engage-point this memo
proposes to make event-conditioned) · `morse_smale_stratified_parallax_dpose_20260708.py` (the canonical
equation this extends) · upstream `modules.py:108` (SegNet last-frame — the seg⊥pose structural kernel) ·
`modules.py:60–95` (PoseNet input/`compute_distortion` — the d_pose = ‖p−t‖²/6 relation) · #206/#227
(seg⊥pose ~99.95% null) · #141 margin field · CLAUDE.md L80 class order.

---

## 0. THE OPEN QUESTION, PRECISELY

v7.5 D.9 (`TerminalPoseFinish`, `curriculum_dsl.py:2247`) engages the pose term **TERMINALLY** — pose is
BLIND (effective `w_pose→0`) until the muon switch fires (`--muon-start-event`, `powerlaw_meat` = the
d_seg-converged regime), then the joint pose-descent runs. The **MEASURED** rationale (settled, do not
re-litigate): pose co-trained from ep0 on an INCOHERENT render produced the ~1.79 plateau (live #205
w_pose=1.0 sits d_pose ~1.75 at ep200, pre-Muon); R1 warm-started from a CONVERGED render reached 0.0011.

The **operator question**: is TERMINAL optimal, or should pose engage EARLIER — the moment the render is
"coherent enough"? The binding variable is render COHERENCE. This memo's thesis: **render coherence is
exactly what the conditioning of the ξ→PoseNet Jacobian MEASURES**, so the Jacobian's smallest singular
value σ_min is the rigorous, computable "basin" sensor — and the earliest-safe engage-point becomes a
MEASURABLE quantity (the σ_min(epoch) curve), not a guess.

---

## 1. THE JACOBIAN J_ξ — definition, shape, chain rule (DERIVED)

Fix the current witness parameters θ (the render at this epoch). For pair k, the render operator produces
a two-frame tensor `F = R(θ, ξ) = (f0, f1)` where the per-pair twist **ξ = (ρ, ω) ∈ se(3) ≅ ℝ⁶**
(translation-first convention, `tac.lie.se3.CONVENTION`) parametrizes the ego-motion that generates the
seg-FREE canvas frame f0 relative to the seg-locked witness frame f1 (§4). `R` includes the eval
roundtrip (`#204`: bicubic↑874 → uint8-STE → bilinear↓512×384). PoseNet's scored output is

  **p(ξ) := PoseNet(R(θ, ξ)).pose[:6] ∈ ℝ⁶**   (the first 6 of the 12-dim pose head; `modules.py:83`,
  `compute_distortion` uses `[..., :h.out//2]`).

**The Jacobian is**

  **J_ξ(θ) = ∂p/∂ξ ∈ ℝ^{6×6}**,  (J_ξ)_{ij} = ∂ PoseNet(R(θ,ξ)).pose[i] / ∂ξ_j.

**Chain rule** (the two factors that BOTH must be full-rank):

  **J_ξ = J_P · J_R**,  where
  - **J_P = ∂PoseNet/∂F |_{F=R(θ,ξ)} ∈ ℝ^{6×N}** — the SCORER sensitivity (how the 6 outputs respond to
    pixel changes; N = frame-pair pixel count at eval res). Frozen PoseNet; a property of the scorer.
  - **J_R = ∂R/∂ξ ∈ ℝ^{N×6}** — the RENDER-FLOW Jacobian (how pixels move as ξ changes, through the R
    roundtrip). A property of the render θ.

The 6×6 product is rank-deficient iff J_P and J_R are rank-deficient in a **shared** ξ-subspace. §2 shows
J_R is the binding factor at low render-coherence.

**Conditioning (SVD J_ξ = U Σ Vᵀ, Σ = diag(σ_1 ≥ … ≥ σ_6 ≥ 0)):**
- **rank** r = #{σ_i > tol}.
- **condition number** κ = σ_1 / σ_6 = σ_max / σ_min.
- **σ_min = σ_6** — the SMALLEST singular value. Its right singular vector v_6 is the ξ-perturbation
  direction that produces the LEAST output change: ‖J_ξ v_6‖ = σ_6. σ_min→0 ⟺ some ego-motion direction
  is INVISIBLE through render+PoseNet ⟺ the descent cannot move the output along u_6 ⟺ any target whose
  residual has mass along u_6 is UNREACHABLE. **σ_min is the basin variable.**
- **effective rank** r_eff = exp(H(σ̃)), σ̃_i = σ_i/Σ_j σ_j, H = −Σ σ̃ ln σ̃ — a threshold-free soft rank.

Per-pair J_ξ^(k); aggregated over the n600 (or a stratified subsample, §5).

---

## 2. THE COHERENCE ↔ CONDITIONING LINK (DERIVED, MEASURED-consistent)

**Why σ_min→0 on a flat/incoherent render.** The warp writes f0 from the render source by advection:
f0(x) = source(W_ξ(x)), W_ξ the ξ-parametrized flow field. Differentiate (brightness-constancy structure):

  **∂f0/∂ξ_j (x) = ∇source(W_ξ(x)) · ∂W_ξ(x)/∂ξ_j = [∇source]ᵀ [∂W/∂ξ_j].**

The **image gradient ∇source appears as a MULTIPLICATIVE factor.** On a piecewise-constant (flat cartoon)
render, ∇source = 0 in every cell INTERIOR and is nonzero only on the codim-1 separatrix (the boundaries).
Therefore:

  **J_R = ∂R/∂ξ is supported ONLY on the boundary set** (measure ≈ the annulus; **MEASURED** ~4.7% of
  pixels carry ~97% of the d_seg mass, #333; lane-edge residual 4.67%, L80). In cell interiors J_R ≈ 0 for
  EVERY ξ direction.

**The aperture problem, formalized.** At a boundary point the gradient points along the edge NORMAL n̂:
∇source = |∇source| n̂. Hence

  **∂f0/∂ξ_j (x) = |∇source(x)| · [ n̂(x)ᵀ ∂W(x)/∂ξ_j ].**

Only the NORMAL component of the flow ∂W/∂ξ_j is observable; the tangential component lies in the kernel.
Stacking over boundary pixels, the OBSERVABLE ξ-subspace is span{ n̂(x)ᵀ ∂W(x)/∂ξ : x ∈ boundary }. A flat
cartoon has FEW, long, near-straight edges → few distinct normals n̂ → this spanning set has LOW rank →
**σ_min(J_R) → 0 → σ_min(J_ξ) → 0** (since σ_min(J_ξ) ≤ σ_1(J_P)·σ_min(J_R)). **DERIVED.**

**Why σ_min RISES as d_seg converges.** As the argmax partition converges, the render acquires structure:
more, finer, CURVED boundaries (lane dashes, the hood outline, movable edges — the very long-tail features
whose ERASURE is the measured d_seg residual, L65) → a RICHER set of normals n̂ AND larger boundary
measure → the observable ξ-subspace FILLS OUT → σ_min rises and plateaus. Chain: **d_seg convergence →
richer f1 partition → richer f0 canvas (its warp) → ∇source richer → J_ξ better-conditioned → σ_min↑.**
**DERIVED**; **MEASURED-consistent** with the flat floor 1.2–1.8 (A0 1.685 / A2 1.486 / A2+ 1.223) vs
R1-from-converged 0.0011.

**⛔ THE HONESTY BOUND (from the aperture FALSIFICATION — do not over-read σ_min).** The aperture probe
(`pose_aperture_probe_measured_20260708`, MEASURED) proved that on a fixed CHEAP carrier, making the flow
observable (painting texture) made d_pose WORSE (best 15.1 ≫ 1.685), because the cheap carrier's flow
W = ground-homography is the WRONG flow (diagnostic: warp(f)-vs-f reads 166–186). **Lesson, precisely
stated: σ_min(J_ξ) > 0 is NECESSARY (observability / a well-posed local inverse) but NOT SUFFICIENT — the
reachable set must ALSO contain the target, which additionally requires the flow the render CAN express to
span the CORRECT output directions.** Two distinct axes:
- **σ_min = the OBSERVABILITY axis** (can the render express output-space motion in all 6 directions?).
- **flow-model correctness = the CONTENT axis** (does the expressible motion point at the true target?).

The fixed cheap carrier failed the CONTENT axis (falsified). **JOINT pose-descent is different on exactly
this axis:** θ co-adapts, so the render is NOT restricted to a fixed cheap warp family — it acquires
whatever structure minimizes the realized d_pose through R. σ_min is therefore the correct basin sensor
FOR JOINT DESCENT (θ supplies the content DOF the fixed carrier lacked); it is NOT a resurrection of the
fixed-cheap-carrier (which is dead on the content axis, verdict_scope FORMULATION). **This distinction is
the load-bearing nuance; σ_min gates OBSERVABILITY, joint θ-descent supplies CONTENT.**

---

## 3. THE BASIN TRIGGER (DERIVED structure; the threshold is a MEASURABLE pre-registration)

The pose descent (Gauss-Newton/LM on ξ_eff per pair, or the D.9 joint finish) drives p(ξ)→target t. The
GN step is Δξ = −(J_ξᵀJ_ξ + λI)⁻¹ J_ξᵀ(p−t). Local behavior is governed by σ_min:
- σ_min ≈ 0: the GN system is singular along v_6; the residual component of (p−t) along u_6 is
  IRREDUCIBLE within the render family → the ~1.2–1.5 floor (**MEASURED**, matches A2/A2+).
- σ_min ≥ σ*: a bounded local inverse exists, ‖Δξ‖ ≤ ‖p−t‖/σ_min; the image of the ξ trust-ball of radius
  ρ under p covers an output ball of radius ≥ σ_min·ρ − O(curvature·ρ²). **The target at output-distance
  d_0 is reachable when σ_min·ρ_budget ≳ d_0.**

**The reachability floor (DERIVED).** From `modules.py` `compute_distortion` (MSE over 6 dims):
d_pose = ‖p−t‖²/6, so the initial output-space residual is **d_0 = ‖p(ξ_init)−t‖ = √(6·d_pose_init)**,
directly readable from the verdict d_pose row. Reachability within the finish-stage trust-region ρ_budget
gives the pre-registered floor

  **σ* = d_0 / ρ_budget = √(6·d_pose_init) / ρ_budget.**

**The fire criterion (pre-registered, aggregate-with-tail-guard — mirrors `powerlaw_meat`/annulus):**

  **FIRE pose-finish at the earliest verdict-epoch where**
  **median_k σ_min(J_ξ^(k)) ≥ σ*  AND  frac_k[ σ_min(J_ξ^(k)) ≥ σ_floor ] ≥ q.**

The median gives the basin level; the quorum fraction q (ASSUMED default 0.8) guards against a
well-conditioned head carrying a still-starved low-|t| tail (the tail matters: d_pose anti-correlates with
|ego-t|, **MEASURED** corr −0.45/−0.68, so low-motion pairs condition LAST).

**Honest pre-registration of the threshold (the part that is NOT knowable a priori).** ρ_budget
(step-budget-dependent) and the absolute σ* are not derivable in advance. The MEASURABLE, disciplined
form: express the trigger RELATIVE to the σ_min PLATEAU the telemetry observes at d_seg convergence:

  **FIRE when median σ_min ≥ f_basin · σ_min^plateau,  f_basin ∈ (0,1].**

The telemetry (§5) MEASURES the whole σ_min(epoch) curve (rising → plateau). f_basin=1.0 reproduces the
current TERMINAL policy exactly. **The optimality of f_basin<1 (earlier engage) is the RESUME-A/B
HYPOTHESIS — the instrument makes it measurable; it is NOT proven here.** This turns the operator's "engage
earlier?" intuition into a single scalar knob with a measured curve behind it.

---

## 4. THE seg⊥pose SAFETY — a Jacobian-geometry REASON for the ~99.95% null

**The ξ-channel decoupling is EXACT-by-construction (DERIVED).** `modules.py:108`:
`x = x[:, -1, ...] # Use only last frame` — **SegNet reads ONLY frame1 (the last frame).** The pose
carrier's ξ (and its dxi params) shapes ONLY frame0 (the seg-FREE canvas; the FRAME0-FREE-CANVAS firewall,
operator-approved). Therefore

  **∂ d_seg / ∂ξ ≡ 0  EXACTLY** — ξ lives entirely in the seg-free frame's pixel subspace, which is in the
  EXACT kernel of the d_seg readout (SegNet never evaluates frame0). This is the Jacobian-geometry REASON
  the ~99.95% seg⊥pose null (**MEASURED** #206/#227) is not a coincidence: engaging pose EARLY cannot
  disturb the converging d_seg through the ξ channel, at ANY epoch. **This is why earlier engage is SAFE.**

**The θ-channel residual is MEASURED, not exactly-zero.** In JOINT descent the shared render params θ can
move BOTH frames, so ∂d_seg/∂θ ≠ 0 and the pose gradient w.r.t. θ has a small projection onto the
d_seg-relevant θ subspace. That residual coupling is the ~0.05% (**MEASURED**), not derivable to zero.
**The same JVP machinery MEASURES it** (bonus telemetry): the principal angles between col(J_ξ) (the
pose-reachable output directions realized via θ-perturbations) and row(∂d_seg/∂θ) (the annulus-relevant θ
directions). Large principal angles ⇒ safety confirmed live; a collapse ⇒ a loud alarm. **DERIVED that it
is measurable; the value is OWED-on-measurement.**

---

## 5. THE DEFAULT-ON BASIN TELEMETRY (spec: quantities, compute cost, cadence)

Per the "default-off is orphaned signal" law: score-neutral read-only observability DEFAULTS ON when
CHEAP; score-neutral but HEAVY-compute observability defaults to a TRACKED-CADENCE knob with its reason
recorded (the exact pattern of `grad_interaction`/`curvature` in `TelemetryCadence`,
`curriculum_dsl.py:917`). The basin sensor splits cleanly into two tiers on that boundary:

**T0 — the coherence PROXY (default ON, per-verdict, near-free).** ∇source is the multiplicative factor in
J_R (§2), so a render-boundary-gradient-energy scalar is a monotone PROXY for σ_min at near-zero cost —
one finite-difference on the already-rendered f0 (no PoseNet, no backward):
- `render_grad_energy = mean(|∇f0|²)` (and its per-Morse-class restriction → per-class coherence).
- Also emit the annulus mass already computed by the annulus telemetry (free reuse).
This is the always-on basin HINT: it moves BEFORE T1 and costs nothing. **DERIVED** monotone-correlated
with σ_min; the exact map is calibrated once T1 lands (T0 = proxy, T1 = authority).

**T1 — the true J_ξ conditioning (default-HELD cadence, ~0.6× a verdict eval).** On a stratified subsample
K ⊂ [n600] (|K| = k_pairs, ASSUMED default 32; **stratified by |ego-t|** so the anti-correlated low-motion
tail is represented — otherwise the median hides the starved pairs):
- per pair: σ_1…σ_6 of J_ξ^(k) → σ_min, κ = σ_1/σ_6, r_eff.
- aggregates over K: **median σ_min, p10 σ_min (the tail), median κ, median r_eff, basin_frac =
  frac_k[σ_min ≥ σ_floor]**.
- per-class σ_min (mask J_R support to class-c annulus) — connects to v8 per-class carriers.
- (bonus) the seg⊥pose principal-angle min (§4 live safety readout).

**Compute cost (DERIVED).** J_ξ^(k) is 6×6; via reverse-mode `mx.vjp` (available, used in
`test_metal_fused_r_operator.py`) it costs 6 VJPs (one per scored output) through R+PoseNet per pair, each
VJP ≈ 2× a forward ⇒ ~12 forward-equivalents/pair. × k_pairs=32 = **~384 forward-equivalents ≈ 0.64× one
n600 verdict eval** (600 forwards). The 6×6 SVD is negligible. **This is NOT free-per-epoch** — hence T1
follows the `curvature` pattern: a HELD cadence knob (`jacobian_basin_every`, ASSUMED default = every 4th
verdict, governor-gated), reason recorded, NEVER a hidden switch. T0 (near-free) carries the always-on
duty; T1 supplies the authoritative curve on cadence.

**The offline-against-checkpoints escape hatch (the cleanest build).** We ALREADY mandate per-stage EMA
checkpoints (the resumability non-negotiable). J_ξ σ_min at each saved shadow can be computed POST-HOC,
reconstructing the entire σ_min(epoch) basin curve with **ZERO impact on the live run** (read-only, off the
training stream). So the σ_min(epoch) curve that feeds the resume-A/B need not touch the hot loop at all.

---

## 5b. THE BUILD CONTRACT (operator scope-elevation 2026-07-09 — BUILD IT, run it LIVE on v7.5, launch WAITS)

The telemetry is no longer "consider adding" — it is BUILDING and running LIVE on the v7.5 launch as an
OBSERVER, and the launch WAITS on it. Five binding build clauses so the build agent has an unambiguous,
launch-path-safe contract for a MULTI-DAY run:

**(B1) SCORE-NEUTRAL BY CONSTRUCTION — the assertable invariant.** The telemetry READS the current witness
state (θ / EMA shadow / rendered frames) and the FROZEN PoseNet, computes σ_min(J_ξ) via `mx.vjp`, and
EMITS a telemetry row. It MUST NOT: write to any trained parameter, the optimizer state, the loss, the
gradient, the archive bytes, the RNG stream consumed by training, or the EMA shadow. Byte-identity of the
trained artifact is preserved BY CONSTRUCTION. The build asserts this two ways: (a) the sensor takes the
witness state as a READ-ONLY argument and returns a dict (no in-place mutation, no return into the train
loop); (b) a launch-path parity check — one short run WITH the telemetry on vs OFF produces bit-identical
per-stage checkpoints (the fused-R determinism harness already gives byte-compare machinery). This is the
same class as the `annulus`/`confound` observers: read-only ⇒ no re-seal.

**(B2) FAIL-OPEN — the telemetry can NEVER crash the run.** The entire per-verdict sensor body is wrapped
`try/except BaseException → log the exception (typed `jacobian_basin_telemetry_error` row: epoch, verdict,
exc repr) + SKIP this row + continue training`, exactly like the confound telemetry. A JVP shape mismatch,
an MLX op gap, an SVD non-convergence, an OOM in the sensor — ANY failure logs and skips; the training loop
proceeds untouched. A multi-day run MUST survive any telemetry error. The sensor also self-disables after N
consecutive failures (ASSUMED N=3) with a loud `jacobian_basin_telemetry_disabled` row so a persistent
fault degrades to zero cost instead of spamming. NO telemetry failure is ever a training failure.

**(B3) COMPUTE COST + CADENCE — concrete, honest, run-safe.** Full n600 per-verdict is TOO HEAVY: 6 VJPs ×
600 pairs × ~2 fwd-equiv = ~7200 forward-equivalents = ~12× a verdict eval — unacceptable per-verdict on a
multi-day run. **Recommended build default: SUBSAMPLED + CADENCED.** k_pairs = 32 stratified by |ego-t|
(the anti-correlated tail must be represented) ⇒ ~384 fwd-equiv ≈ **0.64× one verdict eval**, emitted every
`jacobian_basin_every` = **4th verdict** (governor-gated). Net overhead ≈ 0.64/4 ≈ **0.16× a verdict eval
amortized per verdict** — negligible against a multi-day run. T0 (the near-free ∇f0 proxy, §5) rides EVERY
verdict as the always-on hint. If even the k=32/every-4 cost is contested at launch, fall back to k=16 or
every-8 — the σ_min(epoch) CURVE (not a single precise value) is what the basin needs, so a coarse cadence
still resolves it. Concrete build knobs: `k_pairs=32, jacobian_basin_every=4, stratify_by_t=True`.

**(B4) LAUNCH-PATH SAFETY — the GATE_KEY_PREFIXES lesson (a config test is NOT enough).** The build MUST
pass a LAUNCH-PATH test: actually START the run (governed launcher) with the telemetry ON and confirm the
run reaches + passes the FIRST verdict/telemetry emission — not merely that the config compiles and a unit
test of the sensor passes in isolation. The GATE_KEY_PREFIXES incident is the anchor: a "safe" additive
change passed config/unit tests yet crashed EVERY launch because the config tests never exercised the live
launch path. Required gate: `tools/launch_witness_run.py` (or the smoke equivalent) starts, renders, hits
the verdict path with `--jacobian-basin-telemetry` on, emits ≥1 basin row (or ≥1 fail-open skip row —
either proves the path is wired and non-crashing), THEN the multi-day launch proceeds. This gate is the
thing the launch WAITS on.

**(B5) OBSERVER-FIRST on the FIRST v7.5 run — the trigger stays TERMINAL until the sensor is TRUSTED.** On
the first run the basin telemetry is an OBSERVER ONLY: it MEASURES and LOGS σ_min(epoch), median/p10/κ/
r_eff/basin_frac, and RECORDS the epoch at which the basin criterion (§3) WOULD have fired — but the
pose-finish stage stays at its SEALED TERMINAL default (`TerminalPoseFinish` on `--muon-start-event`,
f_basin≡1.0). The sensor does NOT actuate the engage-point on run 1. Rationale: an event-TRIGGERED
engage-point is a score-AFFECTING actuation (it changes WHEN pose engages ⇒ changes the trained artifact),
so it is a LEVER, not an observer — and per "levers default off, registered, duty-to-measure," it must not
fire until the basin sensor is TRUSTED by measurement. Run 1 produces exactly that trust artifact: the
measured σ_min(epoch) curve + the would-have-fired epoch, against the actual terminal outcome. The
basin-TRIGGERED actuation (f_basin<1) is the NEXT iteration / the resume-A/B — launched from run 1's
per-stage checkpoints against the measured curve. Observer now; actuator once measured.

---

## 6. TRIALITY LEGS (DERIVED-now vs OWED-on-measurement)

**DSL (design DERIVED-now; flags OWED-on-build — the trainer has no J_ξ hook, so conservative compile =
TrainerSupportGap, exactly like `powerlaw_meat`):**
- A `JacobianBasin` sensor dataclass folded into `TelemetryCadence`
  (`curriculum_dsl.py:907`): `{t0_grad_energy: bool=True (default ON, near-free),
  t1_sigma_min: bool=True (default ON as OBSERVER per B5, subsampled+cadenced per B3),
  k_pairs:int=32, stratify_by_t:bool=True, jacobian_basin_every:int=4, sigma_floor:float,
  quorum_q:float=0.8}`. BOTH are OBSERVERS (read-only B1, fail-open B2); T1 records its cadence reason
  (off-is-tracked-queue). This is the run-1 build.
- A new ENTRY-trigger criterion `jacobian_basin` (sibling of `powerlaw_meat` in `_EXIT_EVENT_CRITERIA`,
  `curriculum_dsl.py:611`) keyed on `median σ_min ≥ f_basin·σ_min^plateau AND basin_frac ≥ q`. This is the
  ACTUATOR — a score-affecting LEVER (it changes WHEN pose engages), so it stays default-off / duty-to-
  measure and does NOT fire on run 1 (B5); it is the run-2 A/B arm, gated on the run-1 basin curve.
- `TerminalPoseFinish(..., start_event="jacobian_basin", f_basin=…)` — a new option beside the existing
  `--muon-start-event`/`start_epoch` (`curriculum_dsl.py:2247`). f_basin=1.0 = the current TERMINAL policy
  (byte-identical incumbent, the run-1 default); f_basin<1 = the earlier-engage run-2 A/B arm. `start_epoch`
  stays the fail-safe CAP.
- Config-orphan discipline: the sensor + the actuator land as `Lever`/sensor factories WHEN BUILT, never a
  hand-added trainer flag.

**DAG:** FEED-posejac (this memo + the σ_min basin instrument + the T0/T1 telemetry spec + the
f_basin resume-A/B); appended same-commit.

**Equations (structure DERIVED-now; MEASURED anchors OWED):** register
`pose_jacobian_basin_conditioning_v1` (or extend `morse_smale_stratified_parallax_dpose_v1`):
- the coherence↔conditioning law σ_min(J_ξ) ↓ 0 as boundary-normal diversity ↓ (§2, DERIVED);
- the reachability floor σ* = √(6·d_pose_init)/ρ_budget and the relative trigger f_basin·σ_min^plateau (§3);
- the exact ξ-channel seg⊥pose kernel ∂d_seg/∂ξ≡0 (§4, DERIVED) + the θ-channel residual (MEASURED-owed).
Anchors OWED: the σ_min(epoch) curve from run-1's per-stage checkpoints (offline, §5) + the resume-A/B on
f_basin. n600 + exact-eval owed before any promotable pose number.

---

## 7. BUILD TIMING — the operator decision (2026-07-09) + honest tradeoff

**DECIDED (operator scope-elevation): BUILD T0 + T1 and run BOTH LIVE on the v7.5 launch as OBSERVERS; the
launch WAITS on the launch-path gate (B4).** This memo's original "T1 offline-first" recommendation is
SUPERSEDED for run 1 by the operator's decision to measure the basin LIVE — reconciled below; the offline
route is retained as the zero-risk backup.

- **T0 (near-free ∇f0 proxy, default-on, per-verdict): build into v7.5 now.** One finite-difference on an
  already-rendered frame — score-neutral, no re-seal, negligible wall-clock. The always-on basin hint.
- **T1 (authoritative σ_min, subsampled+cadenced per B3, OBSERVER per B5): build into v7.5 now, run live.**
  At the B3 default (k=32, every-4) its amortized overhead is ~0.16× a verdict eval — safe for a multi-day
  run. It runs as an OBSERVER (logs σ_min(epoch) + the would-have-fired epoch; does NOT actuate the
  engage-point). Guarded by B1 (score-neutral assertion), B2 (fail-open), B4 (launch-path gate).
- **The offline-against-checkpoints route (§5) is retained as the BACKUP + the run-2 A/B substrate.**
  Because per-stage EMA checkpoints are mandated, the σ_min(epoch) curve can ALSO be reconstructed post-hoc
  at zero live risk — this is the safety net if the live sensor fails-open (B2) for a stretch, AND it is the
  substrate for the run-2 resume-A/B on f_basin (which relaunches from run-1 checkpoints against the curve).

**Honest tradeoff (why live-observer on run 1 is the right call).** PRO: the ONE run measures the basin
curve against its ACTUAL terminal outcome — the highest-fidelity trust artifact for the sensor, and it
front-loads the signal instead of waiting for a post-hoc pass. CON: T1 is new hot-path code, so it MUST
clear the B4 launch-path gate + a JVP-vs-finite-difference parity check on the sensor BEFORE the multi-day
launch (a config/unit pass is not sufficient — the GATE_KEY_PREFIXES lesson). The fail-open wrap (B2) caps
the downside: worst case the sensor self-disables and the run is byte-identical to no-telemetry. The
combination — T0+T1 live as observers, TERMINAL trigger unchanged on run 1, actuator deferred to the run-2
A/B — captures the basin signal at launch while keeping the trained artifact sealed and the engage-point at
its trusted default. **The basin-trigger OPTIMALITY remains a hypothesis the run-2 resume-A/B decides —
σ_min is the instrument that makes that A/B possible, not a proof that earlier wins.**

---

## 8. FINAL STATE

$0, no GPU/MLX/paid, no launch, no trainer edit, no re-seal. Pointer **0.19110 UNMOVED — MEANS.** This memo
delivers: (1) J_ξ defined + chain-ruled (DERIVED); (2) the coherence↔conditioning link via the
∇source-multiplicative aperture structure, bounded by the aperture-falsification honesty (DERIVED,
MEASURED-consistent); (3) the σ_min basin trigger σ* = √(6 d_pose_init)/ρ_budget → the measurable relative
form f_basin·σ_min^plateau (DERIVED structure, MEASURED threshold); (4) the EXACT ξ-channel seg⊥pose kernel
∂d_seg/∂ξ≡0 as the Jacobian-geometry reason for the ~99.95% null (DERIVED) + the θ-residual (MEASURED-owed);
(5) the two-tier telemetry (T0 near-free default-on proxy + T1 subsampled/cadenced σ_min authority) with
derived compute cost; (5b) the BUILD CONTRACT for the operator's live-on-v7.5 scope-elevation — B1
score-neutral-by-construction (assertable), B2 fail-open (never crashes the multi-day run), B3 compute+
cadence (k=32 stratified, every-4-verdict, ~0.16× amortized), B4 launch-path gate (GATE_KEY_PREFIXES
lesson — not just a config test), B5 OBSERVER-FIRST on run 1 (measure σ_min + would-have-fired epoch,
pose-finish stays TERMINAL, actuator deferred to the run-2 A/B); (6) triality legs (sensor DERIVED-now,
actuator = run-2 lever); (7) build-timing: T0+T1 live as OBSERVERS on the v7.5 launch, offline-against-
checkpoints retained as backup + run-2 A/B substrate. The Jacobian conditioning is the INSTRUMENT; the
earlier-engage optimality is a HYPOTHESIS the run-2 resume-A/B MEASURES.
