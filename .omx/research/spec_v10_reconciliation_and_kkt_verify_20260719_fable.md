# SPEC_v10 reconciliation + KKT-derivation fresh-eyes verification (2026-07-19, Fable arm)

**Role:** delegated fresh-eyes adversarial verification (Task 1) + reconciled-SPEC assembly (#521/#540 owed, Task 2).
**Authority:** read/derive/fix-artifacts only — no launch, paid dispatch, score, promotion, or pointer authority.
**Pointer:** `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED** — everything here is MEANS.
**Method:** per `docs/operating_manual_craft_handoff.md` — every check RE-DERIVED from primary artifacts
(gt_n600.npz, ladder/receiver/inverse-solve receipts, registry rows, source code), never trusted from the memo
under review. Every finding labeled MEASURED / DERIVED / INFERRED / ASSUMED.

## Verification targets

- (a) `.omx/research/v10_flattened_lagrangian_kkt_derivation_20260719.md` (the KKT memo)
- (b) `src/tac/canonical_equations/pose_plane_proximity_law_20260719.py` (the pose plane-proximity law)

## Verdict summary

**The derivation's SKELETON is sound** (coordinates, marginal closure, crossover, complementary-slackness
reading, channel anisotropy) — every constant I could re-derive checked out. **Four over-strong claims and one
arithmetic error are real defects**; all are FIXED (append-style corrections on the memo; direct edit +
follow-up commit on the law module) and folded into the reconciled SPEC. None of the defects changes the
strategic conclusion (solve-near-source + residual-vs-predictor bytes; pose expected slack); they change what
is THEOREM vs what is OPEN-EMPIRICAL-AND-PRE-REGISTERED.

## Findings

### V-1 — DEFECT (arithmetic): "~9 orders of slack" is wrong; the measured ratio is ~5.7 orders

- **DERIVED (re-derived here):** crossover `5/√(10·d) = 100 ⇒ d* = 2.5e-4` ✓ (the derivative algebra checks).
  Measured near-source per-row means `5.35e-10..1.14e-9`. Ratio `2.5e-4 / 5.35e-10 = 4.7e5` ⇒
  **≈5.7 orders of magnitude** (vs `1.14e-9`: `2.2e5` ⇒ ≈5.3). "~9 orders" is not obtainable from any
  reading of these numbers.
- **Blast radius:** the phrase propagated to (1) the law module ×3 (module docstring, classifier docstring,
  `one_line_summary`), (2) SPEC_v10 §15 item 1, (3) DAG `FEED-pose-falls-out` ("~9 orders of slack"),
  (4) the canonical-equations registry row for `pose_plane_proximity_corollary_v1` (the `equation_payload`
  embeds the stale `one_line_summary`).
- **Consequence check:** the conclusion ("slack is enormous") SURVIVES — half a million × is still enormous —
  but the number was fabricated-by-error and must not be re-cited.
- **FIX:** law module edited (this landing); reconciled SPEC carries ~5.7; correction recorded in the new DAG
  FEED (the old FEED is append-only history); **OWED to MAIN:** re-emit the registry row for
  `pose_plane_proximity_corollary_v1` from the corrected module via the canonical append path (the JSONL is
  append-only; this arm did not hand-append a row with a guessed schema).

### V-2 — DEFECT (byte accounting): "costs at most a rank-6 adjustment, O(6) numbers, not O(N) bytes" is over-strong

- **DERIVED:** the pose constraint restricts exactly 6 DOF (`rowspace(J)` is ≤6-dim) — that DOF statement is
  sound. But "O(6) numbers" as a BYTE claim silently assumes the receiver can expand a rank-6 correction
  `δ = Jᵀc` for stored `c ∈ R⁶`. It cannot: the rows of `J = ∂P/∂ŷ|_(y₀,y₁)` are (i) PoseNet-derived — no
  scorer weights at inflate time (strict-scorer rule), and (ii) evaluated at the SOURCE plane — video-derived,
  hence COUNTED if shipped (rule-118 boundary), at 6×N scale. So the correction is 6 DOF applied at ENCODE
  time; its BYTE cost is whatever the chosen description grammar charges for the perturbed description —
  an open empirical quantity, precisely the `bytes(τ_pose)` curve the VJP arm owes.
- **FIX:** appended correction on the KKT memo; reconciled SPEC states the DOF/bytes distinction.

### V-3 — FLAG (unearned genericity): "μ = 0 generically" is not earned; pose inactivity at in-band solutions is OPEN-EMPIRICAL and must stay pre-registered

- **DERIVED:** genericity ("a random residual has negligible projection onto a 6-dim subspace of 1.18M")
  does not apply here: seg-driven residuals concentrate on the boundary annulus (MEASURED: 96.9% of c2
  residual flips in-annulus, `witness_per_stage_attribution_20260719T020657Z.md` ep725 row), and PoseNet
  Jacobian rows also concentrate on structured/edge content. Two edge-concentrated objects are not in
  general position.
- **Evidence audit:** the near-source rows (5.35e-10..1.14e-9) are the trivial `r→0` limit — they instantiate
  `h(0)≈0` (continuity), NOT J-orthogonality of seg-shaped residuals. The ONLY measured large-`r` row (c2
  witness, plane RMSE 25) has d_pose 63 — i.e. the one real-world residual we measured has an ENORMOUS
  J-projection. Calling the measured rows "this theorem's instances" over-reads them. Whether IN-BAND
  positive-band-scale seg-shaped residuals keep a small J-projection is exactly the bindingness question the
  law module correctly PRE-REGISTERS — it is a falsifiable prediction, not a theorem instance.
- **FIX:** appended correction on the KKT memo ("generically" → conditional + pre-registered); law module
  already carried the pre-registration (kept, sharpened).

### V-4 — FLAG (linearization validity): measured margin scale supports the structure; the Lipschitz radius itself is UNMEASURED custody

- **MEASURED (this verification, gt_n600.npz `margins`, 117,964,800 plane pixels):** median 5.89, mean 5.61,
  q(0.01) 0.359, q(0.001) 0.0355, max 16.24; low-margin fractions: 2.67% < 1.0, 1.38% < 0.5, 0.28% < 0.1.
  The annulus is a few % of pixels — this CORROBORATES the memo's complementary-slackness reading (bytes
  concentrate where constraints are tight) and matches the 96.9%-of-flips-in-annulus attribution row.
- **DERIVED:** head-space flip distance `d = m/‖Δw‖` with ‖Δw‖ ∈ [2.60, 4.01] (verified against
  `segnet_recursive_fractal_factorization_20260715.md`; SVs [3.128, 2.154, 2.025, 1.796] ✓): annulus pixels
  sit at `d < ~0.2` (small perturbations — first-order trustworthy where λ>0); median pixels at `d ≈ 1.5–2.3`
  where the nominal per-channel radii are large and validity rests entirely on (i) `Lip_local`, which is
  **CONFIGURED, not measured** (`MarginBandConfig` docstring; the #549 run measured ONLY the zero-band
  control — no positive-band operating point exists anywhere), and (ii) the `r_max` clip.
- **ADDITIONAL GAP (appended):** "exactly the per-channel box inner-approximation of this half-space system"
  is over-strong. The implemented box (`derive_hyperplane_channel_band`) budgets, per pixel, only that
  pixel's OWN 3 channels against its own hyperplane (triangle inequality over 3 terms). The true constraint
  at pixel p couples its whole receptive field (MEASURED ERF r50≈85px, r90≈300px — fractal memo):
  simultaneous in-box perturbations of many pixels can accumulate at p beyond its diagonal budget. The box
  is the inner approximation of each single-pixel DIAGONAL slab; JOINT validity across ~10⁵ overlapping
  constraints is an open empirical question — measured only by the positive-band bindingness/repair-rate
  harvest, with the frozen hard oracle as sole authority (both artifacts already say the oracle is final;
  only the word "exactly" over-claimed).
- **Noise floor:** pair-125 (MEASURED, #549 receipt): ONE native-f32 argmax flip at ZERO perturbation
  (d_seg 5.09e-6) — an fp32 ULP-tie floor sits beneath the linear model at every band scale; the registered
  f32 receiver-arithmetic admissibility law is load-bearing, not optional.

### V-5 — FLAG (convexity scope): correct as scoped, one quarantine item missing

- The memo scopes convexity "first-order in the residual" and quarantines (a) the Lipschitz validity radius
  and (b) the hard-oracle check ✓ — the true constraint set (argmax regions of a deep net) is indeed
  non-convex and the memo says so. **Missing item (c):** the DESCRIPTION variables are quantized/integer
  (`Q(·)` in the objective; integer numerators in the lattice) — the LP/QP claim holds for the continuous
  relaxation; the integer/repair leg is combinatorial and lives with the oracle, outside the convex program.
  Appended.

### V-6 — FLAG (over-strong): "waterfill is effectively TWO-term" is conditional on V-3, not settled

- With μ=0 the 2-term reading follows; μ=0 is the OPEN bindingness question. Honest form: **2-term PENDING
  the bindingness harvest**; if pose binds, it re-enters as a low-dimensional (rank-≤6) third term whose byte
  price is MEASURED (V-2), not assumed. The "flat at 1 admissible point" reading is consistent with the #549
  receipt's `INCONCLUSIVE_FLAT_OR_NOISY` instance-scope verdict ✓. Appended.

### V-7 — DEFECTS in the law module regime thresholds (FIXED by direct edit, follow-up commit)

1. **"~9 orders" ×3** → corrected to ~5.7 orders (V-1).
2. **POSE_FREE at `rmse < 1.0`:** all three supporting rows are EXACT realizations (plane RMSE ≈ 0). The
   <1.0 radius is an **ASSUMED design radius** (pre-registered, falsifiable), not a measured regime edge —
   previously the docstring cited the rmse=0 measurements as if they covered (0,1). Now labeled.
3. **FAR threshold `rmse ≥ 0.5·25.04`:** generalizes ONE instance (already instance-scoped in
   `domain_of_validity`) into a regime boundary — now labeled ASSUMED interpolation.
4. **`NEAR_SOURCE_DPOSE_MAX = 1.14e-9` is the worst per-row MEAN, not the worst observation:** observed
   per-pair maxima 3.69e-9 (n6 pair 424) and 2.04e-9 (#549 chunk max) EXCEED the constant while remaining
   inside the ×10 predicted band. Now labeled (constant kept; band unchanged).
5. **Constants cross-check vs `yhat_rd_ladder_20260719_codex.md` (all ✓):** FAR_PLANE_RMSE 25.044688 ✓ ·
   FAR_PLANE_DPOSE 63.031066895 ✓ · rung-B d_seg 0.003455480 ✓ · rung-A d_pose 1.144e-9 ✓ · crossover
   2.5e-4 ✓ (re-derived).

### V-8 — VERIFIED SOUND (no change)

`N = 384·512·3 = 589,824`, `2N = 1,179,648` ✓ · score-marginal closure `dS` formula and the 2.5e-4
crossover ✓ (re-derived) · rank-4 head SVs + Lane ‖Δw‖ up to 4.01 ✓ (fractal memo) · 96.9% annulus flip
share ✓ (attribution memo ep725) · ker(A) ≈ 52% ✓ (null-subspace memo) · direct-payload rate-death
1.70 MB/frame ✓ and 83,838 B generator rung ✓ (lattice + ladder memos) · chroma-cheap-for-pose via the
2×2 box ✓ (frozen factorization) · band-law formula in the memo matches the implementation
(`r_ch = min(r_max, d_feat/(3·Lip·|q_ch|))`, `d_feat = scale·m/‖Δw‖`) ✓ · the annulus-concentration
prediction is corroborated by the measured margin distribution (V-4).

## Fixes landed by this arm

1. **KKT memo:** appended `## 2026-07-19 fresh-eyes verification corrections (append-only)` — V-2..V-6
   corrections; no silent rewrites; original text preserved.
2. **Law module:** direct edit (follow-up commit): ~5.7-orders fix ×3; ASSUMED labels on both regime
   thresholds; per-row-mean vs per-pair-max label. Behavior of `pose_regime_from_plane_proximity` is
   UNCHANGED (thresholds kept — they are pre-registered design radii; only their labels were dishonest).
3. **Reconciled SPEC** (`SPEC_v10_capstone_RECONCILED_20260719.md`) — Task 2; carries all corrections.
4. **Old SPEC:** dated SUPERSEDED pointer paragraph prepended (no body mutation).
5. **Canonical-doc registry:** `spec_v10_capstone_cold_start_seeded` marked superseded_by the reconciled
   doc; new entry added (per the #533 registry contract).
6. **DAG FEED** appended recording the reconciliation + the V-1 correction.

## OWED to MAIN (routing)

1. **Registry-row re-emit** for `pose_plane_proximity_corollary_v1` (V-1: the appended JSONL row's
   `equation_payload.one_line_summary` still says "9 orders"); append a superseding row from the corrected
   module via the canonical append path — this arm did not hand-append.
2. **Task #521 status:** the reconciled-SSoT deliverable (state-review BUILD item 1 / #540's "reconciled
   SPEC" leg) is now LANDED pending MAIN review; #521 can move to reviewed/complete after MAIN reads the
   reconciled SPEC. MAIN updates the ledger (this arm has no ledger authority).
3. **Bindingness harvest priority unchanged:** V-3/V-4 make the positive-band bindingness + `bytes(τ_pose)`
   curves THE decisive owed measurements — nothing in this verification de-prioritizes the running VJP arm.

## STORES CONSULTED

`CLAUDE.md` · `docs/operating_manual_craft_handoff.md` · gt_n600.npz (margins, MEASURED here) ·
`v10_flattened_lagrangian_kkt_derivation_20260719.md` · `pose_plane_proximity_law_20260719.py` ·
`yhat_rd_ladder_20260719_codex.md` (+ JSON/CSV) · `v10_lattice_rate_verdict_and_composition_20260719.md` ·
`v10_capstone_state_review_20260719_codex.md` · `joint_seg_pose_inverse_solve_20260719_codex.md` (#549) ·
`production_receiver_543_20260719_codex.md` (#543) · `segnet_recursive_fractal_factorization_20260715.md` ·
`frozen_scorer_exact_factorization_20260715.md` · `witness_per_stage_attribution_20260719T020657Z.md` ·
`null_subspace_rate_measure_20260717.md` · `src/tac/optimization/joint_seg_pose_rate.py` (read-only) ·
`.omx/state/canonical_equations_registry.jsonl` · `.omx/state/canonical_doc_registry.json` ·
SPEC_v10 base §0–§15 · DAG tail FEEDs · task #533/#521 JSON.

**Pointer delta: 0. MAIN landing review required.**
