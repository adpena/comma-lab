# ddm_qbt2b_r7_lane_constrained_margin — build the lane-CONSTRAINED margin law (λ_Lane/λ_Movable primal-dual) + sealed r7 config from the r5-born basis; routed by `ddm_qbt2b_r6_born_field_margin_verdict_20260828.md` §5

## MANDATE

Operator 20260821 (standing): *"do whatever it takes and work for as long as it takes autonomously
with full authority and standing go to accomplish frontier score lowering."* Routed finding: r6
(memo `ddm_qbt2b_r6_born_field_margin_verdict_20260828.md`, commit 5463d7af61) proved the margin
law WORKS on a born field (flip 0.0945→0.00972, 9.7×; Road prior-shift cured 25.77%→0.39%) but the
UNWEIGHTED expected-flip objective RE-ERASED the born Lane (werr 9.80%→99.81%) and eroded Movable
(0.65%→10.85%) — the exact σ_cc′/MCF-thin-structure collateral predicted by
`sigma_ccprime_build_20260709.md` (task #382). The weight-family 2×2 is
closed at both extremes (r5 balanced-CE prior shift · r6 unweighted-flip erasure): the resolution
is a CONSTRAINT, not a weight. This arm builds the constrained margin stage NOW because the Metal
slot is free, the born init is banked (sha 4b40acc5…), and every other qbt2b leg is gated on it.

## SCOPE

1. CONSTRAINT LAW (trainer edit, `experiments/ddm_qbt1_qbflow_trainer.py`, margin stage
   `stage_03_joint_boundary_interior_birth`): per-class primal-dual protection — dual ascent
   λ_c ← max(0, λ_c + η_λ·(realized werr_c − bound_c)) for c ∈ {Lane, Movable}, penalty
   λ_c · (per-class expected-flip restricted to class-c target pixels) added to the margin
   objective. Config-gated (new mode field, default = legacy unconstrained, BYTE-IDENTICAL off);
   bounds/η_λ DERIVED with provenance from the r5/r6 measured endpoints (Lane born at 9.8–12.0%
   werr; existence floor 0.50 hard; do NOT guess constants — cite the measurement each value
   comes from). λ trajectories + per-class werr MUST be emitted in per-step history — the
   binding-vs-inert proof (the #404 telemetry law, worked example in
   `ddm_qbt2b_r6_born_field_margin_verdict_20260828.md` §2) for MAIN's endpoint read.
2. VALIDATOR PINS per the r6 precedent: the (mode, bounds, η_λ) tuple pinned as a GROUP in
   `validate_config` (isclose abs_tol 1e-12); inconsistent combinations refuse; legacy configs
   without the field validate unchanged. New fields participate in config_identity automatically
   (verify the exclusion list is untouched) ⇒ cross-mode resume refused.
3. TESTS mirroring `test_birth_event_mode_pins_mode_threshold_pair` (experiments/tests/
   test_ddm_qbt1_qbflow_trainer.py): default-off identity · pair-pin refusals both directions ·
   dual-ascent unit behavior (λ rises when werr > bound, decays to 0 when under) · legacy config
   validates. Full suite green, ruff clean.
4. r7 AUTHORIZED CONFIG: compile through the real compiler (constrained mode ON, birth_event_mode
   existence_majority, birth_max_steps 20, margin_steps 5000, initialization_state =
   `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r5/initialized_r6_from_r5_cap_ema_state.pt`
   sha 4b40acc584546be39839bba7490c9e1ae53286f73a038d30c2070ec27b6d700b — SAME born basis as r6 ⇒
   r7 vs r6 is single-variable by construction), validate both directions incl. JSON round-trip,
   storage projection per the r6 precedent (AP free ~34 GB vs 8.59 GB floor), write to
   `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/AUTHORIZED_N32_R7_5020_20260828.json`
   + print its sha256. NO LAUNCH from the arm — MAIN owns the governed fire (mirror the r6
   argv, counter-693 launch_manifest).
5. Bounded n1 smoke (few steps, constrained ON) proving the law is live (λ emitted, loss term
   nonzero when violated) + checkpoint/RNG/archive resume identity — the palette-birth arm's
   smoke pattern.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_qbt2b_r7_lane_constrained_margin/`.
- NO Metal/n600 launch from the arm; the smoke in scope 5 is n1/CPU-bounded only.
- File ownership: this arm owns `experiments/ddm_qbt1_qbflow_trainer.py` + its test file for the
  session; no parallel arm touches them (check `.omx/state/active_lane_dispatch_claims.md`).
- Advisory axis honesty: every number `[macOS frozen-scorer advisory]`, score_claim=false.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- r2 margin freeze at 0.2504 on an UNBORN field (`ddm_qbt2b_r3_ce_birth_verdict_20260827.md` +
  the R2 postmortem `ddm_qbt1_r1_r2_qbflow_verdict_20260827.md`) — now formally the
  unborn-class pathology (r6 positive control); do not re-add birth-side machinery here.
- r4 unweighted CE cannot birth Lane (werr 99.76 flat; `ddm_qbt2b_r4_extended_ce_verdict_20260828.md`).
- r5 balanced CE births Lane but prior-shifts Road 25.77% (log(w_c/w_c′) boundary shift;
  `ddm_qbt2b_r5_balanced_ce_verdict_20260828.md` §3) — WEIGHTS are the wrong instrument; do not
  implement the constraint as a class weight.
- r6 unweighted flip erases the born Lane 99.81% (`ddm_qbt2b_r6_born_field_margin_verdict_20260828.md`
  §4) — area-priced objectives trade the tail away; the aggregate metric cannot see it.
- lc1 per-edge label injection net-harmful at n32 (−12,884, all 32 pairs;
  `.omx/research/ddm_lc1_20260805/LC1_RECEIPT.md`) — protect via CONSTRAINT on realized werr,
  not via injected labels.

## OPTIMAL FORM

- Family exemplar: the r6 config-gated gate revision — commit 680579a15c is the reference
  implementation for this trainer's law pattern (mode enum + BIRTH_EVENT_MODE_THRESHOLDS map +
  validator pair-pin + mirrored tests); receipt: `experiments/ddm_qbt1_qbflow_trainer.py` at that
  sha + 18/18 tests. Constraint-law design sources: the lane-guard λ_Lane primal-dual +
  born-lane protection (`ddm_lg1_lane_guard_20260731.md`, built for TR1) — adapt the LAW,
  recall-not-reinvent; the σ_cc′ Γ-limit derivation (`sigma_ccprime_build_20260709.md`) for why
  scalar/area-priced terms MCF-erase thin structures.
- SCOPE reductions declared per row: n32 seeded-stratified (legal, the lineage's fixed cohort);
  n1 smoke (legality proof only, produces NO family verdict). MECHANISM reductions FORBIDDEN —
  the dual ascent must run on REALIZED werr (through render→R→uint8→SegNet argmax), never on a
  proxy field.
- **PRIOR-LAW PREDICTION (falsifiable):** the #382/#808 constraint law predicts the constrained
  margin stage holds Lane werr ≤ bound for the WHOLE 5,000-step window while total realized flip
  descends to within ~2× of r6's unconstrained 0.00972 (constraint cost bounded, not
  catastrophic). FALSIFIER: λ_Lane pinned at its ceiling with total flip ≫ 2× 0.00972 OR Lane
  werr breaching 0.50 despite active λ ⇒ the constraint set is INFEASIBLE at this
  capacity/basis ⇒ Lane exits in-field training and routes to the m131 analytic Lane-carrier leg
  (d3a lineage) — count it plainly if it lands.

## DELIVERABLE

`.omx/research/ddm_qbt2b_r7_lane_constrained_margin_20260828.md` — typed rows: constraint-law
design w/ per-constant provenance · test matrix results · smoke receipts (λ live + resume
identity) · sealed config path + sha256 · READY_TO_FIRE handoff block for MAIN (exact argv mirror
of counter-693, r6→r7 paths swapped). Commit via the serializer. End with the own-vehicle
frontier line.
