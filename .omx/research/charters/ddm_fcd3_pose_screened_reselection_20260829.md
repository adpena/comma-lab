# ddm_fcd3_pose_screened_reselection — pose-screened re-selection of the fcd1 GT-benefit pool (task #1320, owning memo `ddm_fcd2_distortion_legs_execute_20260829.md`; parent family #1295, owning memo `ddm_fcd1_field_for_coder_diagonal_20260829.md`)

## MANDATE

Operator 2026-08-29 (standing GO, verbatim): *"do whatever it takes and work for as long as it
takes autonomously with full authority and standing go to accomplish frontier score lowering...
feel free to be creative and weird and think divergently."*
Routed finding: fcd2 (`ddm_fcd2_distortion_legs_execute_20260829.md`, commit 1326458d5b) refused
the fcd1 union at the pose gate — fresh candidate-bound compensation left d_pose at
2.7348e-4 vs base 6.3657e-6 (42.96×), with the carrier railed at signed-int12 (demand ~90,109
code units) and 597/600 refinement rows at `no_improving_step`. The −3,729 B rate credit is real
and unclaimable with the current object. fcd2's LIVE-HYPOTHESIS 3 names this arm's move:
**pose-aware re-selection** — keep only the edits on pairs whose compensated pose is provably
clean, sacrifice part of the byte credit, and clear the gate by construction. The registered
batch folded orders stay FOLDED (their triggers did not fire; do not fire them from this arm).
MAIN's screening arithmetic (DERIVED, recorded in task #1320, owning memo
`ddm_fcd2_distortion_legs_execute_20260829.md`): the richer-carrier route is dominated —
widening 7,200 carrier values int12→int16 ≈ +3,600 B against the 3,729 B credit — so
re-selection is the family's best-EV live route. That arithmetic is a static screen, NOT a
closure; if this arm's screen refutes re-selection, the richer-carrier route re-ranks.

## SCOPE

1. **$0 SCREEN from retained artifacts only** (no new solves): join fcd2's per-pair banks under
   `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/fcd2_distortion_legs/union/schur/`
   (BASELINE.json per-pair uncompensated deltas · GN shard banks · refinement shard banks — take
   the per-pair MIN over banks exactly as fcd2's close did) with the benefit-pool npz
   (`retained/coordinates/`, sha cc09fd9d…) → one table per pair: {edit positions, uncompensated
   Δd_pose, best compensated d_pose, stop reason, per-pair base d_pose}. Report the
   concentration curve (what fraction of total pose excess sits in the worst k pairs) — the
   pose-variance law predicts a heavy tail.
2. **Pre-registered screen ladder**: rung r keeps pairs with best-compensated per-pair pose ≤
   per-pair base + τ_r, τ ∈ {1e-8, 1e-7, 1e-6} (always keep the 8 measured-improved pairs; drop
   pair-level ties conservatively). Record per rung: pairs kept, B-positions kept (of 5,268).
3. **REAL joint re-encode per rung** (`token_rate_model_direction_dependence_v1` +
   `greedy_set_average_vs_marginal_price_v1`: never additive/entropy credit): build the screened
   field (edits only on kept pairs) and re-encode through the fcd1 prepare path. If the tool
   lacks a pair-subset filter, extend it minimally (2 review passes) with the identity control:
   the full-set filter MUST reproduce the union archive sha c45ab4e6… byte-identically.
   Byte-close each rung vs the jt21 base (180,192 B, sha ec0dd68f…).
4. **Fresh Schur chain on the best rung** (best = max projected net at 6.658e-7 S/B with
   pose-clean-by-construction pairs; carried per-pair numbers are SCREENING evidence only —
   publish evidence must be a fresh candidate-bound solve per the qs4 law): baseline → GN
   (fcd2's proven 5×120 disjoint shard pattern) → diminishing-returns refinement → close ×2
   byte-identical → publish gate `d_pose_after <= 6.3656845167356244e-6 + 1e-8` (in-code; a
   refusal is a MEASURED verdict, record it and descend one rung if the ladder allows).
5. **On a published rung** (fires only on repeat-identical publish): n600 frozen scorers on the
   jt21 base AND the screened body (the base leg has never been fired — run it), recompute S
   FROM COMPONENTS (the #877 S-from-components law as practiced in
   `ddm_fcd2_distortion_legs_execute_20260829.md` §Full scorer table), net ΔS = Δseg + Δpose +
   Δrate vs the ±3.5e-6 canonical band; admitted
   → canonical seal (`tools/make_candidate_seal.py`, dual-axis; single-axis waiver reason per the
   #1152 fire-tool contract, owning memo `ddm_br1_pose_basis_reorientation_20260819.md`) + READY
   fire-order into the consumer store — **MAIN retains Modal dispatch under single-flight**.
6. **Typed exit**: per-rung table {pairs kept, bytes vs base, publish verdict, realized legs
   where fired, net ΔS or refusal}; family-counter status (this arm neither fires batches nor
   claims #1295 closure — say so plainly).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; all receipts/payloads to the EXISTING consumer store
  `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/` under a new `fcd3_*` subtree
  (do NOT fork a new store; fcd2's NEXT_IF_RESUMED names this store; ~30 GiB free measured).
- Axis honesty: every local row `[macOS-CPU frozen-scorer advisory]`, score_claim=false,
  promotable=false; ONLY a MAIN-fired T4 row is authority.
- Claim your own scorer lane via `tools/claim_lane_dispatch.py` before any n600 leg; the r10/fcd2
  claims are terminal — do not touch them.
- Use the REPAIRED splice/publish tooling as landed by fcd2 (commit 1326458d5b) — do not
  re-derive rider order; the regression test pins it. Grep BOTH argparse surfaces before
  emitting any flag (never invent).
- Fresh compensation ONLY (`ddm_qs4_collateral_suppression_20260813.md`, task #1039): screening
  reads retained numbers, publishing requires the fresh solve on the exact screened object.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- fcd2 (`ddm_fcd2_distortion_legs_execute_20260829.md`): exact union + current carrier CLOSED at
  INSTANCE scope (42.96× base, repeat-identical); more refinement CLOSED (mean moved 2.79e-6, no
  budget stop); RR5/DX2-as-plain-Rice CLOSED as apparatus defect (repaired + regression);
  scoring/sealing/dispatching a publish-refused archive CLOSED by ordering contract.
- fcd1 (`ddm_fcd1_field_for_coder_diagonal_20260829.md`): same-move compensation dead 45.18×;
  entropy/average/additive-credit pricing dead; B/H token labels ≠ realized SegNet flips —
  realized seg is measured only through the scorer legs, never inferred.
- qs5 (`ddm_qs5_verdict_and_no_toy_enforcement_20260813.md`): in-compile compensation PROVEN
  reach on a 3-pair object — transfer to ANY new object must be re-solved, never carried.
- dg2 (`ddm_dg2_diagonal_distortion_verdict_20260824.md`): uncompensated diagonal refused 686×,
  pose 93.3% of damage — the reason pose-cleanliness is the screen's primary axis.
- tv1/tv2 co-location genus (task #1253, owning memos `ddm_tv1_evaluator_tolerance_curve_20260824.md`
  + `ddm_tv2_evaluator_tolerance_curve_20260824.md`): seg slack and pose damage can share
  support — the falsifier below names this genus as the refutation shape.

## OPTIMAL FORM

- Family exemplar: fcd2's landed chain is the reference — memo
  `ddm_fcd2_distortion_legs_execute_20260829.md` + commit 1326458d5b, reference receipts
  `fcd2_distortion_legs/union/schur/baseline/BASELINE.json` (sha 188f832c…) +
  `close_refined/CLOSE.json` (sha b8275253…) + `publish_refined/PUBLISH.json` (sha f849374f…)
  in the consumer store. This arm runs the SAME landed chain on screened objects.
- SCOPE reductions declared: pair-granularity screening (position-level pose-Jacobian screening
  is the NAMED successor if pair-level fails, not part of this arm); solve population may be
  restricted to edited pairs ONLY if the tool's existing contract supports it with byte-identity
  proof on untouched pairs — otherwise full 600. MECHANISM reductions FORBIDDEN (no surrogate
  scorers, no MLX pose authority — MLX-PoseNet drift 0.55% rel; no entropy pricing).
- **PRIOR-LAW PREDICTION (falsifiable):** the pose-variance law (13.4× spread,
  `ddm_na10_negative_audit_fresh_laws_20260819.md`) + the sa1 linear-in-mass damage law (0.91×
  of linear, recorded in `ddm_iv1_inversion_pose_actuator_20260818.md`) predict pose excess
  concentrates in a pair
  tail, so SOME rung retains ≥ ~1/3 of the byte credit (≥ ~1,200 B ≈ −8e-4 S rate) at
  gate-clean pose. FALSIFIER: every rung that clears the pose gate retains < 45 B of real
  re-encoded credit (below the ~30 B / 1e-5 S fire bar with margin) — that refutes pose-screened
  re-selection at FORMULATION scope on this body (pose damage and byte credit co-located, the
  tv1/tv2 genus), re-ranks the richer-carrier route, and leaves the batch ladder / #1295 closure
  as the family's remaining moves. Count it plainly if it lands.

## DELIVERABLE

`.omx/research/ddm_fcd3_pose_screened_reselection_20260829.md` — typed rows: (1) the per-pair
screen table summary + concentration curve; (2) per-rung {pairs, positions, REAL re-encoded
bytes, projected rate ΔS}; (3) fresh Schur publish receipt(s) w/ repeat identity; (4) realized
n600 scorer table + S from components + band adjudication where fired; (5) seal + fire-order for
MAIN OR the per-rung refusal table w/ the falsifier counted; (6) NEXT_IF_RESUMED +
LIVE-HYPOTHESES + DEAD-ENDS. Commit via the serializer. End with the own-vehicle frontier line.
