# SNeRV LF payload rate-distortion reverse-waterfill planner — v1 (task #45/#46)

UTC 2026-06-09 · claude · subagent `snerv_lf_rd_waterfill_scaffold_20260609` ·
lane `lane_snerv_lf_rd_waterfill_20260609` · `[macOS-CPU advisory]` /
planning-control false authority → PROPOSAL ONLY (`promotable=False`;
`score_claim=False`; `ready_for_exact_eval_dispatch=False`; every ranked row
`requires_exact_remeasure=True`).

This memo documents the BUILD landing of `tools/snerv_lf_payload_rate_distortion.py`
+ reusable core `src/tac/optimization/lf_payload_rate_distortion.py` + 37
behavioral tests. It was BUILT + TESTED, NOT executed against real archives —
G1b is measuring section bytes now; the tool consumes its schema, does not wait
on it. Per the operator's task spec.

## THE LAW (the single predicate the tool computes per candidate action)

    keep payload component c  iff  -ΔS_distortion(c) > 25·Δbytes(c) / 37,545,489
    where ΔS_distortion = 100·Δd_seg + Δsqrt(10·d_pose)

This is SCORER-RESPONSE waterfilling, NOT pixel-variance. A payload section pays
rent only when the distortion it buys (in EXACT contest score units) exceeds the
rate it costs. The non-rate half (`ΔS_distortion`) uses the nonlinear pose term
`sqrt(10·d_pose)` on the TERM (not on raw d_pose) so the planner is correct at
the low-distortion operating point where pose dominates the marginal (CLAUDE.md
"SegNet vs PoseNet importance — operating-point dependent"). One currency with
the downstream exact waterfiller (the contest score), never re-derived.

## Why this is the right tool NOW (the G1b reframe)

Per `snerv_b_first_scorer_probe_verdict_20260609.md` +
`strategic_reevaluation_weeks_to_centuries_20260609.md`: SNeRV's LIVE surface is
already evaluator-CLOSE (d_seg≈0.0023 from ep0, a DESIGN property of store-LF).
Training is not the lever. The score-determining questions are DOWNSTREAM:
**export binding (G1b)** + **RATE** (the stored-LF payload's bytes; sister Z8
evidence: wavelet/LF storage was ~99.5% of the archive and the rate killer).
This planner is the RATE half: given the per-section byte decomposition + the
measured scorer atlas, it ranks what to DROP / QUANTIZE / RECODE so the smallest,
highest-value candidate set is exactly re-measured first — instead of throwing
unbudgeted bytes at the receiver.

## Input contracts (grep'd against the real schemas; no invented fields)

### (a) G1b verdict — `snerv_g1b_export_binding_verdict.v1`
Source: `/Volumes/VertigoDataTier/pact/snerv_mistake_b_g1a_20260609T201221Z/g1b_export_binding_measure.py`
(read-only). The CLI reads:
- `path_a_advisory.byte_decomposition.{lf_payload_bytes, linf_steps_payload_bytes,
  decoder_bytes, metadata_bytes, receiver_archive_header_bytes,
  archive_bytes_total_linf, archive_bytes_total_l2}` — the per-section rate
  decomposition.
- `path_a_advisory.archive_surface_distortion.{d_seg_mean_linf, d_pose_mean_linf}`
  — the receiver baseline (apples-to-apples archive surface; falls back to
  `independent_inflate_confirm.{independent_inflate_d_seg, independent_inflate_d_pose}`).
- `axis_tag` — carried through to every output row.

### (b) Atlas — `scorer_spectral_sensitivity.v2`
Source: `/Volumes/VertigoDataTier/pact/scorer_spectral_atlas*/scorer_spectral_sensitivity.v2.json`.
The CLI reads:
- `grid.{n_bands, channel_bases, rgb_channels, yuv_channels, orientations,
  frame_incidences, amplitudes_lsb}` — THE MEASUREMENT SCOPE ENVELOPE.
- `cells[].{band_index, H_seg, H_pose, channel_basis, channel, orientation,
  frame_incidence, amplitude_lsb}` — the per-cell measured sensitivity.
- `authority_tier`, `source_raw.path` — provenance.

Atlas headline (mechanism-grade): seg peak vertical/y band0 frame1_only
(H_seg≈0.0057); pose peak vertical/y band0 both_opposite (H_pose≈0.276, ~43×
the frame1-only pose incidence).

### (c) section-map (optional) — operator-supplied
JSON mapping each byte-decomposition section name → its atlas-grid
`CoefficientGroup` spec: `{band_indices, channel_basis, channel, orientation,
frame_incidence, amplitude_lsb, recodeable_floor_bytes, droppable}`. When omitted,
a CONSERVATIVE default places ONLY the LF payload (band 0, agnostic on the other
axes) and leaves every other section UNPLACED → scope-invalid → the only honest
move is an exact re-measure (we never guess a spectral identity).

### Output — `snerv_lf_payload_rd_plan.v1`
`{ranked_actions, not_paying_rent, needs_exact_remeasure,
section_sensitivity_estimates, best_action_id, best_value_per_byte,
total_predicted_bytes_freed, total_predicted_delta_score, inputs.unmapped_sections,
... + false-authority contract}`. The `CandidateActionEvaluation` PROPOSAL row
schema is `snerv_lf_payload_candidate_action_proposal.v1`:
`{action_id, action_kind, section_name, est_delta_d_seg, est_delta_d_pose,
delta_bytes, est_delta_distortion_score, est_delta_rate_score,
est_delta_score_total, value_per_byte, keep_section_under_law,
pays_rent_predicted, atlas_scope_valid, scope_reason, baseline_*, +
{authority, axis_tag, score_claim=False, promotion_eligible=False,
promotable=False, rank_or_kill_eligible=False,
ready_for_exact_eval_dispatch=False, requires_exact_remeasure=True}}`.

## The fail-closed scope rule (Catalog #385 sister, the measured-constant discipline)

A MEASURED atlas sensitivity is only valid INSIDE its `measurement_scope` (the
grid the atlas swept). A payload section whose coefficient group falls OUTSIDE the
envelope — a band the atlas never measured, an amplitude beyond the swept range,
an unmeasured channel/orientation/incidence — would force an EXTRAPOLATION. The
planner REFUSES: the action is `atlas_scope_valid=False`, its distortion estimate
is `None` (NOT zero, NOT a guess), and it is segregated into
`needs_exact_remeasure` — NEVER ranked above a scope-valid action. "Measured can
be cargo-cult too": measured-on-4-pairs-at-one-amplitude used globally is fragile;
this gate makes the over-extension impossible by construction.

## The route (downstream consumer; branch B/C)

This is the PROPOSAL surface. The DOWNSTREAM authority is
`tac.optimization.evaluator_action_waterfill.CandidateActionEvaluation` (exact
measured d_seg/d_pose/bytes vs a base archive; `pays_rent` iff exact ΔS<0). The
loop:

1. G1b lands `snerv_g1b_export_binding_verdict.v1` with section bytes + receiver
   baseline.
2. This planner ranks DROP/QUANTIZE/RECODE actions by predicted value-per-byte
   (THE LAW), smallest-highest-value-first; scope-invalid → needs_exact_remeasure.
3. The TOP ranked action is APPLIED to the payload + the receiver is EXACTLY
   re-scored → a `evaluator_action_waterfill.CandidateActionEvaluation` row.
4. If it pays rent exactly, accept; the base changes; re-run the planner (effects
   are noncommutative — `requires_recompute_after_accept=True`).
5. When a byte-closed candidate beats the frontier on dual exact eval → PR.

## The centuries framing (this is the V6 template)

Per `strategic_reevaluation_weeks_to_centuries_20260609.md`: the thousand-year
object is the proof-carrying evaluator-equivalent program compiler. This planner
is exactly its **allocation kernel under the measured law geometry**: it consults
the scorer's measured transfer function (the atlas) to allocate bytes where they
buy the most scorer-effect, and it carries every estimate as a typed PROPOSAL
that V3's exact ΔS-judge re-measures before admission. The same module generalizes
to ANY frozen evaluator whose sensitivity geometry has been atlas-measured and
whose payload decomposes into byte-accountable sections — the
`FrozenEvaluatorContract` + Evidence-Constitution shape. Nothing here is a
stopgap: the LF payload is the first real RD-allocation target, and the planner
is the reusable kernel that will allocate over every future carrier's payload.

## Files

- Core: `src/tac/optimization/lf_payload_rate_distortion.py` (commit `b28a15fe1`).
- CLI: `tools/snerv_lf_payload_rate_distortion.py` (commit `51d964cbb`).
- Tests: `src/tac/tests/test_lf_payload_rate_distortion.py` — 37 behavioral tests
  (commit `51d964cbb`).

## 6-hook wire-in declaration (per Catalog #125)

1. **Sensitivity-map** = ACTIVE — the planner CONSUMES the measured scorer
   spectral atlas (`scorer_spectral_sensitivity.v2`) as its per-section
   sensitivity prior.
2. **Pareto constraint** = ACTIVE — THE LAW is the rate/distortion Pareto
   admission predicate per section (keep iff distortion-bought > rate-cost).
3. **Bit-allocator hook** = ACTIVE (PRIMARY) — this module IS a reverse-waterfill
   bit allocator over the payload sections; its ranked actions are the allocation
   plan.
4. **Cathedral autopilot dispatch** = N/A — PROPOSAL surface; it ranks what to
   exactly re-measure, it does not dispatch (the downstream
   evaluator_action_waterfill row is the dispatch-eligible surface once measured).
5. **Continual-learning posterior** = N/A at build time — fires when real G1b +
   exact re-measure rows land (each accepted action updates the base; the planner
   re-runs).
6. **Probe-disambiguator** = ACTIVE — `atlas_scope_valid` / `scope_reason` IS the
   disambiguator between an atlas-estimable section and one that can only be
   resolved by exact re-measure (the fail-closed scope verdict).

## Authority / no-fake discipline

- Every output row carries the false-authority contract; nothing claims a score
  or is promotable. The atlas itself is `exact_cpu_advisory` /
  `mechanism_update_eligible` ONLY — nothing here updates the score roadmap
  (metric-laundering firewall).
- NO FAKE (Slot EEE Class 2): the 37 tests verify ACTUAL LAW behavior against a
  hand-derived known optimum (high-sensitivity-low-byte KEPT; low-sensitivity-
  high-byte DROPPED; scope-violation fails closed with `None` estimates;
  value_per_byte ordering; real-atlas adaptation places band 0 in scope). Every
  test would FAIL if the planner returned canonical constants instead of computing
  the law.

## The one design decision I most want challenged

**The section → atlas-grid `CoefficientGroup` placement is the load-bearing
assumption, and it is currently operator-supplied (or a conservative LF-only
default).** The atlas measures sensitivity in spectral-grid coordinates (band ×
orientation × channel × incidence × amplitude); the G1b byte decomposition is in
SECTION coordinates (lf_payload / step-maps / decoder / metadata). The mapping
between them — "which atlas cells does the LF wavelet blob's energy actually
occupy?" — is NOT measured by either artifact. I made the planner fail closed
(refuse to estimate unplaced sections) rather than guess, and I default-place
ONLY the LF payload at band 0 (the most defensible single placement). But a band-0
point placement for the WHOLE LF blob is itself a coarse approximation: the LF
payload spans multiple bands/orientations, and its TRUE distortion value is the
energy-weighted sum over the cells it actually occupies. The honest alternative is
to MEASURE the section→cell mapping directly (perturb each section's bytes, observe
the receiver d_seg/d_pose) — which is exactly the exact-re-measure the planner
defers to, but it means the planner's RANKING (not just its admission) leans on a
placement the atlas cannot validate. **Should the default refuse to place even the
LF payload (forcing a measured section-map always), or is the band-0 LF default a
reasonable prior that the exact re-measure corrects?** I chose the latter (a usable
default + fail-closed everywhere else) but flag it as the assumption most likely
to mislead the ranking before the section→cell mapping is itself measured.
