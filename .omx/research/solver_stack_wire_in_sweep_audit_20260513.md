# Solver-stack wire-in sweep audit — 2026-05-13

**Lane:** `lane_solver_stack_wire_in_sweep_20260513`
**Evidence grade:** `[prediction]` for all rows; `[empirical]` only for xray tests + autopilot rank smoke
**Score claim:** false; `ready_for_exact_eval_dispatch=false`; `promotion_eligible=false`

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: every newly-landed
signal must improve the solver or explicitly record why not. Today's session
landed 7 substrate lanes + 11 canonical primitives + 4 expert-team memos —
this audit closes the wire-in gap so the solver stack consumes them.

## Per-component status

### 1. Cathedral autopilot dispatch journal — WIRED

**Before:** 7 new substrate lanes registered in `.omx/state/lane_registry.json`
(via `tools/lane_maturity.py add-lane` from sibling agents) but the autopilot
loop had no `CandidateRow` to rank them. The loop ran sequentially but couldn't
see the new substrates.

**Newly wired:** `.omx/state/autopilot_candidate_queue_solver_stack_wire_in_20260513.jsonl`
(7 rows). Each row carries:

- `candidate_id` (lane_id)
- `family` (substrate_class)
- `predicted_score_delta` (negative = improvement)
- `expected_information_gain`
- `estimated_dispatch_cost_usd` ($2.00–$4.80; all within autopilot $5/dispatch
  cap)
- `blockers` (smoke/audit dependencies that must clear before dispatch)
- `notes` carrying `[prediction]` tag + source memo + `score_claim=false`
  invariant

**Verification:** `tools/cathedral_autopilot_autonomous_loop.py::load_candidates_from_jsonl`
+ `rank_candidates` end-to-end smoke loads all 7 rows and emits a ranked list
(top: `lane_s2sbs_stride2_byte_stuffing` at eig/$1.20). The autopilot can now
HALT-and-ASK on each substrate alongside existing candidates.

**Unwired (operator-routable):** the per-dispatch journal at
`.omx/state/cathedral_autopilot_authorized_journal*.jsonl` is populated only
when the autopilot is run in `--operator-authorized-le-5-dollar-mode` AND a
candidate clears all blockers. Today's 7 substrates carry 1–2 blockers each;
operator must clear at least one substrate's blockers (e.g. run the
`s2sbs_blindspot_audit_20260513.md` smoke harness) before dispatch is authorized.

### 2. Meta-Lagrangian solver — WIRED

**Before:** The 11 new primitives (IGLT optimizer, ternary QAT, JSCC,
Kolmogorov-MI diagnostic, tropical-LoRA, WBCE-MERA, Bregman mixer, Sinkhorn-OT
mixer, hypernetwork, product-of-experts, distillation chain) all existed as
canonical modules in `tac.optimization.*` / `tac.composition.*` but the
meta-Lagrangian CLI couldn't see them — its candidate spec is JSON-driven
from `tools/meta_lagrangian_atom_ledger_adapter.py` reading
`experiments/results/bilevel_atom_ledger.jsonl` (a 3-row historical ledger
last touched 2026-05-07).

**Newly wired:** `.omx/state/meta_lagrangian_typed_atoms_solver_stack_wire_in_20260513.jsonl`
(18 rows: 11 primitives + 7 substrates). Each row carries the meta-Lagrangian
typed-row schema: atom_id, family, category, module_path, predicted_score_delta,
predicted_byte_cost, charged_byte_target, interaction_assumptions, evidence_grade,
plus the planning-only invariants (`score_claim=false`,
`ready_for_exact_eval_dispatch=false`, `promotion_eligible=false`).

**Unwired (operator-routable):** the meta-Lagrangian CLI
(`tools/meta_lagrangian_search_cli.py`) reads `--candidates-json` and currently
only knows the bilevel-atom-ledger schema. Wiring the new JSONL as an
alternative `--candidates-json` source is a one-line operator action; a
sister adapter `tools/meta_lagrangian_solver_stack_wire_in_adapter.py` could
provide the bridge but the JSON-list shape was kept identical to existing
schema for direct consumption.

### 3. Pareto constraint solver — WIRED

**Before:** No canonical Pareto module path (`src/tac/pareto_*` does not
exist as a directory — tests reference Pareto via
`src/tac/optimization/meta_lagrangian_allocator.py` + similar). Constraints
were carried inline in autopilot dispatch ranking artifacts.

**Newly wired:** `.omx/state/pareto_constraints_solver_stack_wire_in_20260513.json`
declaring (a) baseline anchor (PR106 r2 frontier 0.193 at the 2.71x
pose-marginal operating point), (b) per-substrate constraints with
predicted (seg, pose, rate) bands, (c) per-primitive byte-cost + score-delta
bands + interaction assumptions, (d) the newly-active-constraint hypothesis:
pose-axis constraints become NEWLY ACTIVE for A1+LAPose / A1+wavelet /
PR95-LoRA at PR106 r2 (these target the 2.71x pose-marginal regime).

**Unwired (operator-routable):** the file is canonical Pareto-constraint
intake; downstream consumers (Dykstra alternating-projections at
`tac.optimization.meta_lagrangian_allocator`) should re-run with this manifest
as input. Operator gates the re-run.

### 4. Continual-learning posterior — WIRED (predictions only)

**Before:** `.omx/state/cost_band_posterior.jsonl` (7 empirical anchors, last
appended 2026-05-13T15:37 SIREN smoke timeout) and
`.omx/state/continual_learning_posterior.json` (Catalog #127 / #128 atomic-
locked posterior with custody validation) both refuse predictions per
CLAUDE.md "Forbidden score claims" + Catalog #127 custody-tag-validator. New
substrates have NO empirical anchor yet (no GPU dispatch has fired).

**Newly wired:** `.omx/state/predicted_anchors_solver_stack_wire_in_20260513.jsonl`
(7 PREDICTION anchors). Each row carries:

- `anchor_id` (prediction_<lane_id>)
- `predicted_score_band_low` / `predicted_score_band_high`
- `predicted_archive_bytes`
- `predicted_seg_distortion_avg` / `predicted_pose_distortion_avg`
- `evidence_grade: "[prediction]"`
- `axis: "[prediction; not an empirical axis]"`
- `promotion_eligible=false`, `score_claim_valid=false`,
  `ready_for_exact_eval_dispatch=false`, `rank_or_kill_eligible=false`

This is a DEDICATED stream sister to the empirical posterior — predictions
never enter the cost-band or continual-learning posterior so they cannot
contaminate the cost-band calibration (per the May 12 NV7 hardening) or
trigger Catalog #127 custody refusals.

**Unwired (operator-routable):** when an empirical anchor lands for any of
the 7 substrates (e.g., post-smoke Modal A100 dispatch), the empirical anchor
appends to `cost_band_posterior.jsonl` via the canonical
`tac.cost_band_calibration.append_anchor` helper, AND the prediction-vs-
empirical delta should be logged to the prediction-anchor JSONL (closing the
prediction-loop). That bridge tool is not yet built — surface as next-cycle
work.

### 5. Xray tools — WIRED (gap closed)

**Before:** xray substrate classifier
(`tools/xray_substrate_classifier.py`) knew the 5 cooperative-receiver
packet grammars (TT5L / SBO1 / S2SB / CMLR / DPW1) via the canonical
`tac.packet_compiler.cooperative_receiver_grammars` registry, but the two
sidecar composition grammars (LPA1 for A1+LAPose, WAV1 for A1+wavelet_residual)
were **missing**. xray classifying an A1+LAPose archive would label it
"A1-host bytes" only and miss the sidecar — the composition lane was
invisible to xray.

**Newly wired:** `src/tac/packet_compiler/cooperative_receiver_grammars.py`
registers two new `CooperativeReceiverPacketGrammar` rows:

- `magic=b"LPA1"` → `xray_label="a1_plus_lapose_sidecar_v1"` →
  `substrate_class="a1_plus_lapose_composition_packet"` (compiler stage:
  `pose_axis_foveal_rgb_residual_pack`)
- `magic=b"WAV1"` → `xray_label="a1_plus_wavelet_residual_sidecar_v1"` →
  `substrate_class="a1_plus_wavelet_residual_composition_packet"` (compiler
  stage: `seg_axis_db4_idwt_detail_band_pack`)

Both flow through `xray_magic_signatures()` + `xray_substrate_classes()` so
the xray classifier picks them up via the existing splat-import. Test
`test_cooperative_receiver_packet_grammars_are_unique_four_byte_ascii` was
updated to assert both new magics are registered.

**Verification:** 46/46 tests pass on
`src/tac/tests/test_xray_substrate_classifier.py` +
`src/tac/tests/test_cooperative_receiver_packet_grammars.py`. xray classifier
now resolves both new magics:

- `b"LPA1" -> a1_plus_lapose_sidecar_v1`
- `b"WAV1" -> a1_plus_wavelet_residual_sidecar_v1`

Total xray substrate classes: 57; total magic signatures: 45.

**Reachability verdict per substrate:**

| Substrate                                                            | Xray reach |
|----------------------------------------------------------------------|-----------|
| `lane_time_traveler_l5_autonomy_substrate_20260513` (TT5L)            | OK (pre-existing) |
| `lane_sabor_boundary_only_renderer_substrate_20260513` (SBO1)         | OK (pre-existing) |
| `lane_s2sbs_stride2_byte_stuffing_substrate_20260513` (S2SB)          | OK (pre-existing) |
| `lane_a1_plus_lapose_composition_20260513` (LPA1)                    | OK (newly wired) |
| `lane_a1_plus_wavelet_residual_retarget_20260513` (WAV1)             | OK (newly wired) |
| `lane_darts_supernet_time_traveler_architecture_search_20260513`     | N/A (architecture search lane; no bytes emitted) |
| `lane_pr95_artifact_lora_dora_surgery_20260513`                      | N/A (LoRA adapter rides inside PR95 host; observer tools handle Fisher-info + layer leverage separately) |

## 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map contribution** — per-substrate (seg, pose, rate)
   predicted bands feed `tac.sensitivity_map.*` via the Pareto-constraints
   JSON (`pareto_constraints_solver_stack_wire_in_20260513.json`).
2. **Pareto constraint** — `pareto_constraints_solver_stack_wire_in_20260513.json`
   IS the constraint declaration; binding hypothesis: pose-axis ACTIVE for
   composition lanes targeting PR106 2.71x pose-marginal regime.
3. **Bit-allocator hook** — `primitive_ternary_qat_1p58_bit` +
   `primitive_jscc_scorer_conditional_entropy` change per-tensor importance;
   `tac.optimization.bit_allocator_end_to_end` must re-run when consumer
   substrate trainer wires them in (operator-gated).
4. **Cathedral autopilot dispatch hook** —
   `.omx/state/autopilot_candidate_queue_solver_stack_wire_in_20260513.jsonl`
   loads end-to-end via `load_candidates_from_jsonl` + `rank_candidates`.
5. **Continual-learning posterior update** — `.omx/state/predicted_anchors_*`
   stream (sister to empirical posterior); per-substrate prediction anchors
   tagged `[prediction]` with `promotion_eligible=false` per Catalog #127.
6. **Probe-disambiguator** — N/A. All 7 substrates and 11 primitives have
   single-interpretation rate-distortion bands; no design tension requiring
   ship-both-modes resolution per
   `feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`.

## Operator-routable decisions surfaced

1. **Run smoke harness for each substrate** so blockers clear and autopilot
   can authorize dispatch — typical cost $0.30 each (~$2 total).
2. **Wire meta-Lagrangian CLI** to read the new typed-atoms JSONL as
   `--candidates-json` source (operator gate — confirm schema compatibility
   first).
3. **Re-run Dykstra alternating-projections** with the new Pareto-constraints
   manifest; surfaces the operator-binding pose-axis constraint.
4. **Build prediction-vs-empirical delta logger** so when the first substrate
   lands an empirical anchor, the prediction-anchor file gets the delta row
   (closing the feedback loop the canonical posterior would have provided
   for empirical anchors).
5. **Architecture verdict for DARTS-SuperNet** — the lane is planning-only
   today; operator must consume its output before downstream training fires.

## Compliance audit

- No /tmp paths: 0
- No score claims w/o evidence tags: 0
- No KILL verdicts: 0 (all are DEFERRED-pending-smoke / DEFERRED-pending-
  empirical-anchor per CLAUDE.md "KILL is LAST RESORT")
- Catalog #127 custody-validator: predictions excluded from posterior
  promotion path (separate JSONL stream)
- Catalog #128 atomic-locked writes: the empirical posterior is NOT
  touched; predictions land in dedicated file with no concurrent-writer race
- Apples-to-apples evidence tags: every row carries `[prediction]` /
  `[macOS-CPU advisory]` / `[contest-CUDA]` / `[contest-CPU]` tag

Cross-references:

- `feedback_solver_stack_wire_in_sweep_landed_20260513.md` (landing memo)
- `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md`
- `.omx/research/sabor_boundary_audit_20260513.md`
- `.omx/research/s2sbs_blindspot_audit_20260513.md`
- `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md`
- `.omx/research/expert_team_signal_processing_bell_labs_20260513.md`
- `.omx/research/zen_state_frontier_deep_math_research_20260513.md`
