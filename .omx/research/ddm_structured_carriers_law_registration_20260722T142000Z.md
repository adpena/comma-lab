---
schema: ddm_structured_carriers_law_registration.v1
date_utc: 2026-07-22T14:20:00Z
task: 540
master_task: 578
feeds_tasks: [603, 613]
lane_id: ddm_structured_carriers_law_registration
research_only: true
execution_allowed: false
score_claim: false
promotion_eligible: false
main_landing_review_required: true
---

# DDM structured-carriers law registration

## Outcome first

Registered one canonical equation with three SHA-bound measured anchors:

`ddm_describe_line_rate_distortion_bracket_v1`

| leg | measured receiver-closed endpoints | scoped disposition |
|---|---|---|
| v7 exact values | n64 `43,112,153 B / d_seg 0.000171422958 / d_pose 0.000081666650`; n256 `171,332,654 B / 0.000154534976 / 0.000104117518` | evaluator-green, rate-dead at `215.560765x` and `856.663270x` the 200 KB falsifier; **FORMULATION** scope |
| v8 sparse post-hoc pixel values | tight masks cover `4.0182%-4.5302%` of sites, while the measured finite-tau ladder reaches `6.0744%`; exact bytes collapse `93.90%-94.54%`, yet tight-mask `d_seg` remains `0.025907576084-0.029359102249` | ERF/distortion-dead for this exact-value finite-tau formulation; learned, joint-solve, and faithful-base repairs remain open |
| v9 structured carriers | n64 `51,668 B / 0.045286496480 / 159.104827981350`; n256 `72,397 B / 0.040169219176 / 157.798907948748` | first measured in-box rate point, evaluator budget unspent; zero measured G2CS1 symbols; **INSTANCE/FORMULATION-open** |

The v9 n64-to-n256 exact-byte marginal is `20729/192 = 107.9635416667 B/pair`.
Linear interpolation gives `2628875/24 = 109536.4583 B` at n600, below 154.6 KB; this is
explicitly **DERIVED_FROM_MEASURED_N64_N256_NOT_MEASURED_N600**, not an empirical n600 row.

## Consolidated law and binders

The measured line brackets the still-unmeasured chart/event solve:

`structured v9 (rate green, distortion red) -> G2CS1 + xi events -> exact v7 (distortion green, rate red)`.

The v7 rate binder is not the evaluator binder. Road + Undrivable + MyCar opaque exact homes carry
`94.21%-94.73%` of exact bytes, while Lane + Boundary bind residual `d_seg`. The n256 q4 row still
costs `108,637,789 B` and misses the `0.00116` gate by `0.000058597094`.

The v8 mechanism converges with the prior regions-not-pixels law: measured SegNet ERF context is
`r50 approximately 50-160 px` with median approximately `85 px`, and `r90 approximately 206-424 px`
with an operational spill check near `300 px`. Pasting exact values at sparse sites over a
photometrically alien base omits that context. These ERF values are cited mechanism evidence from
the existing measured factorization, not a new anchor or a widened v8 family verdict.

The v9 receiver structurally forbids pixel RGB patches, consumes five region-coherent carriers,
keeps one Pose6 byte home, and exposes G2CS1 coefficient rerasterization. Its measured rows use zero
G2CS1 symbols, so the correction budget remains unspent. The primary live degree of freedom is the
joint Fisher-margin/curvature-ranked G2CS1 coefficient plus xi-transported birth/death-event solve,
admitted only through the corrected inner Jacobian, hard semantic cells, exact bytes, and Pose tube.

## Registry wiring

- Producers: `tools.run_direct_description_entropy_priced_member` and
  `tools.run_ddm_v9_carrier_compose`.
- Consumers: `tac.optimization.v10_constructive_solver` (the current v10 surface),
  `tac.optimization.direct_description_entropy_priced_member` (the #613 crux), and
  `tac.witness_control.costate_organ_v2`.
- Equation callable: fail-closed three-leg classifier plus explicitly labeled derived byte
  interpolation; it emits no contest score and cannot promote advisory rows.
- Empirical residuals: the three zero residuals compare a real categorical law prediction against
  the receipt's scoped verdict. No numeric projection is mislabeled as empirical.
- Ledger history preserves the initial registration and one same-landing re-registration after the
  final binder/support metadata was added; latest-event reduction exposes the final payload.

No DSL lever is added: this landing consolidates settled measurements and has no launch authority.
The DAG leg is
`.omx/research/ddm_structured_carriers_law_registration_DAG_FEED_20260722T142000Z.md`; the equations
leg is
`.omx/research/ddm_structured_carriers_law_registration_canonical_equations_20260722T142000Z.md`.

## Canonical anti-pattern disposition

No v8 sister anti-pattern is registered. `tac.canonical_anti_patterns` represents recurring
class-level forbidden patterns, while the custodied v8 receipt is explicitly FORMULATION-scoped.
Promoting it to the negative registry here would silently widen the verdict. The existing
regions-not-pixels convergence remains mechanism context and a design guard; MAIN may register a
class-level anti-pattern only with a separately reviewed cross-formulation recurrence record.

## Blocker delta and pointer honesty

- Discharged: #540 measurement-line equations debt for the v7/v8/v9 bracket; nine receipt hashes;
  typed evaluator; producer/consumer routing; Catalog #344 JSON roundtrip; draft consumption links.
- Remaining: nonempty G2CS1 + xi-event solve, n600 empirical row, contest CPU/CUDA custody, candidate
  archive, score claim, and promotion.
- Optional and still unmeasured: Brenier pointwise monotone compander; Jones-splay/MTF coder rows wait
  for a nonempty receiver-admitted G2CS1 stream.

`0.1910828242 [contest-CPU]` is unchanged. MAIN must review the full branch diff before landing.

## Bounded re-derivation argv

```text
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python tools/list_canonical_equations.py --equation-id ddm_describe_line_rate_distortion_bracket_v1 --json
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest -q src/tac/canonical_equations/tests/test_ddm_describe_line_rate_distortion_bracket_20260722.py
shasum -a 256 .omx/research/ddm_v{7,8,9}*receipt*.json
```

## STORES CONSULTED

- Delegated authority file, verified SHA-256
  `4bd5e34e0d38ec8c45691353aba16501d6c2e51a3681b161eab583e062982a64`.
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`,
  `SPEC_v75_optimal_single_trunk_20260708.md`, and `SPEC_v8_perclass_decomposition_20260708.md`.
- All ten DDM canonical-equation drafts from measurement ladder through v8; v7/v8 landing memos,
  cross receipts, and six window receipts; v9 findings, equation note, DAG FEED, SHA receipt, and
  two measurement receipts.
- `segnet_recursive_fractal_factorization_20260715.md`; Brenier and Splay crosswalk notes;
  `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- `reports/latest.md`, `.omx/state/lane_registry.json`,
  `.omx/state/canonical_equations_registry.jsonl`, `.omx/state/subagent_progress.jsonl`, both
  delegated inboxes, and operator broadcasts through `2026-07-21T13:15:53Z`.
