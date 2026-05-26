# MLX Candidate Contest-Equivalence Gate LANDED 2026-05-26

**Lane**: `lane_mlx_candidate_contest_equivalence_gate_20260526` L1
**Task**: SLOT-E harness landing (operator-approved "E. Build the contest-equivalence smoke harness")
**Evidence grade**: infrastructure (the gate ITSELF carries [macOS-MLX research-signal] non-promotable markers per Catalog #341)
**Cost**: $0 + ~130 s wall-clock per invocation (sister #1264 tool runtime + gate logic)

## Purpose

Lock the corrected #1258 methodology fix (empirical anchor 2026-05-26: `|S_MLX − S_PyTorch| = 0.000011`, **72× smaller than PR110 frontier delta**) into permanent infrastructure as a canonical PASS/FAIL gate for Path 3 candidate dispatch eligibility.

## Operational contract

Any Path 3 substrate-class-shift candidate (DreamerV3 RSSM / Z7-Mamba-2 / NSCS06 v8 chroma_lut / etc.) that's been MLX-trained AND exported to a contest archive via the #1251 export bridge + #1257 packaging cascade **MUST pass this gate before any paid CUDA dispatch is authorized**.

### Gate threshold rationale

Default `--gate-threshold-contest-units 0.001`:
- **90× margin** over empirical anchor (0.000011)
- **1.3× larger** than PR110 vs PR101 frontier delta (0.000789)

A candidate that FAILS this gate is **structurally unable to faithfully measure even a frontier-tightening score difference**. Failing the gate means the operator should NOT spend paid CUDA on it; instead audit the candidate's MLX↔PyTorch decoder parity per #1251 + #1257 + #1258 corrected methodology.

### Exit codes

- `0` = PASS (candidate's MLX scorer faithfully predicts contest score)
- `1` = FAIL (drift exceeds gate; do NOT dispatch)
- `2` = CLI / measurement error

## Smoke-test verdict (canonical PR95 archive baseline)

`experiments/results/mlx_candidate_contest_equivalence_gate_smoke_20260526T064049Z/gate_verdict.json`:

| field | value |
|---|---|
| `schema_version` | `mlx_candidate_contest_equivalence_gate_v1` |
| `verdict` | **`PASS`** |
| `actual_contest_score_difference` | `1.08e-05` (matches sister #1264 anchor 0.000011 to 3 sig figs) |
| `gate_threshold_contest_units` | `0.001` |
| `margin_below_threshold` | `0.000989` (98.9% headroom) |
| `ratio_actual_vs_empirical_anchor` | `0.98×` (within measurement noise) |
| `ratio_actual_vs_pr110_frontier_delta` | `0.014×` |
| `exit_code` | `0` |
| `axis_tag` | `[macOS-MLX research-signal]` |
| `score_claim` | `false` |
| `promotion_eligible` | `false` |
| `ready_for_exact_eval_dispatch` | `false` |

## Integration pattern for Path 3 dispatch wrappers

```bash
# In scripts/operator_authorize_substrate_<path3_candidate>_modal_*_dispatch.sh:
.venv/bin/python tools/gate_mlx_candidate_contest_equivalence.py \
    --archive-zip "$CANDIDATE_ARCHIVE" \
    --candidate-label "$CANDIDATE_NAME" \
    --gate-threshold-contest-units 0.001 \
    --output-json "$REPORT_DIR/equivalence_gate.json" \
    || { echo "[gate] FAIL — refusing paid CUDA dispatch per Catalog #1264 corrected methodology"; exit 1; }
# Gate PASSED — proceed to canonical operator_authorize.py dispatch
.venv/bin/python tools/operator_authorize.py --recipe "$RECIPE_NAME" ...
```

## What this gate empirically enforces

Per the corrected #1258 reading + Catalog #341 dual-tier consumer architecture:
- ✅ Catalog #1 (MPS-fallback default forbidden) — gate refuses any MLX scorer that drifts >0.001 from PyTorch
- ✅ Catalog #127 (custody validator) — non-promotable markers stamped at construction
- ✅ Catalog #192 (macOS-CPU advisory not promoted without Linux verification) — same axis-tag discipline
- ✅ Catalog #317 (one-arg local-MPS-vs-Modal dispatch switch) — gate is the local-vs-paid routing decision point
- ✅ Catalog #341 (Tier A vs Tier B consumer architecture) — gate is Tier A observability-only, axis_tag=`[macOS-MLX research-signal]`
- ✅ Catalog #323 (canonical Provenance umbrella) — gate emits Provenance via `tac.provenance.builders.build_provenance_for_predicted`
- ✅ CLAUDE.md "MLX portable-local-substrate authority" — gate is the canonical disambiguator between MLX-faithful vs MLX-noise routing
- ✅ CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" — gate is NOT a contest score claim; paired contest CPU + CUDA on exact submission archive bytes still required for any score claim

## Sister surfaces

- `tools/measure_pr95_mlx_pytorch_actual_contest_score_difference.py` — the canonical measurement (sister #1264)
- `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py` — produces the candidate archives this gate consumes (#1257)
- `tools/export_pr95_mlx_to_pytorch_state_dict.py` — produces the PyTorch state_dict (#1251)
- `.omx/research/pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md` — empirical anchor source (corrected closure footer)

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map = N/A (defensive validator gate)
- hook #2 Pareto constraint = N/A
- hook #3 bit-allocator = N/A
- hook #4 cathedral autopilot dispatch = **ACTIVE** (every Path 3 candidate's gate verdict is queryable for dispatch ranking)
- hook #5 continual-learning posterior = **ACTIVE** (gate verdict rows feed `.omx/state/` canonical posterior consumers)
- hook #6 probe-disambiguator = **ACTIVE** (the 0.001 threshold IS the canonical disambiguator between "MLX-faithful-enough-to-dispatch" vs "MLX-too-noisy-to-trust")

## Discipline applied

Catalog #229 PV (read tac.provenance API surface + sister tool CLI contract) + #117/#157/#174 canonical serializer (commit pending) + #110/#113 APPEND-ONLY (NEW file) + #208 (no `/Users/adpena/...` in body) + #287 (every rationale ≥4 chars, non-placeholder) + #323 canonical Provenance umbrella + #341 dual-tier consumer architecture (Tier A) + CLAUDE.md "MLX portable-local-substrate authority" + "Apples-to-apples evidence discipline" + "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant hardware" non-negotiables.

## Frontmatter (canonical v2 per Catalog #300)

- council_tier: T1
- council_attendees: [Shannon, Dykstra, PR95Author]
- council_quorum_met: true
- council_verdict: PROCEED
- council_predicted_mission_contribution: frontier_breaking_enabler (locks the MLX-as-contest-grade empirical anchor into reusable infrastructure that every Path 3 candidate inherits)
- council_override_invoked: false
- council_dissent: []
- council_decisions_recorded:
  - "op-routable #1: integrate gate into future scripts/operator_authorize_substrate_<path3>_*.sh wrappers"
  - "op-routable #2: extend gate to non-PR95 candidate architectures via generalized state_dict + decoder protocol when needed"
  - "op-routable #3: consider promoting gate to STRICT preflight check that refuses Modal/Vast.ai/Lightning dispatch invocations targeting MLX-trained candidates that lack a recent PASS verdict"
- horizon_class: frontier_pursuit
- related_deliberation_ids:
  - pr95_mlx_full_inflate_parity_closure_landed_20260526
  - pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526
  - pr95_mlx_pytorch_export_parity_bridge_landed_20260525
EOF