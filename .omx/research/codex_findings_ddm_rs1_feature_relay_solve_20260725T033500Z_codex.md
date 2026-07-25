# Codex findings — DDM RS1 feature-space relay solve

UTC: 2026-07-25T03:35:00Z  
Lane: `lane_ddm_rs1_feature_relay_solve_20260725`  
Delegation checkpoint: `codex_delegate:ddm_rs1_feature_relay_solve:20260725T030635Z`  
Evidence axis: `[macOS-CPU frozen-scorer advisory]`  
Score claim: `false`  
Pointer: `0.1910828242 [contest-CPU]` — **UNMOVED**

## Disposition

`BLOCKED_INTERNAL_STATION_DYNAMICS_NOT_CUSTODIED`.

This is a `FORMULATION x current SHA-bound #484/AT1/SN1/MS4D/J8F custody`
blocker. It is not a feature-relay or multiple-shooting family verdict. No G3
top24 solve, bounded n600 replay, relay-radius comparison, score claim,
promotion, or pointer move was performed.

The authority's admission row saying the station measurements are measured is
true only for the named loci and existing aggregate/final-head measurements.
It does not establish the candidate-specific segment dynamics required to
instantiate a relay solve.

## Fresh premise audit

| source | established | does not establish |
|---|---|---|
| #484 PRE-SE | block2 `encoder.model.blocks.1.2.se.forward_pre`, shape `(1,144,96,128)`; block3 `encoder.model.blocks.2.2.se.forward_pre`, shape `(1,288,48,64)`; aggregate retained-mass measurements | G3/J8F candidate targets, margin-Fisher Grams, segment Jacobians, or continuity secants |
| AT1x | measured relay depths `camera_input_x` and `scorer_plane_y`; the receipt explicitly records `unmeasured_internal_layers_claimed=false` | block2 or block3 internal-layer dynamics |
| SN1 | n600 aggregate feature moments, boundary energy, and asymmetry telemetry | intermediate Fisher pullback or full segment Jacobian |
| MS4D | complete post-R rank-4 final-head metric with `DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT` | input-to-head actuator dynamics or either internal station edge |
| J8F | exact receiver parse-back and realized n600 end-verdict harness, ready for a reviewed continuation | internal station measurements for a new proposal |

The two historical #484 raw receipt identities are named in tracked sources,
but neither receipt is available in this worktree nor either SSD waterfall.
The first receipt's byte count is recovered from the tracked reliability
receipt; the second tracked source publishes its SHA-256 but no byte count, so
the config records `bytes=null` instead of inventing one.

## Landed construction

- A typed two/three-station Fisher-metric multiple-shooting solver. It uses
  independent station variables, exact linear continuity constraints, and an
  fp64 minimum-norm KKT solve. No damping, learning rate, or Euclidean station
  objective is introduced.
- A one-shot composed final-station solve for the v17-style direct control.
- A strict end-only admission helper requiring n600, exact parse-back,
  realized-through-R/uint8 custody, frozen scorers, integer archive bytes, and
  negative realized joint delta.
- Equal-budget validity-radius comparison over the same radius ladder. Radius
  is the contiguous accepted prefix, not the largest isolated accepted point.
- A SHA/byte-bound execution gate that audits #484, AT1x, SN1, MS4D, v17, #580,
  J8F, and its own solver/runner sources before any G3 work. Its optional
  `station_bundle` binding consumes the exact measured bundle named below,
  instantiates both predictive solves, and still refuses end acceptance until
  the realized ladder exists.

## Exact blocker

The sealed receipt names eight absent edges:

1. historical #484 raw receipts unavailable in the storage waterfall;
2. G3-top24 candidate-to-block2/block3 station-target join;
3. block2 categorical margin-Fisher Gram plus Euclidean control;
4. block3 categorical margin-Fisher Gram plus Euclidean control;
5. measured `range(A) input -> block2` Jacobian;
6. measured `block2 -> block3` Jacobian;
7. measured `block3 -> rank-4 head` Jacobian;
8. per-candidate station continuity secants.

Receipt:
`.omx/research/ddm_rs1_feature_relay_solve_20260725T030635Z/receipt.json`,
SHA-256
`23d074104ba2081fad50bdb00df5a3415cb076f7dee6f0ab18982350d91e7e71`.

Blocker:
`.omx/research/ddm_rs1_feature_relay_solve_20260725T030635Z/BLOCKER.json`,
SHA-256
`703dfa876ac0e0d5663eaa6ef0561d43b14f0bb18d1f85dbc5ea3c0328c2c116`.

## Exact next measurement

Materialize one SHA-bound G3-top24 station bundle for the same J8F integer
candidate rows. It must carry block2/block3 target deltas, categorical
margin-Fisher Grams, labeled Euclidean controls, the three measured segment
Jacobians, and realized continuity secants. Re-run the strict gate; only after
it admits may the multiple-shooting/direct ladder run and send endpoints to the
existing J8F n600 receiver/scorer harness.

## Triality and wire-in

- Typed DSL/config:
  `.omx/research/configs/ddm_rs1_feature_relay_solve_20260725.json`
- Equations:
  `.omx/research/ddm_rs1_feature_relay_solve_canonical_equations_20260725.md`
- DAG/feed:
  `.omx/research/ddm_rs1_feature_relay_solve_DAG_FEED_20260725.md`
- Sensitivity map: categorical margin-Fisher station Grams are a strict missing
  input; no synthetic substitute was written.
- Pareto/rate: end acceptance uses exact `100*d_seg + sqrt(10*d_pose) +
  25*bytes/37545489`; intermediate predictions are analysis-only.
- Bit allocator/autopilot: fail closed until the station bundle exists; then
  reuse J8F's receiver-closed proposal path.
- Continual learning: the blocker and exact reactivation criterion are durable
  in the receipt and standalone feed.
- Probe disambiguator: direct composed solve and relay solve are both callable
  and must be compared on the same realized radius ladder.

## STORES CONSULTED

Delegated authority file; `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; v7.5/v8 operating specs; current
frontier, lane, subagent, task, and DDM directive surfaces; #484 PRE-SE memos
and equations; AT1x tracked/SSD atlas; SN1 receipt/telemetry; MS4D bundle and
Seg metric; v17 law/receipt; #580 projector; J8F config, receipt, and findings;
both SSD storage waterfalls.

## MAIN review required

MAIN must independently review the custody inference, the exact source
bindings, the KKT/acceptance/radius contracts, all new Python files with their
two clean review passes, and the intentional refusal to run G3/n600 before
merging. This memo is advisory historical provenance and grants no FIRE
authority.
