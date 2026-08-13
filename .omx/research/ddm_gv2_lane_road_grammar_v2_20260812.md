# ddm_gv2 — Road↔Lane event grammar v2 receipt (2026-08-12)

## Verdict

**The pre-registered falsifier fired.** On the sealed stratified-random n32 pair set, the complete
254-event target-reachable construction census produced **0 net-flip-positive events** under the frozen affected-pair
CPU-torch scorer. One event was neutral and 253 were harmful. The top-200 schema-compatible store
therefore has no event eligible for the honest positive-gain, support-disjoint, nonnegative-pose-bound
stack: **0 flips = 0.0 S optimistic eligible gain**, versus the required **0.000216 S** and the
0.001 stretch target.

This is `[macOS-CPU advisory, frozen scorer, affected-pair n600 denominator]`, not an exact score and
not contest authority. The verdict scope is nevertheless the charter's declared
**FORMULATION closure of the sparse Road↔Lane token-event family on CP135**: unchanged EC1-wire,
connected 12/24/48/96-site boundary segments, both signs, current local Road↔Lane-error anchors, and
JS4-Jacobian-minimized support. Do not build GV3 on this base. Route the Seg leg to learned implicit
edge conditioning in the JS1 stage-0 lineage.

No exact pose-null event was constructed. The unchanged VD1 wire carries discrete token substitutions;
it cannot carry JS4's continuous projected correction or BO1's output-space Q3 correction. GV2 used the
JS4 Jacobian honestly to choose the less pose-sensitive of two connected supports before any scorer
result existed, then retained the nonlinear CPU measurement as the pose-stack prediction. Calling this
an exact projection or Q3 placement would be false.

## What was built

The alphabet derives from three receipts rather than a new class prior:

- PC2/m91 found Road as the 87.8% flip hub and Road↔Lane as 49.23% of its measured edge mass on the
  TB1-ep399 advisory vehicle. Those percentages are historical formulation evidence, not transferred
  CP135 measurements.
- LC1 showed why a dense Lane target is wrong: its PE3 ideal-label substitution introduced 5,557
  Lane predictions where both GT and the base were Road. GV2 therefore anchors each sign to a current
  base Road↔Lane error and restricts edits to the existing directed token boundary.
- VD1's exact T4 gen-1 census found only 5 jointly flip-positive/pose-safe events and +6 flips total.
  GV2 increases the event scale from singleton/small chunks to connected 12, 24, 48, and 96-site
  segments while leaving VD1's event wire unchanged.

For each of the 32 JS4 pairs, GV2 constructs up to eight units: four scales for Road→Lane and four for
Lane→Road. Each unit starts from the closest real local CPU Road↔Lane error to a directed token-boundary
component large enough for that scale, requires the component to be within three pixels of the error,
and grows one connected 8-neighbour segment. Two 96-site directions failed that target-distance rule and
were not emitted, leaving 254 units. Opposite directional tie-breaks yielded two distinct supports for
151/254 units; the remaining 103/254 collapsed to the same connected support and were retained once. For every distinct support, GV2 retained the EC1 payload,
real Brotli-q11 and XZ payloads, token plane, pre-R field, correction, uint8 camera, and scorer lattice.
The JS4 6×589,824 Jacobian predicted each support's first-order pose shift; the lower nonnegative
first-order d_pose support won before SegNet or nonlinear PoseNet ran.

The winner then ran through the same frozen CPU custody transport used by JS6: exact retained CP135
frame-1 base, candidate receiver camera and R, frozen SegNet, and frozen PoseNet on the affected pair.
Every emitted event is receiver-effective: camera changes range from 19,472 to 267,562 channel values,
and scorer-lattice changes range from 9,580 to 117,993 values.

## Advisory result

| Census | Result |
|---|---:|
| Constructed/scored candidates | 254 / 254 |
| Net-flip positive / neutral / harmful | **0 / 1 / 253** |
| Net-flip gain range | **−106 to 0** |
| Sum of singleton net-flip gains | **−10,345** |
| Emitted ranked store | 200 events |
| Emitted positive / neutral / harmful | **0 / 1 / 199** |
| Emitted per-event pose-bound pass at 1.3e-7 | 16 / 200 |
| Positive ∩ pose-bound-pass | **0 / 200** |
| Intended selection | **0 events** |
| Honest pose-stack prediction | **0.0 ≤ 1.3e-7** |
| Optimistic eligible gain | **0 flips = 0.0 S** |
| Pre-registered 0.000216 S bar | **FAILED** |
| 0.001 S stretch bar | **FAILED** |

The scale response is monotone in the wrong direction:

| Segment sites | Candidates | Mean net-flip gain | Best | Worst |
|---:|---:|---:|---:|---:|
| 12 | 64 | −9.50 | 0 | −22 |
| 24 | 64 | −22.375 | −11 | −41 |
| 48 | 64 | −45.328125 | −27 | −72 |
| 96 | 62 | −87.16129032258064 | −56 | −106 |

The single neutral and first-ranked event is `gv2_0000_3ec0586eea57`, pair 193, Lane→Road, 12 sites.
Its predicted nonnegative global d_pose bound is zero and its standalone Brotli payload is 34 B, but
zero Seg gain cannot qualify as an improving event.

The mechanism matches LC1's warning. The segment itself is boundary-local, but the receiver's spatial
blast radius is broad relative to the sparse correct-sign anchors. Within a radius-3 scorer influence
band, every emitted event reaches at least one target Road↔Lane error, the median reach is 1, and the
median sign-precision prior is only 0.01786. Increasing segment length therefore increases already-correct collateral faster than
it reaches target errors.

## Pose and byte accounting

For event `i`, the retained fields use:

`seg_gain_i = 100 * net_flip_gain_i / (600 * 384 * 512)`

`pose_bound_i = max(0, JS4_first_order_delta_i, nonlinear_CPU_delta_i)`

`standalone_rate_i = 25 * Brotli_q11_bytes_i / 37,545,489`.

The pose quantity is a conservative advisory prediction over the two measured estimates, not a formal
upper bound and not CUDA authority. Across the emitted 200, the predicted pose-bound min/median/max is
0 / 5.13967356267643e-6 / 4.1222204601126236e-5. Only 16 events are individually below the full
1.3e-7 stack budget, and none improves Seg. The nonlinear delta itself ranges from
−1.1118526530873964e-6 to +4.1222204601126236e-5, confirming JS4's prior result that first-order
minimization does not control finite nonlinear leakage.

The 200 exact event objects total 11,946 raw bytes and 9,861 B after individual Brotli-q11 coding;
per-event raw sizes range 29–153 B and Brotli sizes 33–86 B. JO1's earlier +3 B/200 carrier result was
not transferred: GV2's changed alphabet has not been recomposed through JO1, and no rate claim uses
that number. Because every candidate is Seg-harmful or neutral, carrier repricing cannot make the
selection eligible.

The complete per-event table is the store index, including pair, sign, sites, predicted flips, pose
bound, bytes, construction receipt, scorer outputs, and payload hashes:

`/Volumes/VertigoDataTier/pact/ddm_gv2_20260812/event_store_target_anchored_v2/proposal_index.jsonl`

## Store and unchanged-validator proof

The gen-2 store is:

`/Volumes/VertigoDataTier/pact/ddm_gv2_20260812/event_store_target_anchored_v2/`

It contains exactly the 200 indexed proposal directories: 200/200 indexed IDs present, 0 missing, and
0 unindexed directories. The earlier `/event_store/` directory and `/retained/pair_*` census are
superseded retained history from the stricter-anchor review; consumers must use the target-anchored-v2
paths above.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `proposal_index.jsonl` | 1,662,832 | `89589f1cdc18c3e04cc06842a5b6ffc362499115c0dc6f6700cd75a04365225b` |
| `state.json` | 1,583 | `48b161cdfd6d9fc2e5019a30ee5025f43f467621b5b5ad7c6fcabd1a4ca72b10` |
| `FINAL_RESULT.json` | 2,723 | `6dfeb70bd1a475ea6f7c1b102bfb5dfd981a8258ad23bc60d4749710b5e4f597` |
| unchanged-VD1 K=200 bundle | 51,513 | `f3ac1dea47631bc125061184b20e9aa7a83e9430c03c9df430f901d10b8086a4` |
| unchanged-VD1 compatibility receipt | 91,549 | `f03dc22dd0e2f334b5efdb0cb4fcabe7dfc6a5937e88cb835472c42376df7cff` |

`experiments/ddm_vd1_modal_batch_event_validator.py::build_event_bundle` consumed the store unchanged
with `k=200`, selected all 200 unique IDs in `full_200_census` mode, and produced the retained bundle
above. This was a local schema/parse-back proof only. No Modal function or scorer ran.

The command that would re-fire VD1 with a fresh run identity is pinned for provenance but **FOLDED and
must not fire**, because the pre-screen found no positive event and the falsifier explicitly routes away:

```bash
.venv/bin/modal run experiments/ddm_vd1_modal_batch_event_validator.py \
  --archive /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip \
  --runtime /Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime \
  --event-store /Volumes/VertigoDataTier/pact/ddm_gv2_20260812/event_store_target_anchored_v2 \
  --jo1-analysis /Volumes/VertigoDataTier/pact/ddm_jo1_20260812/10_ANALYSIS.json \
  --output-dir .omx/state/ddm_vd1_modal_batch_event_validator/ddm_gv2_vd1_20260812a \
  --k 200 \
  --run-id ddm_gv2_vd1_20260812a \
  --resume-from ddm_gv2_vd1_20260812a \
  --lane-id ddm_vd1_modal_batch_event_validator \
  --instance-job-id modal:ddm_gv2_vd1_20260812a \
  --claim-agent main:ddm_gv2
```

## Recall evidence and boundaries

STORES CONSULTED: `PROGRAM.md`; `CLAUDE.md`/`AGENTS.md`; `.omx/tmp/codex_runs/_common_contract.md`;
`docs/operating_manual_craft_handoff.md`; `.omx/research/main_hot_state.md`; the canonical equation
listing; the sub-0.15 DAG and research indexes; LC1, PC2/m91, BO1/Q3, EC1, JS1, JS4, JO1, JS5/JS6/JS7,
VD1 and VD1-census receipts; the JS4 projector manifest and all pair receipts; the CP135 token/raw/scorer
custody stores. No stronger current-vehicle receipt showed that a discrete EC1 token substitution could
realize the continuous JS4 or Q3 projection.

No archive was built, no full n600 CPU/CUDA scorer ran, no Modal job was dispatched, and no exact score
was measured. The current pointer did not move. `n32` is a seeded stratified pair census for proposal
construction; each reported flip delta is an affected-pair frozen-CPU measurement placed over the fixed
n600 pixel denominator, not a population estimate from sample weights.

Unified-Lagrangian wire-in: sensitivity-map contribution = the retained sign-aware Road↔Lane anchors,
receiver blast radii, and per-event pose fields; Pareto constraint = positive net flips with cumulative
nonnegative pose bound ≤1.3e-7 and counted bytes; bit-allocator hook = the per-event raw/Brotli table,
with allocation refused because no event has positive Seg value; cathedral-autopilot dispatch hook =
VD1 FOLDED and the implicit-conditioning route queued behind its explicit trigger; continual-learning
posterior update = FEED-gv2 in the canonical sub-0.15 DAG; probe-disambiguator = the store separates
discrete JS4-Jacobian minimization from an exact continuous JS4/Q3 projection, closing the former without
making a false claim about the latter.

## Dispositions

- **FOLDED:** GV2 unchanged-wire sparse Road↔Lane token-event formulation. Owner: `ddm_gv2`.
  Consumer store: `/Volumes/VertigoDataTier/pact/ddm_gv2_20260812/`. Fire trigger: none on CP135; do not
  create GV3.
- **FOLDED:** unchanged VD1 contest-CUDA re-fire over the GV2 store. Owner: MAIN scorer-lane router.
  Consumer store: `/Volumes/VertigoDataTier/pact/ddm_gv2_20260812/event_store_target_anchored_v2/`. Fire trigger: none;
  0/254 candidates improved advisory flips, so a paid validation cannot satisfy the charter's bar.
- **QUEUED-WITH-A-FIRE-ORDER:** learned implicit edge conditioning in the JS1 stage-0 lineage. Owner:
  MAIN training-leg router. Consumer store:
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/`. Fire trigger: a
  current-vehicle from-scratch retained stage checkpoint exists, the sole scorer lane is free, and this
  receipt is consumed as a prohibition on another discrete Road↔Lane token-segment grammar.

Effective frontier remains **CP135 S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]**.
Own-vehicle frontier remains **LC2 S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**.
