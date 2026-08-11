# DDM CN4 arc consolidation — 2026-08-11

**Verdict:** consolidation complete for all seven chartered lanes. This unit ran no
scorer, evaluator, Modal job, or payload materializer and moved no frontier. It
registered three executable equation surfaces, backfilled seven lane rows, and
routed every finding in the seven source memos. The effective pointer remains the
already-landed CP135 row, `S=0.16195513827824176 @ 186252 B`
`[contest-CUDA T4,n600]`; the own-vehicle frontier remains LC2,
`S=0.16959899569230852 @ 187226 B [contest-CUDA T4,n600]`.

## Canonical equations

| Equation | Registration | Empirical anchor | Boundary |
|---|---|---|---|
| `cpu_cuda_score_gap_v1` | Existing law gained a real evaluator, a domain refinement, and the LC2 anchor | Identical `f154f0ab…e1a45` bytes: CPU `0.20728492781521812`, CUDA `0.16959899569230852`, so CUDA−CPU is `-0.03768593212290960`; CPU pose is `6.6x` and seg `1.45x` worse | Paired measurement, not a transferable sign or lineage predictor |
| `realization_breakeven_bytes_v1` | Existing law gained a real evaluator and the PZ4 realization anchor | PZ4P envelope promised `19221 B`; PZ4R realized `4089 B`, or `21.27360699235%`; `15132 B` returned as decoder-required state | Envelope is not renderable and is not an archive candidate; score recovery is not measured |
| `radius2_multistart_singleton_escape_v1` | New advisory apparatus law with executable evaluator, declared producers/consumers, and one n600 anchor | Pass 1 accepted `597/600`, `S=0.18474482031130968`; pass 2 accepted `517`, `S=0.17952896607020802`, `d_pose=1.4717098986238853e-05`, `187221 B` | `[macOS-CPU advisory]`; exact contest-CUDA transfer remains untested |

Implementation lives in
`src/tac/canonical_equations/ddm_cn4_arc_consolidation_20260811.py` and
`src/tac/canonical_equations/evaluators.py`. The append-only canonical registry
contains the domain refinement, two anchor appends, and the new equation
registration.

The append-only log preserves one superseded PZ4 append whose inputs accidentally
encoded unmeasured score recovery as `0.0`. A following domain-correction event
marks that field invalid, and the corrected anchor omits it and states
`score_recovery_status=UNMEASURED`; consumers must use the corrected anchor.

## Lane maturity backfill

| Lane | Level and honest gates | Boundary / next gate |
|---|---|---|
| `lane_ddm_rc64p_native_cpu_decode_20260810` | L2: implementation, real archive, contest CPU, contest CUDA | The exact scores belong to the identical LC2 control. Route-B RC64 is receiver-closed but unscored. |
| `lane_ddm_cp135_rate_compose_20260810` | L2: implementation, real archive, contest CUDA | Exact composed row; no contest-CPU transfer is claimed. |
| `lane_ddm_lp135_lossless_pack_20260810` | L1: real archive; research-only | Reconciliation of already-shipped bytes, not a new implementation or row. |
| `lane_ddm_sr1_implicit_edge_conditioning_20260811` | L1: implementation; research-only | Scorer-free coder study; no scored archive. |
| `lane_ddm_pz4p_pose_gauge_preproof_20260811` | L1: implementation; research-only | Non-renderable envelope only; superseded by PZ4R realization. |
| `lane_ddm_pz4r_pgq1_receiver_20260811` | L2: implementation, real archive, three-clean review; research-only | Scorer and Linux runtime closure are pending. |
| `lane_ddm_ps135_pose_resolve_20260810` | L2: implementation, real archive; research-only | Live retained solve; only advisory scoring is present. |

All seven rows were created and marked through `tools/lane_maturity.py`; registry
validation reports 2270 valid lanes. The audit log also contains the CLI events,
but it had unrelated pre-existing unstaged rows, so this unit does not claim sole
ownership of that file's full diff.

## Seven-memo primary disposition

| Source memo | Primary disposition | Result |
|---|---|---|
| `ddm_rc64p_native_cpu_decode_20260810.md` | routed-to-equations | Paired LC2 CPU/CUDA device anchor registered; Route B is represented by its lane row. |
| `ddm_cp135_rate_compose_20260810.md` | routed-to-lane-row | Exact CUDA composition and its owned/borrowed accounting are recorded in the CP135 row. |
| `ddm_lp135_lossless_pack_20260810.md` | certified-no-route-needed | F26 is already completely shipped, F26 ANS loses, and CAP1's `79 B` is already banked; no new consumer is justified. |
| `ddm_sr1_implicit_edge_conditioning_20260811.md` | routed-to-task | The only live extension is joint distortion-side edge conditioning, already consumed by JS1 stage 1 / task `#995`; the scorer-free formulation is closed in its lane. |
| `ddm_pz4p_pose_gauge_preproof_20260811.md` | routed-to-equations | The envelope-to-realized yield anchor preserves the upper-bound miss; PZ4R owns the renderable successor. |
| `ddm_pz4r_pgq1_receiver_20260811.md` | routed-to-task | The receiver row is registered; scoring remains queued behind PS135 lane custody. |
| `ddm_ps135_pose_resolve_20260810.md` | routed-to-equations | Radius-2 multistart escape is registered; the live solve continues under the existing PS135 owner. |

## Finding-level routing ledger

The full arc yielded 21 genuine findings, inside the pre-registered `15–25`
range. That does not falsify the 49:0 orphan monitor. It does show why a raw
finding count is too aggressive for receipt-heavy arcs: several findings are
paired controls, closure receipts, or constituents of one already-owned task.

| # | Finding | Disposition | Consumer |
|---:|---|---|---|
| 1 | LC2 same bytes reverse the older CPU/CUDA sign | routed-to-equations | `cpu_cuda_score_gap_v1` |
| 2 | Native entropy is only `1.1–3.1 s`; HPAC probability work is the CPU wall | routed-to-lane-row | RC64P lane |
| 3 | RC64 Route B is `187222 B`, four bytes below LC2, receiver-closed but unscored | certified-no-route-needed | Dominated for frontier fire by exact CP135; retain only as a decoder diagnostic |
| 4 | CP135 exact CUDA row is `0.16195513827824176 @ 186252 B` | routed-to-lane-row | CP135 lane and existing pointer |
| 5 | VP1/CAP1/HP3 compose to the retained net rate win | routed-to-lane-row | CP135 lane |
| 6 | CP135 coder/SMEVR/LOTTO alternatives did not beat the composed winner | certified-no-route-needed | Closed within the measured candidate set |
| 7 | LP135 found 100% of FD135's `4328 B` ancestry already banked | routed-to-lane-row | LP135 lane |
| 8 | F26 ANS loses RC64 by `6–9 B` | certified-no-route-needed | No retry without a new coder mechanism |
| 9 | CAP1's `79 B` saving is already fully implemented | certified-no-route-needed | Existing CP135 composition |
| 10 | SR1 causal additive edge context saves only `2 B` | routed-to-lane-row | SR1 formulation closure |
| 11 | SR1 pose-scalar edge context costs `43 B` | certified-no-route-needed | Same formulation closed |
| 12 | Joint distortion-side edge conditioning remains distinct and open | routed-to-task | JS1 stage 1 / task `#995` |
| 13 | PZ4P produces a `19221 B` non-renderable upper-bound envelope | routed-to-equations | `realization_breakeven_bytes_v1` anchor |
| 14 | Exact PGQ ranks `<=5` do not close the receiver | certified-no-route-needed | Formulation-scoped rank ladder closed |
| 15 | PZ4P's fire condition produced PZ4R | routed-to-lane-row | PZ4R successor row |
| 16 | PZ4R realizes only `4089/19221 B = 21.27360699235%` | routed-to-equations | Realization-yield anchor |
| 17 | PZ4R direct-v6 archive is `183137 B`, receiver-closed | routed-to-lane-row | PZ4R lane |
| 18 | PZ4R distortion is unknown | routed-to-task | Existing post-PS135 scorer queue |
| 19 | PS135 `+/-1` singleton search is genuinely closed by the zero eighth pass | certified-no-route-needed | Do not repeat singleton search |
| 20 | Radius-2 native/wrong-sign/projected starts escape on pass 1 and continue on pass 2 | routed-to-equations | `radius2_multistart_singleton_escape_v1` |
| 21 | PS135 terminal landing is the trigger for JS1 reseal/Stage C | routed-to-task | Existing PS135→JS1 chain |

## RECALL EVIDENCE

Before routing, the full stores were queried for three concepts: paired-device
score deltas, envelope-to-realized yield, and singleton-to-radius-2 pose search,
plus ownership joins for `#995`, `#998`, and the PS135→JS1 chain. Stores consulted:
research corpus (8359 records), canonical equations (880), memory (2105), DAG
(915), council (297), tasks (531), and docs (96). Recall found the existing
`cpu_cuda_score_gap_v1` and `realization_breakeven_bytes_v1` surfaces, no existing
radius-2 multistart law, existing JS1 ownership for edge conditioning, the
already-banked LP135 constituents, and PZ4R's supersession of the PZ4P envelope.
Those findings prevented twin equations and duplicate tasks.

## Six-pillar and wire-in declaration

- **Equation:** three executable surfaces with explicit units and signs.
- **Domain:** archive lineage, population, neighborhood, axis, and non-transfer
  exclusions are explicit.
- **Empirical anchors:** exact receipt-backed values are attached; PZ4P remains
  an envelope and PS135 remains advisory.
- **Residuals:** LC2 records the miss against the older `+0.033` lineage prior;
  PZ4 records unrealized yield; PS135 records escape prevalence.
- **Recalibration:** each equation names a new-anchor or exact-CUDA trigger.
- **Wire-in:** producers and consumers route through the canonical equation
  registry, seven lane rows, the existing task/DAG owners, and this disposition
  ledger. Mission contribution is `apparatus_maintenance/frontier_protecting`:
  it prevents false device transfer, false envelope pricing, and duplicate
  follow-ons; it is not a score move.

## Verification and authority boundary

MEASURED here means registry state, source-memo receipts, real coder/archive
sizes, receiver closure, and the explicitly labelled exact or advisory axes.
NOT MEASURED here: no SegNet, PoseNet, `upstream/evaluate.py`, Modal dispatch,
contest replay, or new payload generation ran. CN4 did not write or mutate
`/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/`; it read retained receipts
only. Public-PR clones were not modified. Exact CP135 and LC2 numbers are prior
receipts, not newly generated rows.

## NEXT_IF_RESUMED

- **Disposition: routed-to-task. Owner: PZ4R scorer successor after PS135 releases the scorer lane. Consumer store: the canonical task/DAG store plus the PZ4R retained receipt directory. Fire trigger: PS135 terminal landing, lane claim, scorer lock, and storage preflight all pass; then score the exact direct-v6 archive bytes and retain every output.**
- **Disposition: routed-to-task. Owner: JS1 stage-1 joint solver. Consumer store: canonical task `#995` and the JS1 DAG/store. Fire trigger: PS135 final solve lands and the JS1 base is resealed; test distortion-side edge conditioning jointly, not SR1's closed additive coder context.**
- **Disposition: routed-to-task. Owner: PS135 live lane owner. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/` and the PS135 lane row. Fire trigger: the current resumable leg reaches its governed terminal condition; preserve the final archive and only then request exact contest-CUDA transfer.**

## LIVE-HYPOTHESES

- PS135's radius-2 gain may transfer to contest CUDA because it survives receiver closure across two full n600 advisory passes, but identical archive bytes must be evaluated before the gain is authoritative.
- PZ4R may beat LC2 after scoring because it removes `4089 B` with a receiver-closed direct-v6 archive; plausibility depends on its unknown distortion costing less than the corresponding rate gain.
- Joint distortion-side edge conditioning may pay even though SR1's additive entropy context did not, because it changes the generated scorer-facing signal rather than only the coder probabilities.

## DEAD-ENDS

- Repeating the shipped `+/-1` singleton pose search is closed: the zero eighth pass confirmed the singleton optimum for that neighborhood.
- Treating PZ4P's `19221 B` envelope as a candidate is closed: PZ4R restored `15132 B` of decoder-required state and realized only `4089 B`.
- Retrying F26 ANS or searching for an unbanked kilobyte in LP135 is closed on this arc: ANS lost by `6–9 B`, FD135 ancestry was fully banked, and CAP1 was already shipped.
- Porting only RC64 entropy code as the CPU cure is closed for this instance: entropy consumed `1.1–3.1 s` while HPAC probability work accounted for more than `99.5%` of the wall.
- Adding SR1-style causal edge context to the entropy model is closed for this formulation: the gain was `2 B`, and the pose-scalar variant cost `43 B`.

Own-vehicle frontier: LC2 `S=0.16959899569230852 @ 187226 B [contest-CUDA T4,n600]`; CN4 moved it by `0`.
