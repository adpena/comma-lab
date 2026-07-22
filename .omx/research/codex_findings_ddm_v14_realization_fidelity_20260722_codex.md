# DDM v14 receiver-realization fidelity — Codex findings

Date: 2026-07-22
Lane: `ddm_v14_realization_fidelity`
Tasks: #603 / #613 / master #578
Axis: `[macOS-CPU frozen-scorer advisory]`
Authority: `score_claim=false`; `research_only=true`; `promotion_eligible=false`
Verdict: `ADVISORY_V14_RECEIVER_REALIZATION_REPAIR_PARTIAL_STATIC_CELL_FORECAST_FALSIFIED`
Verdict scope: INSTANCE for the counted v14 profile and three G4 static rules; FORMULATION for painting semantic masks through fixed RGB prototypes; predict-project and grammar families remain OPEN.
MAIN landing review: REQUIRED.

## Outcome first

The v13 binder was real and fixable in part. V13 painted at the 384x512 scorer grid, bypassed the
camera-resolution side of R, reused the inherited flat Movable color `[47,50,42]`, and composed G1
with inherited masks instead of making G1 the authoritative replacement. V14 now uses an exact
23-byte counted receiver profile, paints an ordered semantic map at 874x1164, preserves uint8
prototype amplitude, makes G1 replacement exact, and leaves bilinear downsampling to the frozen
evaluator.

That repair moves the full n600 island arm from the historical non-comparable v13
132,606 B / d_seg 0.029592759874 / Movable 0.481331895297 to the receiver-closed v14
133,247 B / **d_seg 0.027470296224** / Movable **0.291615222639**. This is material, but it does
not recover the mask-level 0.000282948812 promise. The G4 horizon rule is the only positive exact
static-field row: 133,755 B / **d_seg 0.027416720920**, joint-objective delta
**-0.005003006483**. The 0.00116 fork therefore does not fire.

One-line verdict: **receiver ordering, resolution, prototype, and amplitude were repaired; the
remaining mask-to-argmax gap is an RGB/scorer projection problem, and the G4 cell forecast mostly
does not transfer through that receiver.**

Pointer honesty: `0.1910828242 [contest-CPU]` — UNMOVED.

## Per-stage diagnosis

The semantic mask and frozen SegNet argmax are the only stages with a class-conditional metric.
Painted RGB, uint8 survival, and the post-R tensor are intervening representations, not additional
argmax authorities. Assigning a conditional d_seg to them would fake an evaluator. The receipt
therefore records exact representation invariants at those stages and measures eight representative
islands through the three honest end-to-end receiver variants.

| Stage | Measured/checked observation | Disposition |
|---|---:|---|
| Exact G1 mask at 384x512 | mask-level d_seg 0.000282948812; eight representative mask errors 35–114 / 196,608 | Grammar promise is real. |
| Ordered RGB paint at 874x1164, pre-R | G1 replaces inherited Movable; order is UndrivableBoundary → Road → Lane → Movable → MyCar; profile RGB is `[107,0,114]` for Movable | Ordering and scorer-grid bypass fixed. No standalone argmax metric exists here. |
| Uint8 realization | prototype bytes survive exactly at amplitude 255; zero coverage expansion selected | Minimal-write death removed. Erode/dilate/AA probes did not beat the exact hard coverage. |
| Bilinear R-down to 384x512 | evaluator-owned R consumes camera-resolution RGB | Correct receiver geometry restored. No standalone class metric exists before SegNet. |
| Frozen SegNet argmax | v13 legacy bypass: total 0.029592759874 / Movable 0.481331895297; v14 full-R: total 0.027470296224 / Movable 0.291615222639 | Flat prototype/context projection remains the binding loss. |
| G4 horizon rule through the same path | total 0.027416720920 / Movable 0.291213257323 | Positive but only 5.2867% of its cell-space d_seg forecast realizes. |

The eight representative n600 islands are source pairs 447, 53, 416, 296, 547, 278, 501, and
346. Their receipt rows bind `exact_mask_error_fraction`, legacy scorer-grid bypass d_seg, legacy
flat paint through full R d_seg, and fixed camera-paint through full R d_seg. The fixed prototype
helps substantially on pairs 447, 547, 278, 501, and 346, is approximately neutral on 53, 416,
and 296, and never changes the underlying G1 mask. This mixed sign is why the negative is scoped to
the fixed-prototype formulation rather than the grammar family.

## Fixed-path ladder

| Pair set / candidate | Exact archive B | d_seg | d_pose | Lane conditional | Movable conditional | Status |
|---|---:|---:|---:|---:|---:|---|
| n64 islands | 58,049 | 0.041460116704 | 159.134995392648 | 0.496831048153 | 0.667507693326 | diagnosis only |
| n64 islands + inherited raw lane | 58,840 | 0.044455448786 | 159.148942166013 | 0.623551246118 | 0.678687832964 | harmful |
| n600 v13 islands control | 132,606 | 0.029592759874 | 163.016398660918 | 0.434971091989 | 0.481331895297 | historical, receiver path not comparable |
| n600 v14 islands | 133,247 | **0.027470296224** | 163.061327281443 | 0.435195521828 | **0.291615222639** | selected base |
| n600 v14 islands + inherited raw lane | 135,645 | 0.032142613729 | 163.044913501613 | 0.616717271976 | 0.341274716245 | rejected |

The four n600 lane windows have total d_seg deltas -0.000722249349, -0.001139322917,
-0.001861572266, and +0.008249918620. Three local wins are overwhelmed by the fourth window.
This is an INSTANCE negative for the inherited raw-q8 lane phase control. It is not evidence for
an AR(1)-whitened BEV successor.

## G4 forecast through the repaired receiver

G4 landed during this run at commit `4da5468e7c`. Its receipt says the top-decile sensitive pixels
carry 89.9% of flips and the field is approximately 98.8% image-static. V14 reconstructed all three
SHA-bound G4 payloads from the preserved recurrence arrays and measured each exact archive over
all n600 pairs.

| Exact rule | Payload B | Archive B | d_seg | delta d_seg | realization fraction | joint delta | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Movable midband | 12 | 133,759 | 0.028509572347 | +0.001039276123 | -0.768138643107 | +0.085178042141 | harmful |
| Horizon row | 12 | **133,755** | **0.027416720920** | **-0.000053575304** | **0.052866679113** | **-0.005003006483** | selected positive |
| Sparse-all | 4,107 | 137,853 | 0.029902106391 | +0.002431810167 | -0.311501203673 | +0.243312458532 | harmful |

The sparse-all rule improves Movable to 0.235867358294 and Undrivable to 0.001540728383, but its
Lane 0.612563148041 and MyCar 0.012543647361 collateral make the total much worse. The eight most
static loci still miss on roughly 536–556 of 600 pairs after realization. This falsifies the
assumption that a cell-space static correction can be painted with a fixed target-class prototype
and retain its forecasted score effect.

The free decoder context result remains valuable but cannot be subtracted from these archives:
490,794 B → 401,633 B, saving 89,161 B / 18.166685%, applies to a future innovation stream that
is absent here. Base + contextual stream is 534,880 B before container overhead, not an in-box
candidate. The receipt records `applied_to_candidate_archive=false`.

## AR(1)-whitened BEV successor

Disposition: `BLOCKED_NO_DECODER_FREE_PHYSICAL_BEV_CUSTODY`.

G4 has image-space stationarity and recurrence, but no independently observed physical homography,
liveCalibration, or decoder-free metric pose. The existing Pose6 chart is a scorer code, not legal
physical BEV custody. The raw q8 phase control was remeasured under the repaired receiver, but it is
not an honest substitute for anisotropic-volatility AR(1)-whitened innovations in a physical BEV
chart. No BEV result is claimed.

## Fork and blocker delta

Pre-registered condition: selected receiver-closed n600 d_seg <= 0.00116 at <= 200,000 exact
archive bytes. Best measured row is the horizon rule at 0.027416720920 / 133,755 B. Fork: **FAIL**.

Base disposition: `FORMULATION_SCOPED_IRREDUCIBLE_PROJECTION_LOSS_DIRECT_RGB_SCORER_SOLVE_OPEN`.
G4 disposition: `STATIC_CELL_FORECAST_DOES_NOT_TRANSFER_TO_RGB_RECEIVER_DIRECT_SCORER_SOLVE_OPEN`.

Blocker delta versus #603: the scorer-grid bypass, paint ordering, G1 union bug, fixed-amplitude
survival, and camera-resolution placement are closed. The remaining gap is not “unknown receiver
plumbing”; it is measured prototype/context interaction through R and frozen SegNet. The named
successor is a direct RGB solve against the frozen scorer using #559/#549 machinery, while retaining
counted parse-back and the same deterministic receiver. Predict-project/worldsheet grammar remains
open.

## Custody and reproducibility

- n64 receipt: `.omx/research/ddm_v14_realization_fidelity_n64_20260722T215500Z/ddm_v14_realization_fidelity_n64_receipt.json`, SHA-256 `bc9d01c6a3691e1103a580d0e1a34088ff134258b2788b2380930fe85fbae703`.
- n600 receipt: `.omx/research/ddm_v14_realization_fidelity_n600_20260722T215500Z/ddm_v14_realization_fidelity_n600_receipt.json`, SHA-256 `82d3249908d42a86575c407ab3d7acdf9b3706b31225f2e46862b2472966e5a9`.
- G4 receiver receipt: `.omx/research/ddm_v14_g4_receiver_projection_n600_20260722T221500Z/ddm_v14_g4_receiver_projection_receipt.json`, SHA-256 `0b35be44d944bd5a929097bb3967ba7b7c7ce068e2f067d1546ab149cc9e44da`.
- G4 source receipt: SHA-256 `bea555b95aeaa11f4209df5333010c41c5495dd789def2a4f7a2a91973f3408c`.
- G4 recurrence arrays: 396,209 B, SHA-256 `dbc85e7a4f593ab9b7a7f4ed017dbb63a064cb681df806d0bb93277ae8f42451`.
- Every full-n600 row has 38 immutable batch-16 checkpoints and deterministic first-batch replay.
- All candidate files are named `not_a_candidate`, fail closed under sampled member mutation, contain no scorer weights and no per-frame GT argmax table, and remain research-only.
- One initial G4 launcher attempt failed closed on a missing repo-root import; the failure log is preserved and the launcher now binds repo root before importing `tools`.
- Lane maturity is L2 with only `impl_complete` and `real_archive_empirical` marked. The global lane-registry validator still reports 111 historical missing-evidence paths outside this lane; no clean global-registry claim is made.

Bounded re-derivation argv (run repeatedly; each invocation advances at most one candidate stage):

```text
python3 tools/measure_ddm_v14_realization_fidelity.py --config .omx/research/configs/ddm_v14_realization_fidelity_n600_20260722.json --output-directory .omx/research/ddm_v14_realization_fidelity_n600_20260722T215500Z
python3 tools/measure_ddm_v14_g4_receiver_projection.py --config .omx/research/configs/ddm_v14_g4_receiver_projection_n600_20260722.json --output-directory .omx/research/ddm_v14_g4_receiver_projection_n600_20260722T221500Z
```

No Modal or paid dispatch was used.

## STORES CONSULTED

- `CLAUDE.md`
- `AGENTS.md`
- `PROGRAM.md`
- `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `reports/latest.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/master_gradient_anchors.jsonl`
- `.omx/state/modal_call_id_ledger.jsonl`
- `.omx/state/cost_band_posterior.jsonl`
- `.omx/state/continual_learning_posterior.jsonl`
- latest Codex findings/session summaries, latest council/design memos, and recent directives
- v13 n64/n600 predictor and lane-phase receipts
- G4 receipt, summary, and recurrence arrays listed above
- frozen target cache `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`

## MAIN landing review required

MAIN must independently review: (1) G1 replacement and paint-order semantics; (2) the 23-byte
profile and G4 static-rule parse/re-encode contracts; (3) the target-derived aggregate-rule
research-only boundary; (4) exact receipt hashes and batch checkpoint closure; (5) the canonical
law anchor and registry append; and (6) the FORMULATION/INSTANCE verdict scopes before merge.
