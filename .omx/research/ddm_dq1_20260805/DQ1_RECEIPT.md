---
schema: ddm_dq1_scorer_slot_debt_drain.v1
date_utc: 2026-08-05
arm: dq1
axis: "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
score_claim: false
promotion_eligible: false
pointer_moved: false
tokens: [no-triality, p0-ledger-ok]
---

# DQ1 - scorer-slot debt drain

## Answer First

No exact row was produced. DQ1 ran one bounded CPU PoseNet rerun for p3v2, drained the ET1 R8 accounting from existing block16 scorer receipt, emitted the honest GR1 subset caveat instead of replaying a prefix-only script, and kept #923 pose-family work queued with fire order. No `upstream/evaluate.py`, no training, no source edits, no contest-CPU/CUDA authority row.

| item | producer claim checked | new DQ1 result | denominator and selection | verdict | consumer impact |
|---|---:|---:|---|---|---|
| p3v2 free-frame pose upper bound | n24 video-order prefix `d_pose=9.123412205850626e-05`, pose term `0.030204986689863777` | n120 stratified `d_pose=0.0004933684543601322`, pose term `0.07024019179644458` | 120/600, NA3 stratified blocks, seed 20260805 | FLIPS at FORMULATION scope | GC17/AU1 and any p3v2 free-wall/candidate-line citation must be downgraded; p3v2 S3 warp n600 remains unchanged |
| GR1 granularity rerace | n48 prefix comparative surface ranked `cell_drop50` over token rows; n600 `cell_drop50` exists | no rerun: rerace script only exposes video-order prefix selection | old n48/600 prefix; old n600 full confirm | SURVIVES for n600 `cell_drop50`; n48 comparative negatives are CAVEATED | Do not use old n48 prefix rows as population-fair formulation kills; add a real stratified selector before reracing |
| ET1 block16 steps=25 | block16 looked seg-live but cap-censored | n32 pose term after `0.1896125904900151` vs bank `0.08453295485558432`, erosion `+0.10507963563443079` | 32/600 receipt subset, block16 realization steps=25 | CAP-CENSORED-STILL / R8 FAIL | Seg-only net `-0.06825022576324405` cannot be banked; next ET1/Q3 action must be joint/R8-priced |
| NA2 #923 pose-family reruns | four pose-family verdicts need strided rerun | 0/4 run by DQ1; all remain queued | NA3 found missing landed harness/render-cache surfaces | QUEUED-with-fire-order | Nothing silently dropped; fire order listed below |

## RECALL EVIDENCE

Governing recall read the DQ1 charter, common arm contract, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, live `main_hot_state`, and the AGENTS/CLAUDE policy sections covering no-fake, scorer authority, serializer, protected paths, and review gates.

Corpus recall was not charter-only. Searches covered `.omx/research`, `.omx/state`, `docs`, `experiments`, `tools`, and `src/tac` with these query families: `p3v2`, `optimal_form_pose`, `gc17`, `au1`, `m96`, `prefix pose bias`, `pose-family`, `#923`, `gr1`, `granularity rerace`, `cell_drop50`, `et1`, `block16`, `R8`, `0.0845`, `pose erosion`, `CAP-CENSORED`, `ss1`, `ca1`, `na2`, `na3`, and `sq2`. The canonical equations registry was queried through `tools/list_canonical_equations.py --json`; the scoped result was general score/rate/pose equations, with no DQ1-specific equation row requiring an update. Research-index/DAG recall searched `CANONICAL_RESEARCH_INDEX`, `sub015_DAG` surfaces, the deferral ledger, and canonical task-status files.

Beyond the charter seeds, recall found that NA3 already derived a stratified n120 pose selector with governing `d_pose_shipped_f16` ratio `1.0057539935665503` versus the population, and found that NA3 had already blocked the four #923 pose-family reruns on missing landed harness/render-cache surfaces. That changed the plan in two ways: p3v2 used NA3's n120 selector directly, and #923 was not re-attempted as fake runnable work. Recall also found that GR1's public rerace script is prefix-only, so the correct DQ1 action was an explicit scope caveat rather than another prefix replay.

The live-board scorer slot was owned by DQ1 for bounded CPU-torch scorer passes only. No full n600 authority job was launched.

## p3v2 Pose Rerun

Scope fields: `axis=[macOS-CPU frozen-PoseNet advisory]`, `verdict_scope=FORMULATION`, `selection_mode=stratified_blocks`, `selection_seed=20260805`, `n=120/600`, `prefix=false`, `source_selection=.omx/research/ddm_na3_20260805/stratified_pose_selection_923.json`, selection SHA-256 `6f7ba6dde77500f09923ecfa7c85a7b977bda1e9f3332e416f6fe05c7bca06f7`.

DQ1 reran the p3v2-class free-frame upper-bound solve on 120 non-prefix NA3-selected pairs, using the existing p3v2 measurement code path and CPU PoseNet. Each pair was appended to a partial JSONL under SSD custody, then the final receipt was written under the same SSD directory.

| row | selection | mean d_pose | sqrt(10*d_pose) | median d_pose | notes |
|---|---|---:|---:|---:|---|
| old p3v2 S1d free upper bound | video-order prefix n24 | `9.123412205850626e-05` | `0.030204986689863777` | not the gating denominator | old source SHA `12838f63ea7164bb8c71b8239e9b26a6f324869ae4f136b1ebad6ca9bfce77d3` |
| DQ1 p3v2 S1d free upper bound | stratified n120 | `0.0004933684543601322` | `0.07024019179644458` | `0.00032041414915977057` | 160 iters per pair |
| DQ1 p3v2 warp base | same stratified n120 | `0.38262594430794045` | `1.9560826779764204` | not material to free-wall verdict | population-matched sanity check |
| old p3v2 S3 warp base | n600 full | `0.3931152819127795` | `1.9827134990027668` | not material to free-wall verdict | old source SHA `ac87e05ee830f0e55821a89a0c00bea7e97705d914aa067bf76a7ca9a618e393` |

Distribution facts for the DQ1 n120 free solve: min `1.2686855991171642e-05`, max `0.0027784248191924494`, `49/120` pairs at or below `2.5e-4`, and `103/120` pairs at or below `1e-3`. Largest five pair errors were pair `141` at `0.0027784248191924494`, pair `129` at `0.0021585479894219183`, pair `105` at `0.0016204885050378967`, pair `114` at `0.0016006382641760967`, and pair `151` at `0.001577539092542798`.

Verdict: p3v2's free-frame upper-bound population-fair rerun FLIPS the producer claim. The binding wall rule was a pose term at or below `0.05`; DQ1 measured `0.07024019179644458`. This does not kill the pose family, does not change the old n600 cheap-warp row, and does not overwrite later pose-in-burn evidence. It only invalidates the old n24-prefix p3v2 free-wall/candidate-line citation as population-fair evidence.

Consumer-impact rows:

| consumer | stale dependency | DQ1 correction |
|---|---|---|
| GC17 | cited p3v2 n24 prefix free-frame row as a pose-wall refutation/candidate line | replace with DQ1 n120 stratified result: p3v2 free upper bound fails the `<=0.05` term rule |
| AU1 | cited p3v2 n24 prefix free-frame row in the same stale direction | replace with DQ1 n120 stratified result and keep the verdict scoped to this formulation |
| p3v2 producer receipt | old n24 prefix headline looked population-relevant | retain as historical prefix evidence only; pair it with DQ1 n120 before any future use |

## GR1 Granularity Rerace

Scope fields: `axis=[macOS-CPU scorer advisory from prior receipts]`, `verdict_scope=FORMULATION for n48 comparative negatives`, `selection_mode=video_order_prefix for n48 receipts`, `selection_mode=full_n600 for cell_drop50 confirmation`, `new_measurement=false`.

DQ1 did not rerun GR1 because the available rerace script exposes `--pairs=48` as `range(0, n_pairs)` and has no stratified selector. Replaying that script would only regenerate the same prefix-shaped comparison the charter asked DQ1 to caveat or replace. The existing n600 `cell_drop50` confirm remains a measured same-object row, but the n48 comparative surface is not population-fair.

| source row | denominator | selection | value | custody SHA-256 | DQ1 disposition |
|---|---:|---|---:|---|---|
| token best `drop27` n48 | 48/600 | video-order prefix | seg+rate `0.851363` | `033215d49dce52b0146cea789e458c8d1bd4fd38000724827524ebc1950a2d2e` | subset-scoped only |
| cell best `cell_drop50` n48 | 48/600 | video-order prefix | seg+rate `0.633895` | `242a4809245f701fa369e96d8f2699a7af5043417035ec096fd4acd28eec064b` | subset-scoped only |
| cell `cell_drop50` confirm | 600/600 | full n600 | archive `359221` B, d_seg `0.004310379028320313`, seg+rate `0.6702284218315308` | `ed88662488492e676e59141a049361ccb6c4b47b73b48aaa81edd831abdb99c1` | survives as the measured same-object row, not a current frontier claim |

Verdict: GR1 is split. The n600 `cell_drop50` measurement survives. The n48 token/cell comparative negatives and nested-rung conclusions are caveated as prefix-subset formulation evidence until a real stratified n48 selector exists.

## ET1 Block16 R8 Accounting

Scope fields: `axis=[macOS-CPU frozen scorer advisory from prior ET1 receipt]`, `verdict_scope=FORMULATION`, `selection_mode=receipt_subset`, `n=32/600`, `steps=25`, `new_measurement=false`, source receipt SHA-256 `4c2d01a7af9bfc1cbff0e6f72188db218a5ea01573e818f5d5434b0a731f47ff`.

DQ1 consumed the existing ET1 block16 steps=25 n32 scorer receipt and performed the mandatory R8 pose-erosion accounting against the `0.0845` pose bank with `+/-0.005` allowance. The block16 geometry/price source SHA-256 was `8b9da08d420bf2b6bb37d4a97bbac5ab27d9ed029385be92f82bc34f5db4a1c1`.

| metric | value |
|---|---:|
| pairs | `32/600` |
| cap-pinned pairs | `31/32` |
| eta pooled | `0.5490603541741959` |
| eta numerator flips / denominator flips | `6077 / 11068` |
| d_pose before mean | `0.0006867947655564421` |
| d_pose after mean | `0.0035952934472334164` |
| pose term before | `0.0828730816101611` |
| pose term after | `0.1896125904900151` |
| pose bank term | `0.08453295485558432` |
| pose erosion vs bank | `0.10507963563443079` |
| R8 allowance | `0.005` |
| d_pose ratio mean / median / max | `11.704030831230188 / 1.9748886655334255 / 123.84641163091061` |
| pairs with ratio > 1.1 / > 1.5 | `27/32 / 20/32` |
| block16 bytes | `46247` |
| rate term | `0.03079397900504106` |
| seg-only net S | `-0.06825022576324405` |
| gross block16 S | `0.1803885565863715` |

Verdict: ET1 remains CAP-CENSORED-STILL and R8 FAIL. The seg-only number is not a verdict because the measured pose erosion is `+0.10507963563443079`, well outside the `+0.005` allowance. This receipt banks no ET1 floor and promotes no ET1 score claim.

## NA2 #923 Pose-Family Queue

Scope fields: `axis=[not measured by DQ1]`, `verdict_scope=FORMULATION queue state`, `new_measurement=false`, NA3 queue-status SHA-256 `933be96c409632c190bcba0725bc7180310b59894b311bd42799c2dbccc86953`.

DQ1 did not run the four #923 pose-family reruns. NA3 had already found that the needed landed harness/render-cache surfaces were missing, so DQ1 preserves them as queued work with fire order instead of pretending the scorer slot made them runnable.

| fire order | item | DQ1 status |
|---:|---|---|
| 1 | `pose_l2_truedepth_stratified_n120_retest` | QUEUED_WITH_FIRE_ORDER_BLOCKED_MISSING_LANDED_HARNESS |
| 2 | `pose_carrier_arms_stratified_n120_retest` | QUEUED_WITH_FIRE_ORDER_BLOCKED_MISSING_LANDED_HARNESS |
| 3 | `pose_mladder_depthwarp_stratified_n120_retest` | QUEUED_WITH_FIRE_ORDER_BLOCKED_MISSING_LANDED_HARNESS |
| 4 | `pose_stratified_texture_stratified_n120_retest` | QUEUED_WITH_FIRE_ORDER_BLOCKED_MISSING_LANDED_HARNESS |

## Evidence And SHAs

| artifact | bytes | SHA-256 | role |
|---|---:|---|---|
| `/Volumes/VertigoDataTier/pact/ddm_dq1_20260805/dq1_p3v2_free_upper_bound_n120_receipt.json` | `4550` | `ba26725cba24c898603e2879ed5759acf7587039feff9111d1b53b209525ac1a` | final DQ1 p3v2 n120 receipt |
| `/Volumes/VertigoDataTier/pact/ddm_dq1_20260805/dq1_p3v2_free_upper_bound_n120.partial.jsonl` | `110059` | `4a6a9128ad0477020ecf58debf6b88084f8f31ec29282f3a0a3d7382c6bc3009` | append-only per-pair p3v2 partial log |
| `/Volumes/VertigoDataTier/pact/ddm_dq1_20260805/dq1_intermediate_aggregation.json` | `8745` | `ed53f9cbca451cc7f1223b0023aac4329d854995625d2b4fdb08c0dfca16be28` | DQ1 aggregation sidecar |
| `.omx/research/ddm_na3_20260805/stratified_pose_selection_923.json` | not copied | `6f7ba6dde77500f09923ecfa7c85a7b977bda1e9f3332e416f6fe05c7bca06f7` | stratified n120 selector |
| `.omx/research/ddm_na3_20260805/pose_family_rerun_status_923.jsonl` | not copied | `933be96c409632c190bcba0725bc7180310b59894b311bd42799c2dbccc86953` | #923 blocked queue state |

## Boundaries

No `upstream/evaluate.py` was run. No n600 authority score was produced. No contest-CPU or contest-CUDA row was produced. No training, launch, or long job was started. No source file or protected path was edited. No staged index entry was touched before the serializer step. DQ1's only new scorer work was the p3v2 n120 bounded CPU PoseNet rerun; GR1, ET1, and #923 are receipt aggregation, scope adjudication, or queue disposition as labeled.

## NEXT_IF_RESUMED

Update GC17/AU1 and any p3v2 consumer memo that cites the old n24 p3v2 free-frame row as a population-fair pose-wall refutation. The replacement source is DQ1's n120 stratified receipt above.

If GR1 comparative verdicts matter, first add a real stratified-pair selector to the rerace script, then rerun both token and cell surfaces at n48 under the same selector. Do not rerun the prefix-only script as if it resolved the charter debt.

For ET1, the next runnable work is a joint Q3/block16 budget sweep with R8 accounting in the loop. Refuse any seg-only ET1 promotion.

For #923, recover or re-land the four missing harness/render-cache surfaces, then execute the NA3 n120 stratified fire order exactly.

```json
{
  "dq1_status": "complete_receipt_no_score_claim",
  "p3v2": {
    "status": "FLIPS",
    "scope": "FORMULATION",
    "n": "120/600",
    "selection_mode": "stratified_blocks",
    "seed": 20260805,
    "receipt": "/Volumes/VertigoDataTier/pact/ddm_dq1_20260805/dq1_p3v2_free_upper_bound_n120_receipt.json",
    "next": "patch consumer memos GC17/AU1 to downgrade old n24 prefix citation"
  },
  "gr1": {
    "status": "SCOPE_CAVEAT_EMITTED_NOT_RERUN",
    "reason": "available rerace script is prefix-only",
    "next": "add stratified selector before any n48 rerace"
  },
  "et1": {
    "status": "CAP_CENSORED_STILL_R8_FAIL",
    "n": "32/600",
    "pose_erosion_vs_bank": 0.10507963563443079,
    "r8_allowance": 0.005,
    "next": "joint sweep with pose/R8 accounting"
  },
  "na2_923": {
    "status": "QUEUED_WITH_FIRE_ORDER",
    "blocked_by": "missing landed harness/render-cache surfaces",
    "fire_order": [
      "pose_l2_truedepth_stratified_n120_retest",
      "pose_carrier_arms_stratified_n120_retest",
      "pose_mladder_depthwarp_stratified_n120_retest",
      "pose_stratified_texture_stratified_n120_retest"
    ]
  }
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.

## 2026-08-05 OA1 correction - p3v2 and ET1 verdict wording

This append-only correction preserves the DQ1 measurements above but supersedes
two verdict readings.

1. `p3v2` at-cap: the n120 stratified 160-iter result above measured pose term
   `0.07024019179644458`, but the same DQ1 partial trace later showed 34/120
   pairs still descending at the cap. MAIN's amendment projected the trace to a
   10x geometric budget at pose term `0.034246` and a conservative tail bound of
   `0.058196`. Corrected statement: p3v2 is BUDGET-CONDITIONAL at 160 iters/pair,
   not a measured final formulation kill.
2. `ET1 block16` and same-genus SQ2 pose reads: the R8 erosion above was measured
   on a seg-corrected intermediate before the terminal pose re-solve / joint
   descent stage. Per the 2026-08-05 staging-law correction, that is expected
   mid-pipeline pose spend, not a final-stage R8 verdict. Corrected statement:
   the seg gain is BANKED as a stage-1 signal pending constrain plus terminal
   pose/joint-descent composition; no seg-only promotion is allowed, but the
   line is reopened for composed-stage measurement rather than killed by this
   intermediate R8 read.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
