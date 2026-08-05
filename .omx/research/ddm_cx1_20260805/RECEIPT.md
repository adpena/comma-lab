# ddm_cx1 Cross-Regime Latched-Constant Audit Receipt

Status: PRE-FIRE GATE INPUT, scorer-free static audit. I ran no training, no scorer, no Metal job, and no archive evaluation. This receipt only audits the v3 finishing-window configuration population requested by the `cx1_prompt.md` charter against the common contract.

Authority boundary: every verdict below is code/config/provenance evidence only. It does not move the frontier, does not select a checkpoint, and does not authorize full FIRE without the named gate evidence.

## RECALL EVIDENCE

Stores consulted before verdicts:

| Store | Query / artifact | Result used |
|---|---|---|
| Pact memory registry | `cross-regime`, `jd1`, `jd3`, `gc19`, `#366`, `#888`, `hinge` | No direct `ddm_cx1` precedent found. Recall did confirm the old #366 DESCENT residual-finisher context, so this arm remains a pre-fire gate input, not a standalone endpoint. |
| Pact hot state | `.omx/state/main_hot_state.md` | Latest live state superseded the local jd3 no-Metal blocker: MAIN now owns the v3 smoke chain, and cx1 is explicitly listed as the fire-gate table input. |
| JD1/JD3 receipts | `ddm_tp1_boundary_receipt_20260805.md`, `ddm_jd3_20260805/RECEIPT.md`, `NEXT_IF_RESUMED.md`, `MAIN_ADDENDUM_OPTIMAL_CONVERSION.md` | v2 failed by loss-space hold plus window-mis-scoped EMA. v3 implemented realized-space hold, stage-scoped EMA, and live gate telemetry. SL2/PE3/EN1 are v4 riders, not v3-stacked fixes. |
| Claude memory | `cross_regime_constant_transfer_genus_finishing_stage_20260805.md` | Genus definition: constants or controls derived in one regime and consumed in another. Known subcases are scale, space, and horizon transfer. |
| Canonical equation registry | `ema_decay_substrate_stage_aware_v1` | Stage EMA must be re-derived from the active finishing window, not inherited from the parent tau horizon. |
| Source search | `experiments/train_tr1_partition_renderer_mlx.py`, `src/tac/optimization/lane_guard.py`, ddm_en1 and lane-guard research receipts | Confirmed v3 callsites, defaults, rollback path, optimizer-state persistence, lane-guard ratchet implementation, and EN1 margin-weight deferral. |
| Chain sweep artifact | `chain_both_bases_sweep.json` | Existing static selection input is finalized as `killed_rc_neg15_7_of_8_rows_valid`; row schema is `ckpt/live/ema/live_minus_ema`, not a full exact-score table. |

What changed beyond the charter seeds:

- The live MAIN state changed while this arm was running: jd3 is no longer merely blocked on this local host; MAIN has launched the v3 smoke chain on a Metal host.
- EN1 margin-weight is now built as a tau consumer, but the live addendum keeps it queued behind v3 adjudication.
- Lane-guard ratchet is built but not enabled in the v3 tickets; this is the main additional cross-regime risk found by recall and source inspection.

## AUDIT POPULATION

Primary v3 tickets audited:

- Entry smoke: `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_entry_ep1336.json`
- Refuse-final smoke: `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/jd3_ticket_v3_refuse_final_ep1354.json`

Primary checkpoints audited:

- Entry checkpoint: `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/tr1_joint_pose_finish_from_full_birth_lane_on_w4m/checkpoints/stage_joint_pose_finish_entry.npz`
- Final checkpoint: `/Volumes/VertigoDataTier/pact/ddm_jd1_20260805/tr1_joint_pose_finish_from_full_birth_lane_on_w4m/checkpoints/stage_joint_pose_finish_final.npz`

Both checkpoints are already inside `joint_pose_finish`, with `jd1_pose_finish.engaged=true`, floor `0.47710885961850485`, `batch_pairs=4`, `num_pairs=600`, `gate_every=5`, parent `ema_decay=0.999960019990005`, and optimizer state arrays present. Because they are already engaged, v3 inherits the JD1 v2 own-form child floor rather than recomputing it from a pre-engagement parent segment.

## DERIVED STATIC ARITHMETIC

| Quantity | Entry ticket | Refuse-final ticket | Verdict |
|---|---:|---:|---|
| checkpoint epoch | 1336 | 1355 | known from checkpoint metadata |
| first resumed epoch | 1337 | 1356 | deterministic resume geometry |
| ticket terminal epoch | 1345 | 1364 | 8 run epochs each |
| updates per epoch | 150 | 150 | `600 // 4` |
| run updates | 1200 | 1200 | window-local stage scope |
| planned gates | 1339, 1344 | 1359, 1363 | two-gate smoke, not full-fire seal |
| stage EMA decay | 0.9966666667 | 0.9966666667 | `derive_jd1_stage_ema_decay(8, 150)` |
| parent EMA warmup | about 50025 updates | about 50025 updates | horizon-mismatched for 1200-update window |
| v3 EMA warmup | about 600 updates | about 600 updates | window-matched |
| realized-hold margin | first-gate `sd(per_pair_d_seg)/sqrt(n_gate)` | same | unknown until latch gate emits |
| max retreats | 2 | 2 | `0` derives to `A1_CONSECUTIVE_REFUSE` |
| pose retreat factor | 0.5 | 0.5 | `0.0` derives to bisection retreat |

## PER-ELEMENT VERDICT TABLE

| Element | v3 value | Regime-risk check | Verdict | Route |
|---|---|---|---|---|
| `jd1-seg-hold-space` | `realized` | Converts #888 from loss-space to realized-gate quantity. Latch is first post-engagement realized gate on the same smoke. | OK-BY-BUILD, needs latch evidence | FIRED in v3; full FIRE requires emitted `jd1_realized_hold_latch` rows for both smokes. |
| `jd1-live-gate-telemetry` | `on` | Required for realized-space validation and live/EMA separation. | OK-BY-BUILD | FIRED in v3; missing telemetry blocks selection. |
| `jd1-ema-stage-scope` | `window` | Re-anchors EMA at resume and derives 0.9966666667 from the 1200-update finishing window. | OK-BY-BUILD | FIRED in v3; fire gate must confirm `stage_window` re-anchor rows in both smoke receipts. |
| inherited JD1 floor | `0.47710885961850485` | Floor was calibrated in the margin-OFF child form before v2 joint finish; v3 resumes after engagement. | OK-AT-SCOPE for these checkpoints | FOLDED. If a future ticket resumes before engagement, repeat one own-form calibration epoch. |
| realized-hold margin | `0.0` flag, derived at latch | Derived from first gate's realized per-pair scatter, so it is not a parent constant. | UNKNOWN-NEEDS-MEASUREMENT until first gate | FIRED in v3; full FIRE requires latch row with `n_gate`, `sd`, and derived margin. |
| max retreats | `0.0` flag, derives to 2 | Controller budget tied to A1 consecutive-refuse count, not parent training horizon. | OK-AT-SCOPE | FIRED in v3; if rollback happens, require a post-rollback gate before checkpoint selection. |
| pose retreat | `0.0` flag, derives to 0.5 | Bisection controller, not transferred from parent schedule. | OK-AT-SCOPE | FIRED in v3; if it fires, consume effective `w_pose` history before route decision. |
| `jd1-w-pose` | `1.0` | v2 proved it can move pose quickly and also harm realized seg. v3 adds hold/retreat, but 1.0 itself remains high-gain. | UNKNOWN-NEEDS-MEASUREMENT | FIRED in v3 only under realized hold. If first post-latch check rolls back or exhausts wall time, QUEUE lower initial `w_pose` or explicit retreat arm. |
| `jd1-seg-hold-weight` | `0.25` | In realized mode this is an enablement validation value, not the realized-gate actuator. | OK-AT-SCOPE / NOT-ACTUATOR | FOLDED. Do not tune this expecting realized-hold strength. |
| lane guard constant budget | `--lane-guard` on, ratchet off | Current constant budget remains slack in finishing-window telemetry; ratchet implementation exists but is absent from both v3 tickets. | MIS-SCOPED FOR FULL-FIRE | QUEUED-WITH-FIRE-ORDER: if smoke shows Lane give-back or lambda slack with d_seg harm, run a separate ratchet-on v3b smoke before full FIRE. |
| lane guard eta / step cap | derived defaults | Eta affects multiplier speed only when the guard is active. With ratchet off and lambda slack, it is not the current binding actuator. | UNKNOWN-LOW-PRIORITY | FOLDED for v3 primary; revisit only with ratchet-on evidence. |
| gate cadence | `gate_every=5`, two gates per smoke | Enough to latch and perform one post-latch check, not enough to certify the full controller trajectory. | OK FOR SMOKE, NOT FULL-FIRE SEAL | FIRED as smoke; full FIRE requires both smoke receipts plus chain sweep adjudication. |
| wall cap | `23` minutes | Eight epochs fit the measured v2 epoch cost with room for one likely rollback replay, but repeated retreats may hit the cap. | OK FOR SMOKE | FIRED. If rollback consumes cap before post-rollback gate, selection is BLOCKED. |
| optimizer state | `persist-optimizer-state on` | Checkpoints contain optimizer arrays and resume code restores them. | OK-BY-BUILD | FIRED and required. |
| deterministic R | absent | Smoke is advisory, but full FIRE reproducibility cannot rest on unstated nondeterminism. | UNKNOWN FOR FULL-FIRE | QUEUED-WITH-FIRE-ORDER: before long full FIRE, either enable deterministic R on Metal or record an explicit accepted noise-floor/no-determinism decision. |
| margin-weight consumer | absent from v3 | EN1 built tau margin weighting, but MAIN/JD3 addendum defers it. | QUEUED, NOT V3 | QUEUED-WITH-FIRE-ORDER after v3 adjudication; do not silently stack into v3. |
| SL2 teacher / PE3 conditioning | absent from v3 | Main addendum classifies both as v4 riders. | QUEUED, NOT V3 | QUEUED-WITH-FIRE-ORDER after v3 adjudication. |
| grad clipping | no active flag/callsite | Source has gradient-norm telemetry/alarm but direct optimizer update has no clipping actuator. | NOT-IN-POPULATION | FOLDED. Only design a grad-control arm if v3 logs nonfinite or repeat gnorm alarms. |
| boundary probe / full confirm | absent | Dropped deliberately from smoke ticket. MAIN owns post-smoke confirm/exact routing. | OK FOR SMOKE | FOLDED for cx1; full-FIRE gate still requires MAIN confirmation. |
| frozen model geometry | inherited checkpoint geometry | Changing grid/code/width/quant constants would invalidate the resumed checkpoint population. | OK-AT-SCOPE | FOLDED. Not a finishing-window latch. |

## FIRE-GATE FLAGS

1. CX1-FG1 `lane_guard_ratchet`: v3 primary may smoke as launched, but full FIRE is blocked if both smokes show slack lane guard plus Lane/seg give-back. The next action in that case is a separate ratchet-on v3b smoke, not an unlogged flag mutation of the live primary.
2. CX1-FG2 `realized_hold_rows`: both smokes must emit latch and at least one post-latch realized-gate row. Missing rows, missing per-pair scatter, or missing live/EMA separation blocks selection.
3. CX1-FG3 `rollback_receipt`: if realized hold breaches, require rollback evidence, restored epoch basis, effective `w_pose`, and a post-rollback gate before selecting the checkpoint.
4. CX1-FG4 `deterministic_full_fire`: before any long full-FIRE run, either enable deterministic R on the Metal host or record an explicit noise-floor decision. Exact promotion remains contest-CPU/CUDA authority only.
5. CX1-FG5 `v4_riders`: EN1 margin-weight, SL2 teacher distill, and PE3 conditioning are queued after v3 adjudication. They must not be silently added to v3 smoke/full-fire labels.

## SHA-256 TABLE

| Artifact | SHA-256 |
|---|---|
| `cx1_prompt.md` | `b6f8943e2576a60a980e7ee682d1abfc6a8e1f3e14a984ed4ecbd95a027d9e9a` |
| `_common_contract.md` | `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771` |
| `.omx/research/ddm_tp1_boundary_receipt_20260805.md` | `1b9787a308f1cf814143e457234ddd5e9d4c24471b1c0c442fa42be2e675a275` |
| `.omx/research/ddm_jd3_20260805/RECEIPT.md` | `79b51477570c49d743ff94261c65bf1ab32bf485fb4c7106a758cc0cf8629128` |
| `.omx/research/ddm_jd3_20260805/NEXT_IF_RESUMED.md` | `e953d404a42b8dbf7644d181f24a236208a4a2349d90d71a78aff968f8dfa653` |
| `.omx/research/ddm_jd3_20260805/MAIN_ADDENDUM_OPTIMAL_CONVERSION.md` | `97b53deee23d149648bc6f4293e39f34d8a8c118f3a76d08a109b18286d0a455` |
| `experiments/train_tr1_partition_renderer_mlx.py` | `bf7190ceaecf0ad168bce6c4c843015b123b362ea48158fc645c089e8d065e75` |
| `src/tac/optimization/lane_guard.py` | `db49c0d6fd1f82242d5eb45616622da3b4d798387df1ed875db600e807e6580a` |
| `jd3_ticket_v3_entry_ep1336.json` | `cdbf2338c56082a56e1bd4d79d0f0b9e7f2be0d57e12539a71ab17666c35b1b2` |
| `jd3_ticket_v3_refuse_final_ep1354.json` | `71dc8c75f7c6c6123ddd28a36a619ddc8036deedaf1324e1b514db1762eee376` |
| `chain_both_bases_sweep.json` | `a3bfb6266774c3c2084fd3aaa4756aa76c785181eb52d70d327e4b6fdda005fa` |
| `stage_joint_pose_finish_entry.npz` | `f10eb595417e6e2ee51809e3479e2a69d1e9dbe220671e4174a8fd39c371ff77` |
| `stage_joint_pose_finish_final.npz` | `8cd6128f522f7e0fd361700f0dbe6a416a88131c9c66aa617394b56bd540b133` |

## BOUNDARIES

- No `.py` file was edited by this arm, so `REVIEW_GATE_OVERRIDE=1` was not used.
- The protected hot file `src/tac/optimization/direct_description_carrier_compose.py` was dirty before this arm and was not touched.
- No scorer slot was claimed and no scorer artifact was produced.
- No archive was built or evaluated.
- The contest pointer remains borrowed/unmoved.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
