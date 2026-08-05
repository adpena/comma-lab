# AL1 Receipt - L1 Alarm Calibration Registry

Date: 2026-08-05

Verdict: **FIRED scoped apparatus landing.** AL1 adds a typed L1 alarm registry plus an in-tree split-conformal p-value and Benjamini-Hochberg FDR utility. It reproduces lp1's lane-guard ratchet false-positive verdict from the banked null construction: observed sum-rises `0.029133`, regenerated null mean `0.050560064`, low-tail percentile `0.0072`, high-tail conformal p-value `0.99280036`, BH q-value `0.99280036`, alarm fire `false`.

No frozen scorer forwards, no `upstream/evaluate.py`, no n600 replay, no launch, no paid dispatch, no threshold recalibration, no protected-file edit, and no live trainer/supervisor behavior change were performed.

## Artifacts

| Artifact | Purpose |
|---|---|
| `src/tac/alarm_calibration.py` | Typed registry rows, split-conformal p-values, BH results, registry-consuming alarm-family adjudicator, and lp1 reproduction helper. |
| `src/tac/tests/test_alarm_calibration.py` | Positive/negative conformal tests, BH tests, synthetic exchangeable-null super-uniformity sanity check, registry-query test, and lp1 reproduction test. |
| `.omx/research/ddm_al1_20260805/al1_l1_alarm_registry.json` | Machine-readable registry export. |
| `.omx/research/ddm_al1_20260805/lp1_lane_guard_reproduction.json` | Machine-readable reproduction of the lp1 banked false-positive verdict. |

## Registry Rows

| Alarm | Score direction | Calibration population | Exchangeability grade | Block calibration | FDR family | Consumer | Falsifier |
|---|---|---|---|---|---|---|---|
| `lane_guard.ratchet` | high `sum_rises_s_units` | same vehicle/window no-erosion Lane realized series; lp1 iid-noise null banked at MC n=20000 | conditional if stationary window; fragile across stages | yes | `lane_guard` | #934 successor, b4s burn reseal, lane guard fire order | held-out or block-calibrated null p-values are not super-uniform |
| `term_domination` | high non-scored share or scored-term deficit | same-stage term-share rows under same vehicle and schedule | partial, stage-scoped | yes | `loss_term` | v9 telemetry port and burn supervisor | same-stage null rows fail calibration or fire is a stage transition |
| `term_inert` | high sustained movement deficit/residual debt | engaged historical rows scoped by stage and vehicle | partial; block calibration required | yes | `loss_term` | force-stack and curriculum gates | alarm disappears under block calibration or null p-values bunch low |
| `gnorm_hijack` | high excess norm/share | same-stage gradient norm shares with matched optimizer controls | partial and fragile | yes | `gradient_health` | force caps and optimizer watchdogs | nonstationary gradients invalidate calibration before the fire |

A1 was checked and intentionally excluded from the machine registry: current source inspection found `experiments/train_tr1_partition_renderer_mlx.py::realized_gate` still emits `realized_gate_dseg_mean = np.mean(dsegs)`. GD1 specifies the HT/per-pair repair, but AL1 did not find it live in the trainer path, so `A1_REALIZATION_GAP_ALARM` is queued as blocked conditional work instead of registered/calibrated.

## Reproduction

The lane-guard ratchet worked example uses `tac.alarm_calibration.lp1_lane_guard_ratchet_null_reproduction()`:

| Quantity | Value |
|---|---:|
| observed `sum_rises_s_units` | 0.029133 |
| observed rise count | 22 |
| null trials | 20000 |
| null seed | 777 |
| null mean, sum rises | 0.05056006413412676 |
| null p5-p95, sum rises | 0.03576021020214661 - 0.06645387019882161 |
| low-tail percentile, sum rises | 0.0072 |
| conformal high-tail p-value | 0.9928003599820009 |
| BH q-value | 0.9928003599820009 |
| alarm fires | false |

Interpretation: lp1's observed give-back statistic is in the low tail of the no-erosion null, while the alarm direction is high rises. The registry-consuming conformal/BH path therefore suppresses the elapsed-horizon false positive.

## RECALL EVIDENCE

Sources searched:

| Scope | Queries / files | Finding beyond charter seeds | Plan impact |
|---|---|---|---|
| Governing context | `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | AL1 is apparatus only; no scorer slot, no promotion authority, own-vehicle frontier unchanged. | Kept work scorer-free and receipt-led. |
| CF1 seed | `.omx/research/ddm_cf1_20260805/CF1_CROSSWALK_RECEIPT.md`, `cf1_crosswalk_receipt.json` | CF1 folds `nonconform` import and asks for internal split-conformal p-values plus BH; per-alarm rows must carry exchangeability and falsifiers. | Implemented internal core and JSON registry. |
| Lane ratchet | `.omx/research/ddm_lp1_lane_program_20260803.md`, `src/tac/optimization/lane_guard.py`, `src/tac/tests/test_lane_guard.py` with queries `sum_rises`, `0.029133`, `20000`, `percentile`, `ratchet_horizon` | lp1 already established the false positive: true horizon 64 gives 0/64 fires; observed rises are low-tail under the iid-noise null. Raw 20k null vector was not found in bounded repo/SSD scope; shipped null construction is durable in code/memo. | Added deterministic regeneration and recorded the source boundary. |
| A1 repair check | `rg "realized_gate|Horvitz|HT|A1_REALIZATION"` across trainer, `src/tac`, and GD1 memo | GD1 specified the HT/per-pair estimator, but current trainer source still emits unweighted `np.mean(dsegs)` for `realized_gate_dseg_mean`. | A1 was excluded from the registry until the statistic repair lands. |
| Alarm consumers | `experiments/train_tr1_partition_renderer_mlx.py`, `src/tac/witness_control/telemetry_producers.py`, `tools/supervise_ddm_b4s_burn4.py`, `src/tac/tests/test_ddm_tp1_v9_telemetry_port.py` with queries `term_domination`, `term_inert`, `gnorm_hijack`, `UNDRIV_EROSION`, `burn supervisor` | Existing consumers are live trainer/supervisor surfaces. A mid-run hook would alter launch/adjudication behavior and exceed AL1's "registry + utility + reproduction only" scope. | Added `adjudicate_alarm_family()` as the first registry-consuming API and queued live wire-in with a named recipient. |
| Sampling/exchangeability | `.omx/research/ddm_na3_20260805/ddm_na3_receipt.md` | Prefix sampling can invert bias by axis; calibration sets must match the target alarm population. | Registry rows explicitly flag block/stage requirements and fragile exchangeability. |

## Tests

Command:

```bash
.venv/bin/python -m pytest src/tac/tests/test_alarm_calibration.py -q
```

Result: `8 passed`.

Additional focused checks:

```bash
.venv/bin/python -m ruff check src/tac/alarm_calibration.py src/tac/tests/test_alarm_calibration.py
```

Result: `All checks passed!`

## Follow-Ons

| Follow-on | Disposition | Fire order |
|---|---|---|
| `cf1_l1_alarm_calibration_table` | **FIRED** | Registry is in `src/tac/alarm_calibration.py` and exported to `.omx/research/ddm_al1_20260805/al1_l1_alarm_registry.json`. |
| `cf1_calibrate_lane_guard_ratchet_null` | **FIRED for banked lp1 reproduction; QUEUED-WITH-FIRE-ORDER for live block calibration** | After #934 existence-hinge A/B or the next live lane-guard rise, feed the matched live/block null into `adjudicate_alarm_family(..., fdr_family="lane_guard")`; do not use the iid diagnostic null as production block calibration. |
| `import_nonconform_now` | **FOLDED** | No dependency added; revisit only if weighted conformal, FDP certificates, or martingales become active enough to justify the Python-version/dependency surface. |
| `confound_burn_supervisor_wire_in` | **QUEUED-WITH-FIRE-ORDER** | Recipient: b4s burn-supervisor/confound-gates successor. At the next safe supervisor/trainer boundary, replace direct threshold-only adjudication for calibrated alarm families with `adjudicate_alarm_family`; do not patch a live hot trainer/supervisor mid-run. |

## Boundaries

`adjudicate_alarm_family()` is a real registry-consuming path and is tested, but it is not yet installed into the live hot supervisor/trainer. This is intentional: AL1's charter forbids threshold recalibration and limits the landing to registry, utility, and reproduction unless a non-hot consumer can be wired safely.

The lp1 reproduction regenerates the durable iid-noise null construction from the memo/code constants because no raw 20,000-sample null vector was found in the bounded scopes searched. It reproduces the decision class and the published null summary within tolerance; it is diagnostic apparatus evidence, not a scorer or score row.

## NEXT_IF_RESUMED

1. At the next non-hot burn-supervisor/confound-gates boundary, wire `adjudicate_alarm_family()` into the alarm-fire path for `lane_guard.ratchet`, then extend to `term_domination` / `term_inert` / `gnorm_hijack` when their matched nulls exist.
2. Do not add or activate `A1_REALIZATION_GAP_ALARM` calibration until the HT/per-pair estimator is present in the trainer telemetry.
3. If live temporal nulls remain autocorrelated, implement block calibration before any alarm-fire disposition is allowed to depend on a p-value.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
