# ddm_cvg1 - lw1 Control-Variate Gate Replay

Date: 2026-08-06

Arm: `ddm_cvg1`

Status: COMPLETE. Scorer-free, banked-rows-only replay. No launch, no scorer
run, no paid dispatch, no `upstream/` edit, no score claim.

Structured table: `.omx/research/ddm_cvg1_20260806/replay_table.json`
(`45,012` bytes, sha256
`b8a5e374b99c63f26bebf4b587a3ffc46724cc0452f045f1fa935a13cbe66ced`).

## Denominator First

| Axis | Endpoint windows checked | Gate-truth joinable EMA rows | CV rows with previous-endpoint control | Verdict |
|---|---:|---:|---:|---|
| `d_seg` | 6 | 6 | 5 | FALSIFIED for tested controls |
| `d_pose` | 6 | 3 | 3 | INSUFFICIENT_CONTROL_VARIATION |
| rate | 6 | 0 | 0 | BLOCKED_NO_JOINABLE_RATE_ROWS |

`d_seg` loses the lw1 ADOPT test: the same-axis previous-endpoint control
worsened held-out RMSE/MAE, while the previous-pose-bias control improved RMSE
but worsened sign decisions. `d_pose` has the known gate36 subset-easy anchor
(`jd7on`: gate `0.002177761877` vs n600 `0.020818391064`, n600/gate `9.56x`),
but all three pose rows share the same previous endpoint control, so residual
correlation is undefined and the side-information admission bar is not met.

Verdict scope: INSTANCE / FORMULATION for this jd4/jd7/jd8q3 banked gate36
replay. This is not a family-level kill of control-variate-corrected gates.

## Replay Metrics

Estimator:

```text
corrected = gate_estimate - beta * (control_subset - control_n600)
```

`beta` is fit by leave-one-window-out on the training rows. The admissible
control is a previous endpoint's same-basis gate36 subset bias, because it was
known at the next window's gate time. The current endpoint's own n600 slice is
not used as a control.

| Axis | Control | n | residual corr | raw RMSE | corrected RMSE | raw sign errors | corrected sign errors | beta stability |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `d_seg` | previous same-axis EMA | 5 | `0.3332` | `0.0006468` | `0.0008917` | 1 | 1 | LOO `[-5.609, 2.523, 4.274, 4.780, 5.575]`; boot p05/p50/p95 `[-17.264, 3.550, 7.257]` |
| `d_seg` | previous `d_pose` bias | 5 | `-0.2804` | `0.0006468` | `0.0003626` | 1 | 2 | LOO `[-0.384, -0.419, -0.500, -0.445, -0.359]`; boot p05/p50/p95 `[-0.555, -0.422, -0.278]` |
| `d_pose` | previous same-axis EMA | 3 | undefined | `0.0129202` | `0.0069914` | 3 | 0 | LOO `[-9.246, -5.932, -9.318]`; boot p05/p50/p95 `[-10.422, -8.165, -5.908]` |
| `d_pose` | previous `d_seg` bias | 3 | undefined | `0.0129202` | `0.0069914` | 3 | 0 | LOO `[-434.593, -278.852, -437.979]`; boot p05/p50/p95 `[-489.892, -383.808, -277.723]` |

Side-information admission:

- `d_seg` same-axis control: rejected; held-out residual reduction is negative.
- `d_seg` previous-pose control: rejected; RMSE improves but sign decisions get
  worse (`1 -> 2`).
- `d_pose` controls: not admitted; the three rows all use the same previous
  endpoint (`jd6`) so `control_subset - control_n600` is constant and cannot
  establish residual correlation.
- rate: no row; endpoint receipts have no same-object n600/full-population rate
  truth, and telemetry `total_counted_bytes` is exact model telemetry rather
  than a sampled gate estimate.

Gate36 remains a different population in this replay. The pose anchor above is
the clearest case; `d_seg` gate rows also systematically underestimate n600 on
the tested windows.

## Row Join

Rows were joined only when `run_id`, window, EMA basis, gate36 pair list, and
endpoint receipt identity were explicit. Live-basis n600 rows were not joined
because no live-basis a1 telemetry row exists in these receipts.

Primary sources:

- `experiments/ddm_jd4_endpoint_n600_both_bases.py`
- `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd4_endpoint_n600_both_bases.json`
- `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd5_endpoint_n600_both_bases.json`
- `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd6_endpoint_n600_both_bases.json`
- `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd7off_endpoint_n600_both_bases.json`
- `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd7on_endpoint_n600_both_bases.json`
- `/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd8q3_endpoint_n600_both_bases.json`
- matching `tr1_window_receipt.json` files under the six `tr1_jd4_cont_*`
  windows.

Missing rows:

- `jd4 d_seg`: no previous endpoint per-pair control; baseline `ep1405` exists
  only as summary means in the current source surface.
- `jd4`, `jd5`, `jd6 d_pose`: final window receipts lack
  `realized_gate_dpose_mean`.
- live basis: no live-basis gate telemetry.
- rate: no same-object population-rate truth row.

## Corrected-Gate Protocol

Do not wire a correction into `src/tac/subset_selection_gate.py` from this
replay. A future owner may wire only after a positive row satisfies all of:

1. Same axis, same basis, same adapter, same selection mode, explicit pair ids.
2. A control known before the target gate's n600 truth is measured.
3. Non-constant control deltas with held-out residual correlation reported.
4. Held-out RMSE/MAE and sign-decision improvement versus the raw gate.
5. Axis-specific coefficients and uncertainty bands; no pooled pose/seg/rate
   beta.

## RECALL EVIDENCE

| Query/source | Evidence found beyond charter seeds | What changed |
|---|---|---|
| Governing files: `.omx/tmp/codex_runs/cvg1_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`/`AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Arm is scorer-free, upstream read-only, live scorer slot owned elsewhere, final receipts required under this directory. | Kept to banked-row replay and report-only artifacts; no scorer, launch, or protected edit. |
| `.omx/research/ddm_lw1_20260806/{RECEIPT.md,NEXT_IF_RESUMED.md}` | lw1 requires leave-one-window-out beta, side-info residual-correlation admission, per-axis adjudication, and honest denominators. | Used previous endpoint controls only; refused current endpoint n600 slices as truth leakage. |
| `experiments/ddm_jd4_endpoint_n600_both_bases.py` + jd endpoint/window receipts | The endpoint receipts carry n600 per-pair arrays and gate36 positive controls; window receipts carry final a1 gate rows, but only later windows have d_pose gate telemetry. | Built the replay table from jd5/jd6/jd7off/jd7on/jd8q3 for `d_seg`, and jd7off/jd7on/jd8q3 for `d_pose`. |
| Targeted search: `rg -i "control[- ]?variate|variance reduction|side information|residual correlation|subset bias|gate36|Lam/Wang|2608.04312|lw1_control_variate"` over `.omx/research`, docs, reports, code | Found lw1, unrelated older side-information/cached-replay references, and no already-landed gate-vs-n600 CV correction in searched scope. | Treated this as the first execution of the lw1 retro-test, not a duplicate. |
| Canonical index/DAG targeted search for `ddm_cvg1`, `lw1_control_variate`, `control-variate`, `gate36` | No pre-existing CV gate replay found; only a broad older ES control-variate mention and unrelated side-information entries. | No canonical equation registered; positive registration would be premature. |

Bounded absence statement: these searches do not prove global nonexistence; they
cover the named scopes and queries.

## Boundaries

Measured here: banked-row replay metrics, row-join denominators, LOO betas,
bootstrap coefficient bands, residual correlations, and sign-decision errors.

Not measured here: any new scorer value, archive byte-closed score, exact eval,
contest CPU/CUDA row, or rate correction.

Own-vehicle frontier unchanged:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer
borrowed/unmoved at `0.1910828242`.
