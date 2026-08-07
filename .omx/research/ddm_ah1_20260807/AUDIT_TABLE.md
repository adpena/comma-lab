# ddm_ah1 Audit Table

status: COMPLETE-SCOPED
arm: ddm_ah1
utc: 2026-08-07
scope: `.omx/tmp/codex_runs/ah1_prompt.md` plus `_common_contract.md`

## Counts First

| item | count | result |
|---|---:|---|
| Persisted final-message packets read in scoped corpus | 13 | et4, et5, et6, fa1, cons1, cq1, mx1b, mx1c, mx1d, fw1, rr8, rr9, rr10 |
| Review findings audited | 8 | RR8-F1..F4, RR9-F1, RR10-F1..F3 |
| Review findings with a named landed or current consumer | 7 | RR10-F1 remains queued |
| Known charter orphan routed now | 1 | FA1 `StageTransitionSoftVelocityBlend` |
| #254 warning backlog before AH1 | 145 | from mx1c receipt / charter baseline |
| #254 warning backlog after AH1 | 140 | five top-tier raw trainers now call `assert_governed_admission` |
| Scorer / archive / remote launches | 0 | AH1 owned no scorer slot |

## Signal Classifications

| source | signal | classification | consumer / route | status |
|---|---|---|---|---|
| ddm_et4 | First-8 timing plus twelfth-move repair evidence; no n600 row. | CONSUMED-BY / PREMATURE | Consumed by ET5/ET6 fold pricing and CQ1 ET4 canonical equation; n600/eval remains premature without scorer slot. | Routed; no score claim. |
| ddm_et5 | Restricted carriage family priced at 84.476 B/flip, 66.35x W, 0/32 selected. | CONSUMED-BY | CQ1 `ddm_et5_restricted_carriage_family_fold_v1`; AH1 verified callable output and anchors. | Closed as folded family. |
| ddm_et6 | Three-family pricing sweep, all zero strict reproduction. | CONSUMED-BY | AH1 audit only; result is a no-scorer fold, not a launch candidate. | Closed as fold. |
| ddm_fa1 | Rank-1 `ADOPT-CLASS` soft stage-transition blend. | ORPHAN route now | `FA1_STAGE_TRANSITION_SOFT_VELOCITY_BLEND.md` plus `FOLLOWON_LEDGER.jsonl`, consumer trainer `experiments/train_levelset_witness_realized_through_R_mlx.py`. | Queued with backtest. |
| ddm_cons1 | 49 clean rc=0 rows landed; 3 receipt-less rows remained; 12/12 sample routed. | CONSUMED-BY | AH1 read later packets for et5/fw1/mx1c and did not re-open stale receipt-less state. | Closed as harvest state. |
| ddm_cq1 | Five canonical-equation registrations. | CONSUMED-BY | AH1 import/evaluator/provenance verification. | Verified. |
| ddm_mx1b | CPU memory-probe fix; local Metal unavailable. | CONSUMED-BY | mx1c/mx1d hardened the fire path around required receipts. | Instance-level cure only. |
| ddm_mx1c | Safe-run wrapping, #254 guard visibility, fail-closed liveness text; 145 warn-only backlog. | CONSUMED-BY | H4 adopted five top-tier guards; residual 140 remains queued. | Partial class burn-down. |
| ddm_mx1d | Hard MLX cap, keyed mem-probe receipts, fire guard refusal on absent receipt. | CONSUMED-BY | RR9-F1 and RR10-F2 consumer. | Local guard refused absent receipt as designed. |
| ddm_fw1 | ET4/HB1 driver rc propagation plus shell guard. | CONSUMED-BY | RR8-F2 consumer. | Landed cure. |
| RR8-F1 | ET4 repair liveness fallback failed open when `ps` fallback was denied. | CONSUMED-BY | mx1c receipt lines 43/89: successful enumerator required; denied enumerator is never quiescence. | Consumed. |
| RR8-F2 | Original ET4 detached driver emitted false success receipt. | CONSUMED-BY | fw1 driver rc propagation; AH1 H1 launcher survival check; AH1 H3 watcher ALERT for nonzero `.done`. | Consumed and hardened. |
| RR8-F3 | Metal fire scheduling was procedural, not structural. | CONSUMED-BY / residual queued | mx1c/mx1d guarded the concrete mx1 path; AH1 H4 reduced #254 backlog 145 -> 140. Residual raw trainers remain queued. | Partial class burn-down, not fully closed. |
| RR8-F4 | Twelfth-move adjudication mixed rounded d_seg flips and integer receipt flips. | CONSUMED-BY | CQ1 ET4 canonical equation separates source artifacts and computed break-even values; AH1 verified anchors. | Consumed. |
| RR9-F1 | ARM-CAP fire outran required Metal mem-probe receipt. | CONSUMED-BY | mx1d fire guard plus CQ1 `ddm_rr9_mem_probe_fire_protocol_v1`; AH1 verified callable refusal. | Consumed. |
| RR10-F1 | MAIN outran live review charter for same Metal fire. | ORPHAN route now | `FOLLOWON_LEDGER.jsonl` row `ah1.rr10.review_interlock`; no AH1 code touch because charter did not assign this implementation. | Queued gap. |
| RR10-F2 | mx1d guard verdict bypassable by stale/forged passed JSON. | CONSUMED-BY | mx1d `tools/mx1_fire_guard.py` validates receipt schema/status, samples, memory limits, host and config match; entrypoint refuses absent/failed verdict. | Consumed for mx1d path. |
| RR10-F3 | Incident memory overclaimed receipt boundary; no safe_run kill/peak receipt. | CONSUMED-BY | AH1 H2 `tools/safe_run.py --status-receipt` atomic sample/kill receipt. | Consumed. |

