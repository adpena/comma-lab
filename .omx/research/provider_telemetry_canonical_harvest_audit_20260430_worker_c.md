# Provider Telemetry / Canonical Harvest Audit - Worker C - 2026-04-30

Scope: Modal, Vast.ai, and Lightning provider telemetry plus local canonical
harvest state. No MCP tools were used. No provider resources were stopped,
destroyed, or mutated. The only repository write in this pass is this note.

## Commands Run

- `.venv/bin/vastai show instances --raw`
- `.venv/bin/modal app list --json`
- Non-writing Modal `modal.FunctionCall.from_id(...).get(timeout=0)` polls for
  the seven stale/non-harvested Modal call sentinels.
- `.venv/bin/python scripts/scan_lightning_supply_chain.py --strict`
- Read-only `jq` scans of `.omx/state/lightning_batch_jobs.json`,
  `.omx/state/vastai_active_instances.json`,
  `.omx/state/vast_reconcile_after_live_cleanup_20260430.json`,
  `.omx/state/dispatch_holds.json`, and `experiments/results/_modal_harvest_summary.json`.
- Read-only Lightning artifact validation via
  `scripts/launch_lightning_batch_job.py validate-artifacts`.

I intentionally did not run `scripts/launch_lightning_batch_job.py refresh-status`
because it mutates `.omx/state/lightning_batch_jobs.json`.

## Live Provider State

### Vast.ai

- Live CLI truth at `2026-04-30T23:56Z`: `vastai show instances --raw` returned
  `[]`.
- Current live Vast spend: `$0`.
- `.omx/state/vastai_active_instances.json` still has 211 historical tracker
  rows with a meaningless stale estimated-cost sum of `$718.07`. Do not use
  that tracker as live spend truth.
- `.omx/state/vast_reconcile_after_live_cleanup_20260430.json` also records
  `live_count=0`.
- `.omx/state/active_dispatches.md` still contains stale queue-drainer prose
  about Vast relaunches, but the live provider inventory supersedes it.
- `.omx/state/dispatch_holds.json` keeps Lane 19 and Lane 20 relaunches blocked.

### Modal

- Live app truth at audit time: all `comma-train-lane` and `comma-auth-eval`
  Modal apps report `Tasks=0`.
- Current live Modal spend: `$0` by task count. Exact billed cost is not present
  in local state; use the Modal dashboard for historical charges.
- `experiments/results/_modal_harvest_summary.json` has seven not-fully-harvested
  sentinels:
  - `lane_sa_v4`: `not_ready`, call `fc-01KQD5WXJXAK8CV82BFSKPJA7V`
  - `lane_sc_plus_plus_v4`: `not_ready`, call `fc-01KQD5X0CYES4VGRH3KKB3QAKE`
  - `lane_so_v3`: `error_RemoteError`, call `fc-01KQD2AN7CKKHW9H2JEYMZQRKF`
  - `mae_v_v2`: `not_ready`, call `fc-01KQCP43HDQZ9SE53HE90N550A`
  - `q_faithful_v3`: `not_ready`, call `fc-01KQCQS0XBEXZDFWW574FN0W5G`
  - `stc_cuda`: `not_ready`, call `fc-01KQDN5G9VKCR4Z2VPD3VD0PE2`
  - `sz_phase2_v2`: `not_ready`, call `fc-01KQCPZM0D2NMZCTH23RT98SK5`
- Direct non-writing function-call polls returned `not_ready` for the six
  `not_ready` calls and `RemoteError: Function call was cancelled by user` for
  `lane_so_v3`.
- Modal training auth-eval outputs remain CPU advisory because
  `experiments/modal_train_lane.py` forces `AUTH_EVAL_DEVICE=cpu`; do not rank,
  promote, retire, or kill from Modal CPU JSON.

### Lightning

- Local strict Lightning supply-chain scan passed at `2026-04-30T23:55:44Z`:
  `status=OK`, `violation_count=0`, `lightning=null`,
  `lightning-sdk=2026.4.10`.
- `.omx/state/lightning_batch_jobs.json` has 22 records:
  13 `DRY_RUN`, 5 `Failed`, 1 `Stopped`, 3 `HARVESTED`.
- No Lightning record is currently `Running` in local state. I did not run a
  state-mutating refresh.
- Recorded Lightning total cost across non-dry records is `$1.307211119`:
  - Failed: `$0.626111109`
  - Stopped: `$0.16823334`
  - Harvested: `$0.51286667`

## Canonical / Interpretable Lightning Packets

All four local Lightning exact-eval artifact directories below validate with
`--require-adjudication`, contain CUDA/T4 evidence, and have matching archive
SHA/byte identity. None is promotion-clean because every one has a component
gate or regression gate requiring review.

| Job | Local status | Archive | Score | Gate status |
|---|---:|---:|---:|---|
| `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4` | state `Failed`, local artifacts valid | 686557 bytes, `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec` | `1.0378905176070103` | SegNet relative gate violation vs PFP16 T4; `COMPONENT_GATE_REVIEW_REQUIRED`. |
| `pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv` | `HARVESTED`, SDK job `Failed` due adjudication | 686635 bytes, `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f` | `1.037045485927815` | SegNet relative gate violation vs prior PFP16 bundle; usable as paired same-run comparator, not promotion-clean. |
| `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv` | `HARVESTED`, SDK job `Failed` due adjudication | 686468 bytes, `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518` | `1.0373951773937642` | SegNet relative gate violation vs PFP16 reference; component review required. |
| `owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1` | `HARVESTED`, SDK job `Completed` | 686531 bytes, `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91` | `1.0393166493980681` | Regression vs paired PFP16 same-run baseline and PoseNet relative gate violation; measured implementation/config no-go for promotion, not an OWV3 family kill. |

Older failed/stopped Lightning jobs without local canonical artifacts are not
harvestable for score from local state alone.

## Canonical Harvest Commands

Vast live-state check:

```bash
.venv/bin/vastai show instances --raw
.venv/bin/python scripts/reconcile_vast_dispatch_state.py
```

No Vast harvest command is currently actionable because live inventory is empty.
Do not run any `destroy` command from this audit.

Modal stale-call polling/harvest, if an operator wants to retry later:

```bash
.venv/bin/python experiments/modal_recover_lane.py --label lane_sa_v4
.venv/bin/python experiments/modal_recover_lane.py --label lane_sc_plus_plus_v4
.venv/bin/python experiments/modal_recover_lane.py --label mae_v_v2
.venv/bin/python experiments/modal_recover_lane.py --label q_faithful_v3
.venv/bin/python experiments/modal_recover_lane.py --label stc_cuda
.venv/bin/python experiments/modal_recover_lane.py --label sz_phase2_v2
```

Do not expect Modal CPU `contest_auth_eval.json` to be canonical. Any
scientifically interesting Modal archive must be rerun through CUDA exact eval:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <modal_archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir <cuda_evidence_dir>
```

Lightning local validation commands used in this pass:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py validate-artifacts \
  --artifact-dir experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r4 \
  --expected-archive-sha256 e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec \
  --expected-archive-size-bytes 686557 \
  --require-adjudication

.venv/bin/python scripts/launch_lightning_batch_job.py validate-artifacts \
  --artifact-dir experiments/results/lightning_batch/pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv \
  --expected-archive-sha256 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f \
  --expected-archive-size-bytes 686635 \
  --require-adjudication

.venv/bin/python scripts/launch_lightning_batch_job.py validate-artifacts \
  --artifact-dir experiments/results/lightning_batch/owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv \
  --expected-archive-sha256 16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518 \
  --expected-archive-size-bytes 686468 \
  --require-adjudication

.venv/bin/python scripts/launch_lightning_batch_job.py validate-artifacts \
  --artifact-dir experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1 \
  --expected-archive-sha256 9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91 \
  --expected-archive-size-bytes 686531 \
  --require-adjudication
```

If a future Lightning job completes remotely but is not local yet, use the
state-aware SSH harvester instead of broad `scp -r`:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py harvest-ssh \
  --state-path .omx/state/lightning_batch_jobs.json \
  --job-name <job_name> \
  --ssh-target "$LIGHTNING_SSH_TARGET" \
  --require-adjudication
```

## No-Go / Kill Recommendations

- No live Vast, Modal, or Lightning kill action is recommended from this audit.
- Keep Lane 19 and Lane 20 relaunches on forensic hold until their
  `dispatch_holds.json` requirements are cleared.
- Treat `lane_so_v3` Modal as a cancelled call only; no method/family kill.
- Treat Modal CPU artifacts as advisory only.
- Treat `owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1` as a
  measured implementation/config no-go for promotion because it regresses the
  paired PFP16 T4 baseline and violates PoseNet gate. Do not broaden to an
  OWV3 family kill.
- Treat `owv3_byte_feasible...r4` and `owv3_r5...r2` as exact CUDA evidence
  requiring component-gate interpretation, not promotion-ready results.

