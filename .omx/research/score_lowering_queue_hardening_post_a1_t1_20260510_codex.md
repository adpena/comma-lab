# Score-lowering queue hardening after A1 regression + active T1 guard

Generated: `2026-05-10T08:37:25Z`

`research_only=true`; `score_claim=false`; `dispatch_attempted=false`.

This ledger supersedes only queue interpretation. It does not modify runtime,
trainer, dispatcher, or provider code. It cites
`.omx/research/current_score_lowering_roadmap_20260510_codex.md` as an input,
but that file is intentionally not edited here because the operator is patching
it locally.

## Inputs inspected

- `CLAUDE.md`, `AGENTS.md`, and `PROGRAM.md` preflight instructions.
- Top memory index entries for Pact diagnostic/dispatch discipline.
- Recent directives:
  - `.omx/research/codex_review_round7_round8_findings_directive_20260509.md`
  - `.omx/research/defer_t9_directive_for_a0be36e_20260509.md`
  - `.omx/research/hnerv_forensics_author_repo_search_directive_20260509.md`
- Today ledgers:
  - `.omx/research/a1_modal_score_gradient_regression_20260510_codex.md`
  - `.omx/research/t1_modal_guard_dispatch_20260510_codex.md`
  - `.omx/research/t1_modal_actuator_20260510_codex.md`
  - `.omx/research/t1_score_domain_training_wirein_20260510_codex.md`
  - `.omx/research/pr101_score_gradient_t8_gradguard_wirein_20260510_codex.md`
  - `.omx/research/kaggle_proxy_sweep_substrate_20260510_codex.md`
  - `.omx/research/cloud_provider_readiness_20260510_codex.md`
  - `.omx/research/current_score_lowering_roadmap_20260510_codex.md`
- Task/briefing surfaces:
  - `.omx/state/dispatch_queue.md`
  - `.omx/research/roadmap_active_queue_refresh_20260509_codex.md`
  - `.omx/research/roadmap_queue_adversarial_review_20260509_codex.md`
  - `tools/operator_briefing.py` output

## Current authoritative state

Live dispatch claim summary:

```text
CLAIM_SUMMARY active=1 stale_nonterminal=0 terminal_latest=565 unparsable_timestamp=0
ACTIVE lane_id=t1_balle_128k_endtoend job=t1_balle_modal_guard_a3311268_20260510T0831Z platform=modal status=active_dispatching agent=codex:modal_t1_balle_endtoend
```

The active T1 guard metadata:

- Lane: `t1_balle_128k_endtoend`
- Job: `t1_balle_modal_guard_a3311268_20260510T0831Z`
- Modal call id: `fc-01KR8GACB3NCW5TNG1E9YFPXHM`
- GPU: Modal `T4`
- Guard run: `epochs=50`, `batch_size=8`, `max_target_pairs=64`
- Score semantics at dispatch: `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`
- Mounted code snapshot: clean at
  `a3311268f0a7a7547a32ae16f85ad3466e6de579`

A1 exact-CUDA regression is terminal and should not be rerun as configured:

- Job: `track1_phase_a1_score_gradient_modal_20260510T0738Z_codex`
- Archive SHA-256:
  `f5d04f22d46bc1c4b863e9e2989c25f9b04e07cb21d54980b5effb654edc127a`
- Archive bytes: `206110`
- `[contest-CUDA]` score: `0.5447505505333358`
- Components: `seg=0.00336345`, `pose=0.00050645`,
  `rate_unscaled=0.00548961`, `n_samples=600`
- Classification: measured-config regression, not a family kill.

## Evidence-axis separation

- `[contest-CUDA]`: strict promotion/regression axis. Requires exact CUDA
  auth eval on a byte-closed archive, `n_samples=600`, auth-eval schema
  blockers at zero, archive/runtime SHA custody, and terminal dispatch claim.
- `[contest-CPU]`: Linux x86 public-leaderboard reproduction axis. Useful for
  public policy and CPU/CUDA split diagnosis, but not a CUDA-frontier claim.
- `MPS` / local macOS CPU: advisory only for sweeps, smoke checks, and
  configuration discovery. Never promote, rank, or kill from this axis.
- `Kaggle`: proxy/config-search only in current tooling. It may generate
  candidate configs, but no archive, no exact auth eval, and no score claim.

## Ranked next score actions

| Rank | Action | Axis | Status | Gate before acting |
|---:|---|---|---|---|
| P0 | Recover active T1 Modal guard and classify exact outcome. | `[contest-CUDA]` if recovery passes strict schema | IN FLIGHT | Do not dispatch another `t1_balle_128k_endtoend` job while the active claim exists. Recover only. |
| P1 | If T1 produces a valid exact-CUDA artifact, run adversarial packet review and plan paired `[contest-CPU]` reproduction before promotion/submission. | `[contest-CUDA]` then `[contest-CPU]` | BLOCKED on P0 result | Auth-eval blockers zero, `n_samples=600`, archive/runtime SHA match adjudication, no-op proof present, terminal claim closed. |
| P2 | If T1 regresses or is non-claimable, retire only the measured T1 guard configuration and use its packet logs to tighten rate/state-dict and proxy/exact gates. | `[contest-CUDA]` result review | BLOCKED on P0 result | Preserve train logs, packet compiler manifest, archive bytes/SHA, auth-eval JSON, and failure class. Do not kill T1 family from a 64-pair guard. |
| P3 | Run Kaggle PR101/T1 proxy sweep only to propose candidate configs for a later byte-closed archive dispatch. | Kaggle proxy | READY-PROXY | Claim `kaggle_pr101_proxy_sweep` before `kaggle kernels push`; output remains `score_claim=false` and `ready_for_exact_eval_dispatch=false`. |
| P4 | Reactivate A1/PR101 archive-in-loop training only as a short, rate-capped exact-eval diagnostic, not a repeat of the regressed configuration. | `[contest-CUDA]` | BLOCKED | Must cap archive-byte growth, emit `archive_builds_manifest.json`, select best-proxy/final archive deliberately, and refuse checkpoint-only non-smoke runs. |
| P5 | PR106 score-aware sidechannel stack / sub-0.17 path. | `[contest-CUDA]` | GATED | Do not dispatch stacked/meta lanes until sister sidechannels land exact-CUDA positives; old briefing one-liners are gated, not ready. |
| P6 | T9 cross-archive composition. | none | DEFERRED | Operator directive says defer unless rescoped to single-axis A1 branching. No cross-archive kitchen-sink dispatch. |

## Concrete commands

P0 recover active T1 guard:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
  --label t1_balle_modal_guard_a3311268_20260510T0831Z
```

Before any new remote/GPU/eval dispatch:

```bash
.venv/bin/python tools/claim_lane_dispatch.py summary
.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
.venv/bin/python -m tac.preflight --scope dev --timeout-s 30
```

Kaggle proxy launch sequence, if the operator chooses proxy search:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --dry-run \
  --lane-id kaggle_pr101_proxy_sweep \
  --platform kaggle \
  --instance-job-id kaggle:adpena/pr101-proxy-sweep \
  --agent codex:kaggle_proxy_readiness \
  --status active_proxy_dispatch \
  --notes 'Kaggle PR101 proxy sweep only; score_claim=false; exact CUDA promotion required'

.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id kaggle_pr101_proxy_sweep \
  --platform kaggle \
  --instance-job-id kaggle:adpena/pr101-proxy-sweep \
  --agent codex:kaggle_proxy_readiness \
  --status active_proxy_dispatch \
  --notes 'Kaggle PR101 proxy sweep only; score_claim=false; exact CUDA promotion required'

uv run --with kaggle kaggle kernels push \
  -p experiments/kaggle_kernels/pr101_proxy_sweep
```

T1 full rerun command template only after P0 closes terminally and a fresh
active claim is made:

```bash
modal run experiments/modal_t1_balle_endtoend.py --execute \
  --label <fresh-t1-label> \
  --epochs <guarded-epoch-count> \
  --batch-size <guarded-batch-size> \
  --max-target-pairs <guarded-pair-count> \
  --timeout-hours 24 \
  --train-timeout-hours <hours-leaving-auth-eval-buffer>
```

Remote command inside a claimed provider worker:

```bash
T1_ALLOW_SCORE_DOMAIN_TRAINING=1 \
T1_RUN_CONTEST_CUDA_AUTH_EVAL=1 \
LOCAL_CUDA_WORKER=1 \
T1_DISPATCH_INSTANCE_JOB_ID=<active-claim-job-id> \
T1_DISPATCH_CLAIMS_PATH=<remote-active-claim-ledger-path> \
EPOCHS=<guarded-epoch-count> \
BATCH_SIZE=<guarded-batch-size> \
SEGMENTATION_SURROGATE=sinkhorn \
GRAD_CLIP_NORM=1.0 \
bash scripts/remote_lane_t1_balle_endtoend.sh
```

## Stale or unsafe claims to ignore

- Any queue row saying `active_count=0` is stale after the current live summary:
  T1 is active now.
- Any queue row saying active A1 Modal must be harvested is stale after the
  recovered A1 regression ledger. The A1 job is terminal and regressed.
- `.omx/state/dispatch_queue.md` lists historical Lane G-era lanes with
  absolute scores around `0.85`-`1.20`; they are not current sub-0.17/sub-0.15
  score-lowering actions without fresh archive custody and exact-CUDA gates.
- Operator briefing exact-eval one-liners that depend on Lightning environment
  variables are blocked while Lightning identity/credit/env is not ready.
- `current_score_lowering_roadmap_20260510_codex.md` is being patched by the
  operator; treat it as an input, not the write target for this update.

## Gating criteria for promotion or retirement

Promote a candidate only if all are true:

1. Exact `[contest-CUDA]` auth-eval JSON exists with `n_samples=600`.
2. `tac.auth_eval_schema` blockers are zero.
3. Archive bytes, archive SHA-256, runtime tree SHA, and adjudication packet
   fields match the evaluated artifact.
4. Score is recomputed from components, not copied from proxy logs.
5. The dispatch claim has a terminal row closing the active claim.
6. CPU reproduction policy is planned or recorded before public/submission
   claims.

Retire only a measured configuration, not a family, unless multiple exact
anchors plus adversarial review prove the family cannot satisfy the target.

## 2026-05-10T09:35Z supersession

The T1 guard is no longer active. Recover closed it terminally with
`failed_t1_modal_recovered_no_score_claim`.

Failure class: `remote_script_failed / missing_canonical_a1_payload`.

The remote trainer failed before training because
`/workspace/pact/experiments/results/A1_canonical` did not exist inside the
Modal worker. No archive, score, or model result exists. The next T1 action is
not a rerun; it is an actuator/export-custody fix that materializes or mounts
the canonical A1 payload before dispatch and records payload SHA custody in
Modal metadata.

Updated immediate priorities:

1. Fix red-team overclaim traps in raw auth-eval, auth-eval schema, stale claim
   handling, T1 recovery status, and unsupported GPU-tier cost estimation.
2. Fix T1 canonical-A1 payload mounting/designation before re-claiming T1.
3. Keep proxy optimizers (`Kaggle`, `Optuna`, `CMA-ES`, `MPS` curves) bounded
   to candidate generation until a byte-closed archive/runtime packet exists.
