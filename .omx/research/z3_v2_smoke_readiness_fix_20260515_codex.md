# Z3 v2 Smoke Readiness Fix - 2026-05-15

research_only=true
lane_id=lane_z3_balle_v2_latent_replacement_build_20260514
owned_scope=TIME-TRAVELER/non-NeRV Z3 frontier execution

## Change

The Z3 smoke trainer now honors `--enable-v2-latent-replacement` when the
provided `--a1-archive-path` contains an A1-shaped inner payload. The smoke
artifact emits the operational `z3v2_latent_replacement` contract instead of
the retired append-only Z3HP1 diagnostic path.

Legacy smoke behavior remains available as a fallback only for non-A1-shaped
isolated fixtures. Encoder/build-contract `ValueError`s are not swallowed by
the fallback.

The full trainer's result-review block now distinguishes v2 from v1: v2
artifacts no longer carry the stale "append-only grammar cannot realize
predicted byte savings" blocker.

## Evidence

Focused tests:

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_substrate.py \
  src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_v2_substrate.py -q
```

Result: `72 passed, 1 warning`.

Local real-A1 CPU smoke, no scorer and no paid dispatch:

```bash
.venv/bin/python experiments/train_substrate_z3_balle_hyperprior_bolton.py \
  --a1-archive-path submissions/a1/archive.zip \
  --output-dir /tmp/z3_v2_smoke_readiness_20260515_codex \
  --epochs 1 --device cpu --smoke --enable-v2-latent-replacement
```

Result:

- `layout=z3v2_latent_replacement`
- `archive_bytes=164026`
- `archive_sha256=b4d1f8e3e38d0d098798aa6f7775f03098319af32be5cbffc1e274f18ebc1c7f`
- `archive_zip_bytes=164134`
- `archive_zip_sha256=99d9e6cbec2058b64be6cb2f2dfe0b4d1018f54bac000587132a7d19bdc2d3d1`
- `z3v2_section_bytes=1251`
- `byte_savings_bytes=14136`
- `v2_smoke_fallback_reason=null`

Strict local pre-deploy:

```bash
.venv/bin/python tools/local_pre_deploy_check.py \
  --trainer experiments/train_substrate_z3_balle_hyperprior_bolton.py \
  --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch --strict
```

Result: `ALL 8 CHECKS PASSED. Safe to dispatch.`

Dry-run actuator, no Modal dispatch:

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch \
  --smoke-epochs 100 --smoke-gpu T4 --smoke-timeout-hours 0.5 \
  --operator-handle codex:z3_v2_smoke_readiness --smoke-only --dry-run
```

Result: wrapper resolved `smoke_validation_contract=training_artifact_v1`,
recognized `smoke_only: true`, and would dispatch only the smoke job.

## Hook Discipline

- Sensitivity map: no score anchor emitted; smoke remains `score_claim=false`.
- Pareto constraint: `archive_contract` records v2 byte savings, but promotion
  remains blocked until exact auth-eval review.
- Bit allocator: byte savings are exposed in the manifest; no allocator prior
  is mutated by smoke.
- Cathedral autopilot: dry-run command above is claim-safe and no-spend.
- Continual learning: no posterior update because there is no auth-eval score.
- Probe disambiguator: v1/v2 arbitration remains explicit through the existing
  trainer flag; the new test proves the v2 mode produces a real `Z3HV2` payload.

## Next Claim-Safe Action

Before any paid run:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_z3_balle_v2_latent_replacement_build_20260514 \
  --platform modal \
  --instance-job-id z3_v2_smoke_t4_training_artifact_v1_20260515 \
  --agent codex:z3_v2_smoke_readiness \
  --status active_smoke \
  --notes "Z3 v2 smoke-only T4 training_artifact_v1; no score claim until auth-eval review"
```

Then run the dry-run command without `--dry-run` only after operator approval
for spend.

## 2026-05-15 Codex Hardening Addendum

After adversarial production/OSS review, Z3 was tightened from "v2-capable"
to v2-only for production packets.

Executable changes:

- The trainer imports cleanly through the package API and local pre-deploy now
  includes an import-time check, so missing exports fail before Modal spend.
- Generated Z3 runtime no longer vendors v1 `archive.py` / `inflate.py` and
  refuses raw A1 or Z3HP1 fallback payloads.
- Z3HV2 production packet uses direct-residual mode and does not ship no-op
  hyperprior weights / `w_hat` bytes until a real entropy-coded residual
  decoder consumes them at inflate time.
- The Z3HV2 grammar rejects non-unit quantization and the trainer rejects
  non-default hidden width until the wire format records it.
- The emitted submission runtime has `submission_runtime_manifest.json` and a
  `runtime_tree_sha256` in stats.
- Remote driver and recipe now default to v2, with legacy v1 documented as
  forensic-only.

Focused verification:

```bash
.venv/bin/ruff check experiments/train_substrate_z3_balle_hyperprior_bolton.py \
  src/tac/substrates/z3_balle_hyperprior_bolton/archive_v2.py \
  src/tac/substrates/z3_balle_hyperprior_bolton/inflate_v2.py \
  src/tac/substrates/z3_balle_hyperprior_bolton/score_aware_loss_v2.py \
  src/tac/substrates/z3_balle_hyperprior_bolton/quant.py \
  tools/local_pre_deploy_check.py \
  src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_substrate.py \
  src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_v2_substrate.py \
  src/tac/tests/test_operator_authorize_canonical_tool.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_substrate.py \
  src/tac/substrates/z3_balle_hyperprior_bolton/tests/test_z3_v2_substrate.py \
  src/tac/tests/test_operator_authorize_canonical_tool.py

.venv/bin/python tools/local_pre_deploy_check.py \
  --trainer experiments/train_substrate_z3_balle_hyperprior_bolton.py \
  --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch --strict
```

Results:

- ruff: `All checks passed!`
- pytest: `95 passed`
- local pre-deploy: `ALL 8 CHECKS PASSED. Safe to dispatch.`

Authority note: Z3 remains non-promotional until exact paired
`[contest-CUDA]` + `[contest-CPU]` auth-eval results land on the same archive
and runtime. The direct-residual v2 packet is an executable score-lowering
candidate, not a score claim.

## 2026-05-15 Modal Smoke Result

Command:

```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
  --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch \
  --smoke-epochs 100 --smoke-gpu T4 --smoke-timeout-hours 1.0 \
  --operator-handle codex-z3v2 --smoke-only
```

Result:

- Modal call: `fc-01KRNPJM15X4NH28375SCE954M`
- Instance/job id:
  `substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T113237Z__smoke__100ep`
- Recovery path:
  `experiments/results/lane_substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T113237Z__smoke__100ep_modal/`
- Terminal recovery: `rc=0`, `timed_out=false`, elapsed `8.578769848999997s`
- Cost-band anchor: T4, 100 epochs, estimated cost `$0.0014059650585861105`
- Claim state: terminal
  `completed_modal_training_recovered_no_score_claim`; no active same-lane
  claims after recovery.

Recovered artifact signal:

- Layout: `z3v2_latent_replacement`
- Base archive payload bytes: `178162`
- Z3 v2 archive payload bytes: `163969`
- Payload byte savings: `14193`
- Archive ZIP bytes: `164077`
- Archive ZIP SHA-256:
  `5f41db95f333ce42c4a95cee6c02f6cb6f4286fbf57dbee7be17dc3c221a14ee`
- Payload SHA-256:
  `d3fc0affcefb246204e0b4bdc0867d544932ec5467c32b5d739f5fba7126f946`
- Runtime tree SHA-256:
  `671ac1f789619fa39ce9dcaa38250c2e11ac6a61abb66556f7c4940083176ba8`
- Z3HV2 section bytes: `1194`

Promotion status:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- Blockers:
  `smoke_no_scorer_load`,
  `requires_separate_auth_eval_result_review_before_score_claim`,
  `z3v2_score_claim_requires_paired_cuda_cpu_auth_eval`

Classification: real executable packet-build signal, not a score. The smoke
proves the hardened v2 direct-residual path emits a byte-smaller archive and
survives Modal T4 packaging. The next score-bearing action is paired
`[contest-CUDA]` and `[contest-CPU]` auth eval against this exact archive and
runtime, followed by the normal result-review packet.

## 2026-05-15 Full Dispatch Launched

After the smoke anchor, Codex fixed the remaining config gate that kept this
recipe marked `smoke_only: true` even though `_full_main` and the v2 runtime
path were implemented. The full dispatch recipe now sets `smoke_only: false`,
`smoke_validation_contract: contest_cuda_auth_eval_v1`, and
`SMOKE_ONLY=0`.

Verification before dispatch:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  src/tac/tests/test_operator_authorize_canonical_tool.py

.venv/bin/python tools/local_pre_deploy_check.py \
  --trainer experiments/train_substrate_z3_balle_hyperprior_bolton.py \
  --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch --strict
```

Results:

- operator-authorize tests: `18 passed`
- local pre-deploy: `ALL 8 CHECKS PASSED. Safe to dispatch.`
- worktree: clean at dispatch
- claim summary before dispatch: `active=0`

Launched full run:

```bash
SMOKE_ONLY=0 .venv/bin/python tools/operator_authorize.py \
  --recipe substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch \
  --agent codex-z3v2-full \
  --label-suffix __full__1000ep \
  --yes
```

Dispatch:

- Modal call: `fc-01KRNQ2B8ZH6ASSAHCXMH0AMZX`
- Instance/job id:
  `substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch_20260515T114115Z__full__1000ep`
- Hardware: Modal T4
- Timeout: `1.5h`
- Env contract includes:
  `SMOKE_ONLY=0`,
  `Z3_BALLE_ENABLE_V2_LATENT_REPLACEMENT=1`,
  `Z3_BALLE_EPOCHS=1000`
- Active claim at launch:
  `lane_z3_balle_hyperprior_bolton_recover_20260514`
  / `active_dispatch`

Non-blocking recovery poll immediately after launch returned still-running.
Canonical harvest command:

```bash
.venv/bin/python experiments/modal_recover_lane.py \
  --call-id fc-01KRNQ2B8ZH6ASSAHCXMH0AMZX
```

No score claim exists until that recovery lands and the returned archive/runtime
passes the exact result-review packet. If CUDA returns promising, the same
archive/runtime must be replayed on the Linux CPU axis before any frontier or
submission-ready language.
