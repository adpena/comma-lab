# T1 score-domain training wire-in (2026-05-10)

## Summary

T1 Ballé hyperprior + 128K decoder now has a real scorer-domain training path.
`--enable-scorer-domain-loss` routes decoded frames through the PR #95
eval-roundtrip primitive, differentiable YUV6 scorer loading, PoseNet, and
SegNet, then feeds tensor-valued `seg_loss` and `pose_loss` into the
JointLagrangianADMM residuals.

This removes the previous measured blocker where Phase 1's train loop used
constant `seg_target` / `pose_target` placeholders and therefore could not
determine the true potential of T1/T8/T13/T19 on weights.

## Evidence

- Shared oracle: `tac.losses.scorer_loss_terms_btchw`.
- Trainer gate: `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py --enable-scorer-domain-loss`.
- Remote actuator: `scripts/remote_lane_t1_balle_endtoend.sh` now supports
  `T1_ALLOW_SCORE_DOMAIN_TRAINING=1` for real A1 + scorer-domain training.
- Remote claim handoff: the remote script requires a real
  `tools/claim_lane_dispatch.py` summary against a copied
  `active_lane_dispatch_claims.md` ledger. Env-only/base64 claim summaries are
  refused as forgeable.
- Remote CUDA guard: the NVDEC/CUDA probe now requires
  `LOCAL_CUDA_WORKER=1`, and the dispatcher command template sets it only for
  the remote provider execution path.
- Focused tests:
  - `tests/paradigm_delta_epsilon_zeta/test_eval_roundtrip_in_training_loop.py`
  - `tests/paradigm_delta_epsilon_zeta/test_phase1_trainer_write_runtime_fix.py`
  - `src/tac/tests/test_phase1_trainer_auth_eval_contract.py`
  - `tests/test_dispatch_t1_balle_endtoend.py`
- Local closure:
  - synthetic score-domain smoke wrote `/tmp/pact_t1_score_domain_smoke/archive.zip`.
  - real A1 + `upstream/videos/0.mkv` one-pair debug wrote
    `/tmp/pact_t1_score_domain_realpair_clip/archive.zip`.
  - real A1 + `upstream/videos/0.mkv` one-pair debug with the remote-default
    `sinkhorn` surrogate wrote
    `/tmp/pact_t1_score_domain_realpair_sinkhorn_gradguard/archive.zip` and
    emitted first-batch gradient reachability proof:
    `decoder_grad_l2=6.281017256051246e12`,
    `balle_main_grad_l2=1.0208590237824937e-08`.

## Claim discipline

No score claim is made from this work. The local runs are CPU debug closure
only, not `[contest-CUDA]` or submission evidence. Promotion still requires:

- dispatch claim before remote GPU/eval;
- full scorer-domain CUDA training run;
- generated archive SHA-256 and runtime custody;
- hermetic runtime/export closure that does not depend on repo-local `tac`
  modules at inflate time;
- exact `contest_auth_eval.py` CUDA evaluation;
- formula recomputation from component fields.

## Next execution

Highest-EV next dispatch is T1 score-domain CUDA training:

```bash
T1_ALLOW_SCORE_DOMAIN_TRAINING=1 \
LOCAL_CUDA_WORKER=1 \
T1_DISPATCH_INSTANCE_JOB_ID=<active-claim-job-id> \
T1_DISPATCH_CLAIMS_PATH=<remote-active-claim-ledger-path> \
EPOCHS=3000 \
BATCH_SIZE=16 \
SEGMENTATION_SURROGATE=sinkhorn \
GRAD_CLIP_NORM=1.0 \
bash scripts/remote_lane_t1_balle_endtoend.sh
```

Run it only inside a claimed remote lane with CUDA available. The dispatcher
template sets `LOCAL_CUDA_WORKER=1` for the remote host; local operator shells
should not set it just to satisfy the probe. The remote script verifies that
`T1_DISPATCH_INSTANCE_JOB_ID` matches an active `t1_balle_128k_endtoend` claim
in the copied ledger before creating logs or starting training. MPS remains
non-authoritative and must not be used for auth eval.

## 2026-05-10 runtime-closure update

Codex closed the first hermetic-runtime blocker from the adversarial review:

- `_write_runtime()` now emits a packet-local `submission_dir/src/tac/` runtime
  subset (`decoder_128k.py` + `balle_hyperprior.py`) instead of a `model.py`
  shim that searches parent directories for repo-local `src/tac`.
- `tac.phase1_packet_compiler` no longer treats undeclared `tac` imports as
  safe unless a packet-local `src/tac/__init__.py` package exists.
- The Phase 1 grammar/export label is now the actual three-member contract:
  `Phase1-three-member-x-decoder-bin-balle-bin` /
  `phase1_three_member_x_decoder_bin_balle_bin`.
- The executable no-op proof now invokes `inflate.sh` through the contest
  three-argument signature with a non-empty `video_names` file, rather than
  calling `inflate.py` directly with an empty list.
- Score-domain mode now refuses `--epochs 0` in both smoke and non-smoke modes,
  because zero epochs skips the first-batch scorer-gradient reachability proof.

Focused verification:

```bash
.venv/bin/python -m pytest \
  tests/paradigm_delta_epsilon_zeta/test_phase1_trainer_write_runtime_fix.py \
  src/tac/tests/test_build_phase1_packet_compiler.py \
  src/tac/tests/test_phase1_trainer_auth_eval_contract.py -q
# 127 passed, 3 warnings
```

Remaining T1 blockers are now narrower: auth-eval custody, exact CUDA evidence,
state-dict wire-format rate tightening, and dispatch-claim lifecycle hardening.

## 2026-05-10 auth-eval promotion closure update

Codex reduced the auth-eval custody gap between the trainer output and an
exact CUDA evidence artifact:

- `scripts/remote_lane_t1_balle_endtoend.sh` now defaults score-domain remote
  runs to `T1_RUN_CONTEST_CUDA_AUTH_EVAL=1`.
- After training, the remote script compiles `OUTPUT_DIR/submission_dir` with
  `tools/build_phase1_packet_compiler.py --mode optimize` and the actual
  Phase 1 three-member export format.
- The compiled packet must emit `build_manifest.json`, `archive.zip`, and an
  executable `inflate.sh` before any auth eval runs.
- The auth-eval stage calls `experiments/contest_auth_eval.py` with
  `--device cuda`, durable `--work-dir`, durable `--json-out`, and
  `--expected-runtime-tree-sha256` from the packet manifest.
- The terminal claim closes as `completed_t1_contest_cuda_auth_eval` only if
  `tac.auth_eval_schema.required_contest_cuda_evidence_blockers(...)` accepts
  the exact JSON, archive byte count, full 600-sample count, CUDA device,
  contest-CUDA semantics, packet/runtime SHA custody, and no-op proof. A
  diagnostic CUDA result, CPU/MPS result, non-T4/non-equivalent hardware, packet
  compile blocker, auth-eval failure, or no-op proof failure closes as a failed
  terminal claim and does not become a score claim.
- The packet compiler CLI default now matches the API default:
  `phase1_three_member_x_decoder_bin_balle_bin`.
- The dispatcher remains non-dispatching: it writes a dry-run remote-command
  plan and exercises the lane-claim helper only with `--dry-run` until a
  provider-specific actuator can create the real job id.

Focused verification:

```bash
bash -n scripts/remote_lane_t1_balle_endtoend.sh
.venv/bin/python -m py_compile \
  tools/build_phase1_packet_compiler.py \
  tools/dispatch_t1_balle_endtoend.py
.venv/bin/python -m pytest \
  tests/test_dispatch_t1_balle_endtoend.py \
  src/tac/tests/test_build_phase1_packet_compiler.py \
  src/tac/tests/test_phase1_trainer_auth_eval_contract.py \
  tests/paradigm_delta_epsilon_zeta/test_phase1_trainer_write_runtime_fix.py -q
# 143 passed, 3 warnings
```

Dry-run claim probe:

```bash
.venv/bin/python tools/dispatch_t1_balle_endtoend.py \
  --provider modal \
  --dry-run \
  --output-dir /tmp/pact_t1_dispatch_probe.GAkLwE
# claim helper invoked with --dry-run; active claims remained 0
```

Next remote command, after creating a real active claim row for the provider
job id and copying `.omx/state/active_lane_dispatch_claims.md` to the remote:

```bash
cd /workspace/pact
test "$(git branch --show-current)" = main
git pull --ff-only origin main
T1_ALLOW_SCORE_DOMAIN_TRAINING=1 \
T1_RUN_CONTEST_CUDA_AUTH_EVAL=1 \
LOCAL_CUDA_WORKER=1 \
T1_DISPATCH_INSTANCE_JOB_ID=<active-claim-job-id> \
T1_DISPATCH_CLAIMS_PATH=<remote-active-claim-ledger-path> \
EPOCHS=3000 \
BATCH_SIZE=16 \
SEGMENTATION_SURROGATE=sinkhorn \
GRAD_CLIP_NORM=1.0 \
bash scripts/remote_lane_t1_balle_endtoend.sh
```

Current blocker: no provider job was launched in this local turn. The next GPU
spend still requires a real active `t1_balle_128k_endtoend` claim tied to the
provider job id; the local claim summary after the dry-run probe was
`active=0 stale_nonterminal=0 terminal_latest=564 unparsable_timestamp=0`.
