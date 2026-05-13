# SIREN sub-0.19 first-anchor blockers (2026-05-13)

generated_at_utc: `2026-05-13T13:49:05Z`  
agent: `codex`  
lane_id: `lane_substrate_siren_20260512`  
scope: local hardening only; no GPU spend; no remote dispatch  
score_claim: `false`  
promotion_eligible: `false`  
ready_for_remote_dispatch: `false`  
ready_for_exact_eval_dispatch: `false`

## Operator constraint

Do not spend more effort on lanes whose expected score remains `>0.19`. For
SIREN, only concrete blockers to a plausible `<0.19` first-anchor/training run
are in scope.

## Current readiness verdict

The current SIREN substrate is locally wired and smokeable, but the current
pure coordinate-MLP scaffold is not yet a credible `<0.19` dispatch target.

Reasons:

- No full SIREN renderer archive has an empirical first anchor. Current local
  evidence is readiness/static checks plus CPU smoke, not score evidence.
- The default full-substrate rate proxy is `83,846` trainable params,
  `167,692` fp16 bytes, rate term `0.11165921956696316`; `<0.19` would leave
  only about `0.07834` score for SegNet+PoseNet distortion before ZIP/runtime
  overhead. There is no evidence the scalar-`t` SIREN scaffold can hit that
  distortion budget.
- The only SIREN exact CUDA evidence in the registry is the prior PR106
  sidecar empty-residual run, not a score-moving SIREN substrate anchor.

Conclusion: do not dispatch the current pure SIREN substrate as-is. The next
useful implementation is score-aware residual selection over PR106 decoded
outputs, not another readiness polish pass.

## Concrete next implementation needed

Implement a scorer-aligned SIREN/FFT residual selector in
`tools/materialize_siren_residual_pr106_sidecar.py`:

1. Consume matched PR106 decoded raw frames and contest GT raw frames with
   runtime-tree SHA custody, as already required by `--residual-mode l2_encoded`.
2. Replace L2-energy coefficient ranking with a contest-score-aware ranking:
   use cached/canonical PoseNet+SegNet component sensitivity or a reviewed
   Hinton-distilled scorer surrogate to estimate per-coefficient
   `delta_score_per_byte`.
3. Emit only candidates whose manifest records old/new archive SHA-256,
   residual payload SHA/bytes, selected coefficient list, expected rate cost,
   proxy evidence grade, `score_claim=false`, and
   `ready_for_exact_eval_dispatch=false`.
4. Prove runtime consumption and no-op boundaries locally before any operator
   can authorize a GPU canary: byte mutation changes `0.bin`, inflate applies
   the residual payload, and payload absence is byte-identical to PR106 base.
5. Dispatch only if the local manifest makes a plausible `<0.19` case, meaning
   the predicted score delta clears the residual byte cost and the measured
   PR106 baseline gap by enough margin to justify exact CUDA.

## Local hardening landed in this pass

- `SirenScoreAwareLoss` now defaults to shared contest constants from
  `tac.substrates.score_aware_common`.
- SIREN trainer checkpoints/provenance now mark validation Lagrangian as
  `training_proxy_non_authoritative`, with proxy score/promotion flags false.
- The readiness audit now checks the shared-constant and proxy-non-authority
  contracts.

## Verification

Commands run:

```bash
.venv/bin/python tools/audit_siren_substrate_readiness.py --json --fail-if-not-ready
```

Result: passed. Key fields:
`local_contract_ready=true`, `ready_for_first_anchor_training=true`,
`ready_for_remote_dispatch=false`, `ready_for_exact_eval_dispatch=false`,
`score_claim=false`, `promotion_eligible=false`, manifest hash
`sha256:22bc5ed5557fb0226514834d7d3454b612d07a72e2ca91bbff00b493704a809a`.

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/siren/tests \
  src/tac/tests/test_siren_substrate_readiness.py -q
```

Result: `21 passed in 3.14s`.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_materialize_residual_pr106_sidecars.py \
  src/tac/tests/test_residual_basis_nonhnerv_scaffolds.py \
  src/tac/tests/test_residual_basis_pr106_sidecar_packing.py \
  src/tac/tests/test_residual_basis_pr106_materializer_helpers.py -q
```

Result: `151 passed in 24.39s`.

```bash
.venv/bin/python experiments/train_substrate_siren.py \
  --video-path upstream/videos/0.mkv \
  --output-dir .omx/tmp/siren_smoke_harden_20260513_codex \
  --epochs 3 \
  --device cpu \
  --smoke \
  --skip-archive-build \
  --skip-auth-eval
```

Result: CPU smoke only; no score claim. Losses:
`1.0080`, `1.0039`, `0.9999`.

```bash
.venv/bin/python -m py_compile \
  experiments/train_substrate_siren.py \
  tools/audit_siren_substrate_readiness.py \
  tools/materialize_siren_residual_pr106_sidecar.py \
  src/tac/substrates/siren/archive.py \
  src/tac/substrates/siren/architecture.py \
  src/tac/substrates/siren/score_aware_loss.py \
  src/tac/substrates/siren/inflate.py \
  src/tac/substrates/siren_readiness.py
```

Result: passed.

```bash
.venv/bin/python -m ruff check \
  experiments/train_substrate_siren.py \
  tools/audit_siren_substrate_readiness.py \
  tools/materialize_siren_residual_pr106_sidecar.py \
  src/tac/substrates/siren/score_aware_loss.py \
  src/tac/substrates/siren/tests/test_score_aware_loss_real_scorer_forward.py \
  src/tac/tests/test_siren_substrate_readiness.py \
  src/tac/substrates/siren_readiness.py
```

Result: passed.

```bash
git diff --check -- \
  experiments/train_substrate_siren.py \
  src/tac/substrates/siren/score_aware_loss.py \
  src/tac/substrates/siren/tests/test_score_aware_loss_real_scorer_forward.py \
  src/tac/substrates/siren_readiness.py \
  src/tac/tests/test_siren_substrate_readiness.py
```

Result: passed.

## Remaining blockers

1. `score_aware_residual_selector_missing`: no implementation yet ranks SIREN
   residual atoms by canonical scorer benefit per byte.
2. `matched_raw_custody_missing`: this pass did not materialize PR106 decoded
   raw frames and GT raw frames with runtime-tree SHA custody.
3. `sub019_case_missing`: no local manifest yet shows a plausible `<0.19`
   candidate before exact CUDA spend.
4. `no_first_anchor`: no full SIREN substrate archive has exact CUDA or
   contest-CPU evidence.
