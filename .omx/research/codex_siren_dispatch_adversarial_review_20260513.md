# Codex adversarial review: SIREN dispatch and A1 CUDA-axis path (2026-05-13)

## Scope

Operator requested a CRITICAL/DEFECT/MINOR/CLEAN review of the SIREN Modal A100
dispatch chain and the A1 CUDA-axis path, with Tier 1 files reviewed first:

- `experiments/train_substrate_siren.py`
- `tools/run_modal_smoke_before_full.py`
- `experiments/modal_train_lane.py`
- `experiments/contest_auth_eval.py`
- `src/tac/deploy/modal/mount_manifest.py`
- `src/tac/substrates/siren/`
- `src/tac/substrates/_shared/trainer_skeleton.py`
- `src/tac/substrates/score_aware_common.py`
- `.omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml`
- `scripts/remote_lane_substrate_siren.sh`

Follow-on checks covered immediate next dispatch candidates:
`sane_hnerv`, `balle_renderer`, `cool_chic`, `vq_vae`, plus the recovered
`a1_plus_lapose` L0 scaffold from partner WIP.

## Findings and fixes

### CRITICAL fixed: substrate runtimes emitted PNG frames, not contest raw streams

The SIREN substrate and immediate sibling substrate runtimes were scaffolded as
per-frame PNG emitters. `experiments/contest_auth_eval.py` validates one
`.raw` tensor stream per video name, with camera-frame byte size matching the
contest evaluator. A PNG-emitting runtime can train and build bytes but cannot
produce an exact auth-eval score.

Fixed by adding `tac.substrates._shared.inflate_runtime` and migrating:

- `src/tac/substrates/siren/inflate.py`
- `src/tac/substrates/sane_hnerv/inflate.py`
- `src/tac/substrates/balle_renderer/inflate.py`
- `src/tac/substrates/cool_chic/inflate.py`
- `src/tac/substrates/vq_vae/inflate.py`

The shared helper owns:

- contest camera size `(874, 1164)`;
- deterministic raw path mapping from `file_list` entries;
- CUDA-or-CPU inflate-device selection with no MPS authority;
- RGB pair resize/quantize/write logic for `.raw` output.

### DEFECT fixed: generated runtimes did not all vendor their substrate package

`sane_hnerv`, `balle_renderer`, and `vq_vae` generated runtime wrappers that
imported `tac.substrates.<name>.inflate` without shipping the substrate package
inside the contest runtime. Exact eval would import-fail after archive build.

Fixed by centralizing package vendoring in
`tac.substrates._shared.trainer_skeleton.vendor_shared_inflate_runtime()` and
making all five immediate substrate trainers vendor their runtime package plus
the shared raw-output helper.

### DEFECT fixed: SIREN trainer could return rc=0 without contest-CUDA evidence

`experiments/train_substrate_siren.py` parsed finite auth-eval scores, but a
diagnostic-only result could still leave the trainer with rc=0 and no posterior
update. That is not usable dispatch evidence.

Fixed by requiring a component-coherent `[contest-CUDA]` score claim when
`--skip-auth-eval` is false. Diagnostic-only CUDA/CPU/MPS auth-eval outputs now
raise before a "successful" trainer exit.

### DEFECT fixed: smoke-before-full only effectively trusted one auth artifact

`tools/run_modal_smoke_before_full.py` now scans every returned auth-eval JSON
artifact deterministically and greens only when at least one artifact is a
finite, component-coherent `contest_cuda` score claim inside the plausible
band. Diagnostic artifacts no longer mask a valid artifact, and no diagnostic
artifact can green the full canary.

### DEFECT fixed: SIREN readiness audit encoded the obsolete PNG contract

`src/tac/substrates/siren_readiness.py` now requires raw-output semantics and
explicitly checks that the PNG writer is absent. The audit gate passed after the
contract update.

### DEFECT fixed: CUDA evidence contract now requires Linux x86_64

`experiments/contest_auth_eval.py` now requires Linux x86_64 for
`[contest-CUDA]` exact evidence in addition to full sample count and
T4/A100/4090/H100/A10G/L40S CUDA hardware. Non-Linux CUDA is diagnostic, not a
contest-axis claim.

### CLEAN after review

- SIREN trainer uses EMA default `0.997`.
- SIREN score-aware training path calls the canonical `score_pair_components`
  through the shared scorer-preprocess contract.
- SIREN train/validation paths keep `eval_roundtrip=True` as the live path.
- Modal launcher already records HEAD/source sentinel evidence and scans
  `workspace/experiments/results`.
- Modal mount-manifest logic is fail-closed on mtime/source stability.
- `contest_auth_eval.py` already keeps durable workdirs when `--json-out` is
  provided and `--allow-temp-work-dir` is not set.

## A1 CUDA-axis review note

No A1 score claim was changed in this pass. The correct next A1 action remains
apples-to-apples CUDA/CPU validation of byte-closed packets and score-domain
validation that changes the runtime-consumed packet. macOS CPU or proxy signals
remain non-authoritative.

The partner A1+LAPose WIP was preserved as an L0 scaffold with archive tests.
It is not dispatch-ready: no trainer, recipe, remote driver, or exact-eval
packet path landed in this pass.

## Verification

Commands run:

```text
.venv/bin/python -m py_compile experiments/contest_auth_eval.py experiments/train_substrate_balle_renderer.py experiments/train_substrate_cool_chic.py experiments/train_substrate_sane_hnerv.py experiments/train_substrate_siren.py experiments/train_substrate_vq_vae.py tools/run_modal_smoke_before_full.py src/tac/substrates/_shared/trainer_skeleton.py src/tac/substrates/_shared/inflate_runtime.py src/tac/substrates/balle_renderer/inflate.py src/tac/substrates/cool_chic/inflate.py src/tac/substrates/sane_hnerv/inflate.py src/tac/substrates/siren/inflate.py src/tac/substrates/vq_vae/inflate.py src/tac/substrates/a1_plus_lapose/*.py
git diff --check
.venv/bin/python -m pytest -q src/tac/tests/test_run_modal_smoke_before_full.py src/tac/tests/test_siren_substrate_readiness.py src/tac/tests/test_contest_auth_eval.py src/tac/substrates/siren/tests/test_siren_roundtrip.py src/tac/substrates/a1_plus_lapose/tests/test_a1_plus_lapose_archive.py src/tac/substrates/sane_hnerv/tests/test_train_substrate_sane_hnerv_full_main.py src/tac/substrates/balle_renderer/tests/test_train_substrate_balle_renderer_full_main.py src/tac/substrates/cool_chic/tests src/tac/substrates/vq_vae/tests -q
.venv/bin/python tools/audit_siren_substrate_readiness.py --json --fail-if-not-ready
/usr/bin/time -p .venv/bin/python -m tac.preflight
```

Results:

- focused pytest suite: passed;
- SIREN readiness: passed;
- preflight: passed in `real 7.84s`, under the 30s budget;
- no remote/GPU dispatch was launched by this review.

## Next score-lowering steps

1. Do not launch a full SIREN dispatch unless a lane claim is active and the
   smoke-before-full wrapper sees a real `[contest-CUDA]` score claim.
2. Treat pure SIREN as infrastructure/empirical signal unless it can plausibly
   enter the sub-0.19 band; the higher-EV SIREN work is PR106 or A1 residual
   composition with score-aware selection.
3. Continue A1+LAPose only after landing a real trainer, recipe, remote driver,
   runtime vendoring, both-axis exact-eval plan, and no-op/full-frame parity
   tests.
4. Keep all CUDA/CPU comparisons axis-labelled; there is no universal CPU-better
   or CUDA-better rule.
