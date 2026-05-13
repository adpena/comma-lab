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

---

## Supersession: SIREN/A1+LAPose dispatch-chain hardening after BUILD-RESUME

Date: 2026-05-13 later pass.

The earlier note that A1+LAPose had no trainer/recipe/remote driver is now
superseded by the local BUILD-RESUME commits through `d4adab79` plus this
hardening pass. The A1+LAPose path has a trainer, recipe, remote driver,
operator wrapper, archive grammar, and focused tests, but it still needs a
claimed smoke/full exact-eval anchor before any score claim.

### CRITICAL fixed: Modal recipe paths could overwrite writable workspace paths

`experiments/modal_train_lane.py` copied the repo to writable `/tmp/pact`, then
blindly applied recipe `env_overrides`. Recipes such as SIREN and A1+LAPose use
`/workspace/pact/...` paths for operator readability; on Modal that mount is
read-only. The blind override could send output/checkpoint/result writes back to
the read-only mount and fail after GPU startup.

Fixed by mapping env override values rooted at `/workspace/pact` to the Modal
writable workspace before launching the lane script. This keeps recipes
provider-readable while preserving the runtime write contract.

### CRITICAL fixed: Modal authorizer could split lane custody

`tools/operator_authorize.py::_dispatch_modal` did not pass the recipe
`lane_id` to `experiments/modal_train_lane.py`. The direct Modal claim could
therefore use an inferred lane such as `remote_lane_substrate_siren` while the
operator-approved claim used `lane_substrate_siren_20260512`.

Fixed by threading `--lane-id <recipe.lane_id>` into the Modal launcher. Catalog
#191 now treats `--lane-id` as a strict required Modal-dispatch surface alongside
`--sentinel-files` and `--require-clean-head`.

### CRITICAL fixed: SIREN remote driver did not verify active claim identity

`scripts/remote_lane_substrate_siren.sh` checked that a claims ledger existed
but did not verify that an active row matched both `LANE_ID` and
`DISPATCH_INSTANCE_JOB_ID`. A split or stale claim could reach GPU work.

Fixed by generating a JSON claim summary at stage 0 and refusing before NVDEC
probe/training unless the active claim matches both lane and job id.

### DEFECT fixed: smoke-before-full could block its own full dispatch

`tools/run_modal_smoke_before_full.py` launched the smoke with the same recipe
lane id as the full dispatch, but did not append a terminal claim row before
starting the full phase. The full claim could conflict with the green smoke row.

Fixed by parsing the smoke `instance_job_id` from `operator_authorize.py` output
and appending a terminal row on timeout, smoke-red, smoke-only-green, and
smoke-green-before-full paths.

### DEFECT fixed: A1+LAPose default pose tilt was non-arbitrary-hostile

The A1+LAPose trainer and loss defaulted `pose_weight_scale=2.71`, a
PR106/A1 operating-point hypothesis. Preflight correctly rejected that as a
hidden default divergence from the contest formula. The default is now `1.0`;
the 2.71 pose tilt remains an explicit experimental CLI choice.

### DEFECT fixed outside git-tracked research state: Catalog #125 memory format

The A1+LAPose companion memory declared the probe-disambiguator hook in bolded
Markdown form that Check #125 did not count. The memory line now contains the
literal `probe_disambiguator:` declaration, clearing strict preflight.

### Partner signal preserved

The D4 deeper council memo
`.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_d4_deeper_20260513.md`
is preserved as durable research signal. It binds A1+LAPose D4 to the as-built
D4.B single-stage trailer format and records D4.A/D4.D reactivation criteria.

### Verification

Commands run:

```text
.venv/bin/python -m py_compile experiments/modal_train_lane.py tools/operator_authorize.py tools/run_modal_smoke_before_full.py experiments/train_substrate_a1_plus_lapose.py src/tac/substrates/a1_plus_lapose/score_aware_loss.py
.venv/bin/python -m pytest -q src/tac/tests/test_modal_train_lane_hardening.py src/tac/tests/test_run_modal_smoke_before_full.py src/tac/tests/test_operator_authorize_canonical_tool.py src/tac/tests/test_check_191_modal_dispatch_threads_sentinel_files.py src/tac/tests/test_siren_substrate_readiness.py src/tac/substrates/a1_plus_lapose/tests/test_a1_plus_lapose_archive.py src/tac/substrates/a1_plus_lapose/tests/test_a1_plus_lapose_trainer_and_loss.py
.venv/bin/python tools/audit_siren_substrate_readiness.py --json --fail-if-not-ready
/usr/bin/time -p .venv/bin/python -m tac.preflight
```

Results:

- focused pytest suite: 57 passed;
- SIREN readiness audit: passed; local contract ready, score claim false,
  promotion eligible false, ready for exact eval false until dispatch;
- strict preflight: passed in `real 9.16s`, under the 30s budget;
- no remote/GPU dispatch was launched by this review.
