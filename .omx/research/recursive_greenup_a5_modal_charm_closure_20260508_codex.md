# Recursive Greenup Closure - 2026-05-08

Scope: local hardening and exact-negative preservation after Phase A packet
work. No subagents were spawned in this pass.

## Durable changes

- A5 frame-conditional runtime packets now rewrite the packet-local PR101
  `inflate.sh` invocation from bare `python` to
  `"${PYTHON:-python3}"`, preserving contest-simple defaults while allowing
  local custody evals to bind the repo virtualenv explicitly.
- The Modal A1 score-gradient worker now extracts the PR101 source ZIP via
  `tac.submission_archive.safe_extract_zip` instead of raw
  `ZipFile.extractall`.
- The ChARM 50k toy eval roundtrip now uses `tac.quantization.Uint8STE`
  instead of a bare `.round()` in the differentiable uint8 simulation path.
- `check_no_auth_eval_optout_help_text_consumer_unverified` now skips the
  public OSS-export mirror with the canonical `_is_oss_export_mirror_path`
  helper, avoiding self-flagging exported copies of training scripts.

## A5 advisory negative

The byte-closed A5 runtime packet remains `172615` bytes with archive SHA-256
`cde5a1e0ad49ec56856a8b1f0e4d7c329193955211d0f9fe314366d992c4c737`.

Local macOS CPU advisory eval command:

```bash
PYTHON=.venv/bin/python .venv/bin/python -u experiments/contest_auth_eval.py \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/packet/archive.zip \
  --inflate-sh experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/packet/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/macos_cpu_advisory_work \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

Result: score `1.937884415209767`, pose `0.07864571`, seg `0.00936123`,
rate contribution `0.11493725`. Evidence grade is `macOS-CPU advisory`, not
`contest-CPU`, and it is not promotable.

Disposition: the measured `eta=4.0` complexity-proxy A5 q-bit schedule is a
measured-configuration negative. The A5 family is not killed. Reactivation
requires a score-domain, Jacobian, or paired-component allocation schedule that
is locally advisory-evaled before any remote exact CUDA/CPU spend.

## Verification

```bash
.venv/bin/python -m py_compile \
  experiments/modal_phase_a1_score_gradient_pr101.py \
  experiments/train_charm_50k_toy_substrate.py \
  tools/build_pr101_frame_conditional_runtime_packet.py \
  tools/build_a5_per_pair_score_marginal_manifest.py \
  scripts/pre_submission_compliance_check.py
```

Result: pass.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_pr101_frame_conditional_runtime_packet.py \
  src/tac/tests/test_modal_phase_a1_score_gradient_pr101.py \
  src/tac/tests/test_a5_per_pair_score_marginal_manifest.py \
  src/tac/tests/test_pre_submission_compliance_check.py -q
```

Result: `33 passed`.

```bash
.venv/bin/python tools/all_lanes_preflight.py --jobs 4 --timings
```

Result: all `28` preflight checks passed. The only warning was the expected
unstaged local custody snapshot
`experiments/results/_modal_harvest_summary.json`, which remains uncommitted
raw/generated state.
