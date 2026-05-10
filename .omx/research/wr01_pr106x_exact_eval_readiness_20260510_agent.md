# WR01 PR106x exact-eval readiness audit - local print-only

generated_at_utc: 2026-05-10T15:07:42Z
repo_head: f3d12cea75567dd05fec15d0b5b3e41d053e55ae
scope: local custody audit and print-only command rendering only
research_only: true
remote_gpu_run: false
dispatch_attempted: false
lane_claim_opened: false
score_claim: false

## Reviewed surfaces

- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/archive.zip`
- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/archive_manifest.json`
- `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/inflate.sh`
- `tools/lightning_dispatch_pr106_stack.py`

## Archive custody

Candidate archive:

- bytes: `186222`
- sha256: `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628`
- ZIP shape: one central-directory entry, member `x`, stored/no compression, no extra field, no comment, DOS timestamp `1980-01-01 00:00:00`
- member payload bytes: `186122`
- member payload sha256: `803a0940f92ec1cb1b70e9815fb6c666650b5371625e17103104baa21e96b4e7`

Source archive recorded by the manifest:

- path: `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip`
- bytes: `186231`
- sha256: `d25bca80057e8b533197895b4c56370678feb4e05fea0312c405bd12f29bec8e`
- member payload sha256: `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`

Changed section:

- section: `latents_and_sidecar_brotli`
- source section bytes: `15849`
- candidate section bytes: `15840`
- byte delta: `-9`
- source section sha256: `94257b33cf3083c5daa0f3b1e127cb7c51bee42a6416b19763eea7bf9ecc3c32`
- candidate section sha256: `2d85c4891e06906833e6a106dd486184121159a2380b581dc0e0b7d56603e939`
- source raw sha256: `a38778c6304bacba39705cd9c45af337d73bc90c6b7b4ccf2563febfc312328e`
- candidate raw sha256: `5fc6d95f9f769c9dad7b575a0ab81198be19fa5339733a50d1f212d76a4dc413`
- raw changed positions: `64`
- raw absolute byte delta sum: `3610`
- local brotli decode status: `ok`, not score evidence

Unchanged sections:

- `packed_header_ff_len24`: `4` -> `4` bytes, sha256 unchanged `7939f08db7d18dd4176e8e11b1232a9be6b2371db7a4e63c1c0871fb520148b6`
- `decoder_packed_brotli`: `170278` -> `170278` bytes, sha256 unchanged `654999f81f0552fb7568e6977e73aa329661c10c79a6ab6cddc3171302352004`

## Runtime wrapper custody

Release wrapper:

- path: `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/inflate.sh`
- sha256: `6052c2ddcb42f2149e9eb364b73735e06748cf1600ab3fd7a793805fcee87709`
- executable: yes
- behavior: resolves repo root and delegates to the public PR106 belt-and-suspenders intake wrapper.

Delegated runtime:

- path: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/inflate.sh`
- sha256: `0cb9c3df78ad870cadd2ac6d46007b6c1aeb6a61c76b7800ae74a00addafb885`

## Ready locally

- The release archive bytes and SHA match `archive_manifest.json`.
- The candidate member payload SHA matches the release manifest.
- The source member payload SHA matches the payload-diff manifest.
- The changed-section boundary is explicit and limited to `latents_and_sidecar_brotli`.
- `wr01_exact_eval_packet.json` records `static_packet_ready=true`, `candidate_static_preflight_ready=true`, `byte_custody_exact_eval_candidate_ready=true`, `runtime_decode_gate_ready=true`, `compliance_ok=true`, and `artifact_consistency_ok=true`.
- Existing static compliance refresh in `wr01_exact_eval_packet.json` has return code `0` and no failed compliance checks.
- `tools/lightning_dispatch_pr106_stack.py --print-only` now renders successfully with repo-relative `--inflate-sh` paths after the local patch below.

## Blocks exact CUDA dispatch

- No exact CUDA auth eval exists for this WR01 candidate.
- No `contest_auth_eval.adjudicated.json` exists for this WR01 candidate.
- No active WR01 lane dispatch claim is open; this was intentionally not opened per operator instruction for this audit.
- The current local Lightning identity environment renders empty `--teamspace`, `--studio`, and `--user` values in the resolved launcher command. Actual dispatch requires those to be configured or supplied through the dispatch surface before removing print-only mode.
- `wr01_exact_eval_packet.json` records `dispatch_unlocked=false` with blockers `missing_lightning_environment`, `missing_active_lane_dispatch_claim`, and `adversarial_priority_review_prioritizes_rate_only_candidate`.
- The adversarial priority review defers WR01 behind the byte-closed rate-only candidate `pr106_q10_151byte_brotli` (`186088` bytes, sha256 `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`) because WR01 changes decoded latent/sidecar bytes for only a 9-byte rate win.
- Local runtime decode validation is custody evidence only; it is not SegNet/PoseNet score evidence.
- The print-only `--source-manifest` path is prospective. It is created by staging in a real dispatch path; this audit did not stage.

## Exact print-only command

This command was run locally and exited `0` after the readiness patch. It prints the resolved `scripts/launch_lightning_batch_job.py exact-eval` invocation but does not stage, claim, or launch:

```bash
.venv/bin/python tools/lightning_dispatch_pr106_stack.py \
  --lane wr01_apply_pr106x_half \
  --archive experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/archive.zip \
  --inflate-sh experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/release_surface/inflate.sh \
  --predicted-low 0.18 \
  --predicted-high 0.25 \
  --job-name exact_eval_wr01_apply_pr106x_half_20260510_print_only \
  --print-only
```

Rendered command caveat: in the current environment, the resolved launcher command includes empty values for `--teamspace`, `--studio`, and `--user`; that is a dispatch blocker, not a score/custody finding.

## Local bug fixed

Bug: a repo-relative `--inflate-sh` passed to `tools/lightning_dispatch_pr106_stack.py --print-only` validated as an existing file, then crashed while rendering because `submit_dispatch()` called `inflate_sh.relative_to(REPO_ROOT)` on a relative `Path`.

Fix: normalize supplied repo-relative `--inflate-sh` paths against `REPO_ROOT`, then fail closed if the wrapper is outside the checkout.

Changed files:

- `tools/lightning_dispatch_pr106_stack.py`
- `src/tac/tests/test_lightning_dispatch_pr106_stack.py`

Post-fix helper SHA:

- `tools/lightning_dispatch_pr106_stack.py`: `b89aede425a8dfe787bbf7912ab88c08316bc5d2f5128baa9d06a55b0b701d7b`

## Verification

- `shasum -a 256 release_surface/archive.zip release_surface/archive_manifest.json release_surface/inflate.sh`
- `zipinfo -v release_surface/archive.zip`
- `unzip -p release_surface/archive.zip x | shasum -a 256`
- `unzip -p experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/archive.zip x | shasum -a 256`
- `jq` summaries of `archive_manifest.json`, `payload_section_diff_vs_pr106x.json`, `hnerv_wavelet_runtime_decode_validation.json`, and `wr01_exact_eval_packet.json`
- `.venv/bin/python -m pytest src/tac/tests/test_lightning_dispatch_pr106_stack.py -q` -> `14 passed in 0.13s`
- print-only command above -> exit `0`

## Solver-stack wire-in disposition

This is a custody/readiness ledger, not a new empirical anchor or deployable lane landing. `research_only=true`.

- sensitivity-map contribution: N/A, no score or component delta measured.
- Pareto constraint: N/A, no exact CUDA result; existing packet already records the rate-only priority blocker.
- bit-allocator hook: N/A, no allocator policy changed.
- Cathedral autopilot dispatch hook: N/A, print-only only and no claim opened.
- continual-learning posterior update: N/A, no empirical anchor.
- probe-disambiguator: N/A, no new 2-mode implementation choice introduced.
