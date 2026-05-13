# HDM4 HLM1 CPU/CUDA Axis Closure (2026-05-13)

## Summary

PR106 HDM4 + HLM1 was evaluated on both exact axes with the same archive and
runtime-content tree. The result is a useful counterexample to any universal
"CPU is better than CUDA" rule: for this archive, `[contest-CPU]` is materially
worse than `[contest-CUDA]`, driven by the PoseNet component.

Do not convert either axis into the other. Treat this as per-archive,
per-runtime evidence only.

## Shared Archive

- archive:
  `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/pr106_r2_hdm4_exact_cuda_hlm1_latent_candidate.zip`
- bytes: `186423`
- SHA-256:
  `8801845d5099b957898fb6c6e58625bfb4cc065085ed2e3154c2cbc702dc91e0`
- runtime content tree SHA-256:
  `e05e79416feed12996efce6e23cc6d21cbc276ff6221c7a2a4eea8dd2d4d2161`
- samples: `600`

## Contest CUDA

- output directory:
  `experiments/results/modal_auth_eval/hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513`
- call id: `fc-01KRGPGJNJ0BYHKW80GZPARMW8`
- axis: `[contest-CUDA]`
- runtime tree SHA-256:
  `de844bdf4170b40e7ca2c94cc546c039b49704f6d99c730d4318803eed81aab9`
- inflated raw aggregate SHA-256:
  `5f65c70f59c78e5a4394dc062fe750cf721619f6d67790c4844d52f14d248993`
- score: `0.20638030907530963`
- PoseNet distance: `0.00003236`
- SegNet distance: `0.0006426`
- evidence grade: `contest-CUDA`
- adjudication: `A++ contest T4`, `IN_PREDICTED_BAND`

## Contest CPU

- output directory:
  `experiments/results/modal_auth_eval_cpu/hnerv_hlm1_fixed_latent_recode_modal_cpu_sourcecommit_20260513`
- call id: `fc-01KRGQEP17DK73T3BTEW4Z98K0`
- axis: `[contest-CPU]`
- platform: `Linux x86_64`
- runtime tree SHA-256:
  `9f96af4e62c60f5c2e4efc9c151dcc2c0e68adb5e8a3db94c49ca135d707b576`
- inflated raw aggregate SHA-256:
  `08675dc4d129c8a1848de9c81dc158a80d6927061a60a8e8ac15befc5ecea107`
- source commit from `PACT_SOURCE_COMMIT`:
  `38fe553ba9dd42981f31aad391ad23a9cebba7e2`
- score: `0.22782680632923968`
- PoseNet distance: `0.00016402`
- SegNet distance: `0.00063196`
- evidence grade: `contest-CPU`
- score claim: `false`
- promotion eligible: `false`

Manual recompute:

```text
100 * 0.00063196
+ sqrt(10 * 0.00016402)
+ 25 * 186423 / 37545489
= 0.22782680632923968
```

## Delta

- CPU minus CUDA score: `+0.021446497253930052`
- raw aggregate differs by axis:
  `08675dc4d129c8a1848de9c81dc158a80d6927061a60a8e8ac15befc5ecea107`
  vs
  `5f65c70f59c78e5a4394dc062fe750cf721619f6d67790c4844d52f14d248993`
- CPU SegNet is slightly better: `0.00063196` vs `0.0006426`
- CPU PoseNet is much worse: `0.00016402` vs `0.00003236`

This reinforces the current apples-to-apples rule: CPU/CUDA behavior is
submission- and runtime-specific. It is not safe to infer direction from other
HNeRV public submissions.

Next xray step: diff CPU/CUDA inflated tensors frame-by-frame and component-map
the PoseNet drift. The shared archive and shared runtime content tree rule out a
payload-byte difference; the remaining mechanism is device/runtime numerical
behavior inside inflate or scorer preprocessing.

## Exact-Pair Mechanism Xray

Canonical diagnostic artifact:
`.omx/research/artifacts/hdm4_hlm1_cpu_cuda_exact_pair_20260513_codex/analysis.json`

- schema: `cpu_cuda_exact_pair_mechanism_analysis.v1`
- evidence grade: `paired_exact_auth_eval_mechanism_diagnostic`
- valid individual axis scores: `true`
- valid same-archive axis score pair: `true`
- valid mechanism analysis: `true`
- same archive SHA-256: `true`
- same archive bytes: `true`
- same runtime content tree SHA-256: `true`
- same inflated raw aggregate SHA-256: `false`
- raw-output pairing status: `different_inflated_outputs`
- mechanism class: `different_raw_outputs_runtime_or_inflate_drift`
- score claim: `false`
- promotion eligible: `false`
- rank/kill eligible: `false`

This proves the HLM1 CPU/CUDA gap is not a payload-byte gap and not a runtime
content gap. It localizes the mechanism to device/runtime behavior that changes
the inflated raw output before the scorer consumes it. The next mechanism pass
should compare the two 3.66 GB raw tensors in streaming chunks, then project the
largest drift regions through PoseNet/SegNet component traces.

## Cleanup Note

An earlier CPU closure attempt at
`experiments/results/modal_auth_eval_cpu/hnerv_hlm1_fixed_latent_recode_modal_cpu_enforced_20260513`
produced the same CPU score, archive hash, and runtime tree, but its provenance
recorded `pact_commit` via failed `git rev-parse` because it was spawned before
`experiments/modal_auth_eval_cpu.py` passed `PACT_SOURCE_COMMIT`. The
sourcecommit rerun above supersedes it for durable claims.

## Frontier Context And Release Compliance Refresh

Operator reminder on 2026-05-13: the public frontier cluster is about `0.19`,
so HLM1 at `0.20638030907530963 [contest-CUDA]` is not enough. It is the current
local exact `[contest-CUDA]` floor, not the target floor.

Axis facts remain non-convertible:

- HLM1 `[contest-CUDA]`: `0.20638030907530963`, archive `186423` bytes,
  PoseNet `0.00003236`, SegNet `0.0006426`.
- HLM1 `[contest-CPU]`: `0.22782680632923968`, same archive `186423` bytes,
  PoseNet `0.00016402`, SegNet `0.00063196`.
- CPU is worse here by `+0.021446497253930052`, even though some public HNeRV
  submissions score better on CPU. The mechanism is per archive/runtime and
  must not be generalized.

The static release-surface `report.txt` was refreshed to include exact archive
SHA, bytes, runtime tree, and `[contest-CUDA]` component values. The strict
pre-submission compliance check now passes:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py --contest-final --strict \
  --submission-dir experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/static_release_surface \
  --archive experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/pr106_r2_hdm4_exact_cuda_hlm1_latent_candidate.zip \
  --archive-manifest-json experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/static_release_surface/archive_manifest.json \
  --auth-eval-json experiments/results/modal_auth_eval/hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513/contest_auth_eval.json \
  --require-auth-eval --require-t4-equivalent --require-submission-runtime-match \
  --expect-single-member 0.bin \
  --expected-archive-sha256 8801845d5099b957898fb6c6e58625bfb4cc065085ed2e3154c2cbc702dc91e0 \
  --expected-archive-size-bytes 186423 \
  --expected-runtime-tree-sha256 de844bdf4170b40e7ca2c94cc546c039b49704f6d99c730d4318803eed81aab9 \
  --dispatch-claims-md .omx/state/active_lane_dispatch_claims.md \
  --expected-lane-id hnerv_hlm1_fixed_latent_recode_exact_eval \
  --expected-job-id exact_eval_hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513 \
  --json-out experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/pre_submission_compliance.contest_final.strict_20260513_codex.json
```

Result: `passed=true`, failed checks `[]`.

This compliance closure is custody hardening only. It does not lower score.
Next score-lowering work must target representation/substrate distortion:
PR95/HNeRV parity training, SIREN/Ballé/CompressAI first anchors, and
structured latent/decoder PacketIR transforms that change distortion or large
sections, not additional tiny rate-only sidecar recodes.
