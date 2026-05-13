# HDM4 Paired CPU Closure And Entropy-Floor Probe - 2026-05-13

## Summary

HDM4 now has a recovered paired `[contest-CPU]` Modal Linux x86_64 closure
against the same archive SHA as the existing `[contest-CUDA]` T4 closure. This
is an axis-specific result, not CPU/CUDA conversion authority.

The same turn hardened `tools/pr106_entropy_floor_probe.py` so the planning
probe can analyze the current HDM4 frontier packet directly:

- PR106 sidecar wrapper `0xfe` is unwrapped with outer custody preserved.
- HDM3/HDM4 decoder sections are decoded through `tac.hnerv_decoder_recode`
  instead of assuming plain Brotli.
- The probe remains planning-only: `score_claim=false`,
  `ready_for_exact_eval_dispatch=false`.

## Paired CPU Closure

- archive:
  `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/archive.zip`
- archive SHA-256:
  `218ae16f3f13b722e9752d698667ed8770151e40d44b5756c0ebbccb7682825f`
- archive bytes: `186492`
- CPU artifact:
  `experiments/results/modal_auth_eval_cpu/pr106_r2_lowlevel_hdm4_cpu_20260513T095252Z_codex/contest_auth_eval.json`
- CPU Modal call id: `fc-01KRGC2RQ6TPA4A5926Q2BJ5JM`
- CPU lane id: `hnerv_hdm4_q_brotli_split_cpu_closure`
- CPU job id: `modal_pr106_r2_hdm4_cpu_20260513T095252Z`
- CPU evidence grade: `contest-CPU`
- CPU score axis: `contest_cpu`
- CPU score: `0.22787475059700513`
- CPU components: `avg_segnet_dist=0.00063198`,
  `avg_posenet_dist=0.00016402`
- CPU samples: `600`
- CPU compliance:
  `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/pre_submission_compliance.cpu_exact_nonfinal.json`
- CPU compliance result: `passed=true`, `48` checks, `0` errors, `0` warnings
- dispatch claims after recovery: `active=0`, `stale_nonterminal=0`

Existing paired CUDA artifact:

- CUDA artifact:
  `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/contest_auth_eval.json`
- CUDA evidence grade: `contest-CUDA`
- CUDA score axis: `contest_cuda`
- CUDA score: `0.20642625334307507`
- CUDA components: `avg_segnet_dist=0.0006426`,
  `avg_posenet_dist=0.00003236`
- CUDA samples: `600`

## CPU/CUDA Mechanism Boundary

Paired diagnostic:
`experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/cpu_cuda_axis_pair_analysis.json`

Result classification:

- `valid_individual_axis_scores=true`
- `valid_same_archive_axis_score_pair=true`
- `valid_for_pair_score_analysis=false`
- `valid_for_mechanism_analysis=false`
- blocker: `cpu_cuda_runtime_tree_sha256_mismatch`
- raw-output pairing: `different_inflated_outputs`
- mechanism class: `different_raw_outputs_runtime_or_inflate_drift`

This means the CPU and CUDA scores are both valid on their own axes for the
same archive, but the gap must not be interpreted as a universal CPU-vs-CUDA
ordering or as scorer-only drift. The raw inflated outputs differ, so
inflate/runtime/device behavior is part of the mechanism until localized.

## Entropy-Floor Probe

Artifacts:

- JSON:
  `experiments/results/pr106_r2_hdm4_entropy_floor_probe_20260513_codex/entropy_floor.json`
- Markdown:
  `experiments/results/pr106_r2_hdm4_entropy_floor_probe_20260513_codex/entropy_floor.md`

HDM4 source summary from the probe:

- archive bytes: `186492`
- payload magic after unwrap: `ff_packed_hnerv`
- outer payload magic: `pr106_sidecar_wrapper`
- decoder section codec: `hdm4_q_brotli_split`
- decoder section bytes: `169990`
- latents section bytes: `15849`
- sidecar format id: `2`
- sidecar payload bytes: `527`
- framing meta bytes: `6`

Planning-only oracle rows:

| group | current bytes | best transform | best Markov2 floor | delta |
|---|---:|---|---:|---:|
| `decoder_q_zz_plus_f32_scales` | `169990` | `delta_mod` | `45660` | `-124330` |
| `fixed_latents_delta_zz_plus_fp16_meta` | `15849` | `delta_mod` | `849` | `-15000` |
| `decoded_payload_sections_without_ff_header` | `185839` | `delta_mod` | `46508` | `-139331` |

Interpretation: this is a model-class lower bound only. It does not charge the
runtime decoder, model tables, transform metadata, or exact-eval packet
overhead. The useful HDM5 direction is a deterministic charged bitstream for a
delta-coded decoder/latent stream that beats the current `169990` decoder
section including all recipe/header/length/scales/runtime overhead.

## Commands

CPU dispatch used the canonical Modal CPU wrapper with explicit release-surface
runtime upload:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
  experiments/modal_auth_eval_cpu.py \
  --archive experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/archive.zip \
  --submission-dir experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface \
  --inflate-sh inflate.sh \
  --output-dir experiments/results/modal_auth_eval_cpu/pr106_r2_lowlevel_hdm4_cpu_20260513T095252Z_codex \
  --detach --provider-detach-ack \
  --lane-id hnerv_hdm4_q_brotli_split_cpu_closure \
  --instance-job-id modal_pr106_r2_hdm4_cpu_20260513T095252Z \
  --claim-agent codex:modal_auth_eval_cpu \
  --claim-notes "HDM4 paired contest-CPU closure using exact release surface archive/runtime; axis=contest_cpu; no CUDA promotion conversion"
```

Recovery:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir /Users/adpena/Projects/pact/experiments/results/modal_auth_eval_cpu/pr106_r2_lowlevel_hdm4_cpu_20260513T095252Z_codex
```

Entropy probe:

```bash
.venv/bin/python tools/pr106_entropy_floor_probe.py \
  --archive experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/archive.zip \
  --json-out experiments/results/pr106_r2_hdm4_entropy_floor_probe_20260513_codex/entropy_floor.json \
  --md-out experiments/results/pr106_r2_hdm4_entropy_floor_probe_20260513_codex/entropy_floor.md \
  --pr101-reference-archive-bytes 177551 \
  --active-floor-archive-bytes 186492 \
  --active-floor-label PR106-R2-HDM4-exact-CUDA
```

Validation:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  src/tac/tests/test_pr106_entropy_floor_probe.py

.venv/bin/python -m ruff check \
  tools/pr106_entropy_floor_probe.py \
  src/tac/tests/test_pr106_entropy_floor_probe.py
```

## Next Work

1. HDM5: implement only after deterministic search finds `<169990` charged
   decoder-section bytes including recipe/header/length/scales overhead.
2. Extend runtime adapter and submission runtime only for a proven HDM5 row;
   do not add runtime branches for losing probes.
3. Localize HDM4 CPU/CUDA raw-output drift with existing xray tools before
   making mechanism claims.
4. Continue PR95/HNeRV parity training work separately; HDM4 is rate-only byte
   work, not representation improvement.
