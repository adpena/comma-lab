# A5 SegNet Boundary/Margin Scalar Negative - 2026-05-09

## Verdict

Measured advisory result: post-hoc scalar SegNet boundary and low-margin
`q7/q8` schedules do **not** beat the existing scalar SegNet-protected A5 point.

Best of this pair:

- low-margin `q7/q8 low0p85`: `0.20112208325925268` macOS CPU advisory
- boundary `q7/q8 low0p85`: `0.20115808226406484` macOS CPU advisory
- prior scalar SegNet-protected `q7/q8 low0p85`: `0.20111041630821824`

Classification: measured-config regressions. This retires global per-pair
post-hoc A5 q-bit ranking by scalar SegNet boundary/margin statistics. It does
not kill A5. Reactivation requires local boundary placement inside a pair,
channel/local-region allocation, or training-time q-bit noise so the renderer
can move error away from SegNet-visible structure.

## New Tooling

`tools/build_segnet_boundary_marginals.py` computes frozen SegNet GT-frame
features with no scorer modification and no dispatch:

- `per_pair_boundary_mass`
- `per_pair_low_margin_mass`
- `per_pair_mean_logit_margin`
- `per_pair_p10_logit_margin`

The full PR101 manifest is:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `reports/a5_segnet_boundary_marginals_pr101_20260509.json` | `60,615` | `6263b909ad31de56fb26a56e8bb51fd288885edce0a2974ed6703eb20ce06c2c` |

The q-bit scheduler now accepts boundary/margin manifests directly. If the
manifest has no source `per_pair_q_bits`, it records
`synthetic_all_base_q_bits_missing_source_vector` instead of failing or hiding
state.

## Candidates

| Candidate | Archive bytes | Archive SHA-256 | macOS CPU advisory | PoseNet | SegNet |
|---|---:|---|---:|---:|---:|
| boundary `q7/q8 low0p85` | `178,243` | `7163499abb7475ee937338e9c79f22d3c326b53c34f7dbaa2b8ed1c872d67190` | `0.20115808226406484` | `0.00003518` | `0.00063717` |
| low-margin `q7/q8 low0p85` | `178,243` | `5b1f6e190f9051357d4c24b2e87fb4cf5bc6e2dc88b397d2f3c16972e7476536` | `0.20112208325925268` | `0.00003515` | `0.00063689` |

Both candidates use `q7` for the lowest `510 / 600` pairs by the selected
scalar and `q8` for the top `90 / 600`.

## Artifact Custody

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `reports/a5_score_marginal_trust_region_q7_low0p85_boundary_20260509.json` | `8,107` | `46ca4302ebf37c9457b24745c060eabfba904a5cb805fe24b0b4447bcf4ded35` |
| `reports/a5_score_marginal_trust_region_q7_low0p85_low_margin_20260509.json` | `8,119` | `157be1a935ddc7b458b47bc03c758a1496879ddc11d41fc50d2d34865d477855` |
| `experiments/results/pr101_a5_q7_boundary_runtime_packet_20260509_codex/candidate_archive_manifest.json` | `30,830` | `405e74e74a6d0e326391efac7fd3cdd79d422e83606ba45d6dc80789bee54c95` |
| `experiments/results/pr101_a5_q7_boundary_runtime_packet_20260509_codex/runtime_consumption_proof.json` | `4,571` | `9ce2c51886dd31a4d80ee0183b5c4f11f623dc4f01c069d01dc2b1698ee528c3` |
| `experiments/results/pr101_a5_q7_boundary_runtime_packet_20260509_codex/pre_submission_compliance.no_auth.json` | `11,114` | `9e2e419b2281843f6aa106187fa3cbe1f62fa180262c0f2e3a3ce794a0717b4e` |
| `experiments/results/pr101_a5_q7_boundary_runtime_packet_20260509_codex/readiness.with_boundary_packet.json` | `10,004` | `0b4483319e14d6f81d7ffcf64fd6a9633ded48bff59ee6191648af08d89463ff` |
| `experiments/results/pr101_a5_q7_boundary_runtime_packet_20260509_codex/contest_auth_eval.macos_cpu_advisory.json` | `7,454` | `8154f7f1c7bb3aea3e85fa03be63bce839f2eca556d3eba66d23a82a0077dbae` |
| `experiments/results/pr101_a5_q7_low_margin_runtime_packet_20260509_codex/candidate_archive_manifest.json` | `30,854` | `8951f121d556e652a516a0868928354a29620ea074b377650cacfb2cdf5aef74` |
| `experiments/results/pr101_a5_q7_low_margin_runtime_packet_20260509_codex/runtime_consumption_proof.json` | `4,577` | `dcba7518b8e569b3a2d2a2b5ec330083bd2c198a7bc14c735f0af86ca965a6f0` |
| `experiments/results/pr101_a5_q7_low_margin_runtime_packet_20260509_codex/pre_submission_compliance.no_auth.json` | `11,160` | `91e194dddc586d723b3a9e66503b41468442442bdf94fe09e645821b851bb23b` |
| `experiments/results/pr101_a5_q7_low_margin_runtime_packet_20260509_codex/readiness.with_low_margin_packet.json` | `10,042` | `aafc738cb229424a55d40f039cd9cf110b9cc5b17ee3b1865216965205caddef` |
| `experiments/results/pr101_a5_q7_low_margin_runtime_packet_20260509_codex/contest_auth_eval.macos_cpu_advisory.json` | `7,471` | `5576c526601bf2da1f600d0378043f1e5a0564698f53c5ffcb6754a3edf3ea30` |

## Commands

```bash
.venv/bin/python tools/build_segnet_boundary_marginals.py \
  --json-out reports/a5_segnet_boundary_marginals_pr101_20260509.json \
  --device cpu \
  --batch-size 16 \
  --candidate-id a5_segnet_boundary_marginals_pr101_20260509_codex

.venv/bin/python tools/build_a5_score_marginal_qbits_schedule.py \
  --score-marginal-manifest reports/a5_segnet_boundary_marginals_pr101_20260509.json \
  --json-out reports/a5_score_marginal_trust_region_q7_low0p85_boundary_20260509.json \
  --candidate-id a5_trust_q7_low0p85_boundary_20260509_codex \
  --base-q-bits 8 \
  --low-q-bits 7 \
  --low-fraction 0.85 \
  --marginal-source boundary

.venv/bin/python tools/build_a5_score_marginal_qbits_schedule.py \
  --score-marginal-manifest reports/a5_segnet_boundary_marginals_pr101_20260509.json \
  --json-out reports/a5_score_marginal_trust_region_q7_low0p85_low_margin_20260509.json \
  --candidate-id a5_trust_q7_low0p85_low_margin_20260509_codex \
  --base-q-bits 8 \
  --low-q-bits 7 \
  --low-fraction 0.85 \
  --marginal-source low_margin
```

Packet builds used `tools/build_pr101_frame_conditional_runtime_packet.py`
with `--recompute-wire-contract-for-q-bits`, then
`scripts/pre_submission_compliance_check.py --strict`,
`tools/build_pr101_frame_conditional_packet_readiness.py`, and finally local
macOS CPU advisory auth eval through:

```bash
PYTHON=.venv/bin/python .venv/bin/python -u experiments/contest_auth_eval.py \
  --archive <packet>/archive.zip \
  --inflate-sh <packet>/inflate.sh \
  --upstream-dir upstream \
  --device cpu \
  --work-dir <candidate>/macos_cpu_advisory_work \
  --json-out <candidate>/contest_auth_eval.macos_cpu_advisory.json \
  --inflate-timeout 1800 \
  --evaluate-timeout 5400 \
  --keep-work-dir
```

Dispatch claims were opened before each advisory eval and closed as:

- `a5_trust_q7_low0p85_boundary_macos_cpu_advisory`:
  `completed_macos_cpu_advisory_negative`
- `a5_trust_q7_low0p85_low_margin_macos_cpu_advisory`:
  `completed_macos_cpu_advisory_negative`

## Interpretation

The ordering among tested scalar `q7/q8 low0p85` schedules is:

1. SegNet proxy scalar: `0.20111041630821824`
2. low-margin scalar: `0.20112208325925268`
3. boundary-mass scalar: `0.20115808226406484`

All three preserve much more SegNet quality than q7-all, but all remain about
`0.0083` worse than the A1 Linux x86_64 contest-CPU anchor while saving only
`19 B`. The problem is not lack of global scalar ranking signal; it is that a
single q-bit per pair is too coarse to steer distortion away from the exact
pixels/classes SegNet cares about.

## Reactivation Criteria

- Implement local boundary-aware allocation inside each pair or latent channel,
  not a single scalar q-bit per pair.
- Train or fine-tune with q-bit noise in the loop so the renderer can change
  representation around SegNet-visible regions.
- Reopen exact-eval spend only if local advisory reaches within `0.001` of the
  A1/PR101 CPU anchor or if a byte reduction is large enough to pay the
  measured SegNet distortion.
