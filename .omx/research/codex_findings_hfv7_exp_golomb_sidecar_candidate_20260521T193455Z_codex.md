# Codex Findings: HFV7 Exp-Golomb Sidecar Candidate

- timestamp_utc: 2026-05-21T19:34:55Z
- lane: hfv7_exp_golomb_pr101_hfv1_sidecar
- status: LANDED_EXP_GOLOMB_BYTE_CLOSED_CANDIDATE_WITH_IN_PROCESS_FRAME_PARITY
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What landed

New builder:

- `tools/build_hfv7_exp_golomb_sidecar_candidate.py`

Updated proof tool:

- `tools/prove_hfv2_sparse_inflate_parity.py`

Lane registry:

- `hfv7_exp_golomb_pr101_hfv1_sidecar` registered at L0.

HFV7 reduces the HFV6 implicit-delta foveation trailer from 16 bytes to 12
bytes by encoding the archive-contained active-pair delta sequence with
unsigned Exp-Golomb order 3:

```text
payload bytes   12
payload bits    96
deltas          [64, 15, 47, 36, 1, 130, 209, 5, 1, 7, 3, 8, 5, 6, 8, 1]
```

The runtime decoder accepts the implicit HFV7 trailer only when its length is
exactly 12 bytes, then decodes the Exp-Golomb stream and applies the same
camera-geometry active-row profile used by HFV5/HFV6. The active pair set is
still archive-contained through the 12-byte delta stream; the profile row and
implicit-trailer interpretation remain research-only until contest-compliance
review accepts them.

Adversarial-review fix applied after the first HFV7 artifact: the builder now
drops copied FEC6 public-submission artifacts (`README.md`, `report.txt`,
`pre_submission_compliance.*`) from the source runtime and regenerates
research-only HFV7 metadata. The current artifact below is the regenerated
package, not the earlier stale-doc package.

## Candidate artifact

- Output directory: `experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z`
- Archive: `experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/archive.zip`
- Submission runtime: `experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/submission_dir_hfv7_exp_golomb`
- Manifest: `experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/hfv7_exp_golomb_manifest.json`
- Paired dispatch plan: `experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/paired_dispatch_plan.json`

Hashes:

```text
eaead36921bfccaafa23b8315af97ac2a7b9526a64787f1a1067d477fe064c14  archive.zip
cd211f67bdeb1de49058e5e19b1e488a822944f56e2848a727dda064a3e51b3a  hfv7_exp_golomb_manifest.json
c124e835c67c78311408d59e7f3d32257ae389d347c613d2627f1e613e3055c9  submission_dir_hfv7_exp_golomb/archive_manifest.json
07baa3cf4b8409b2710b212437bcdcbc56b511852807d75717ae7bd15da7e940  submission_dir_hfv7_exp_golomb/inflate.py
2d785e6e66fb731a9c2622154dae1fbf8db7bcc85ab4b329978b786eb1bc2b58  submission_dir_hfv7_exp_golomb/README.md
0e9bbde71a5eef12d326ca836b98bcfd8c19894e406961205d7fb709a4001595  submission_dir_hfv7_exp_golomb/report.txt
5e3003ac727c1d1d106bab3945306aef625e4758dca5b3bcecd2ce73bf459519  paired_dispatch_plan.json
c2436e48d0955831f515f6f4a1c708d1c8e9a4f391043f6a5e8c0c7b97417793  embedded HFV7 payload
```

## Byte result

```text
dense HFV1 archive bytes        202649
HFV6 implicit archive bytes     178533
HFV7 Exp-Golomb archive bytes   178529
FEC6/PR110 baseline bytes       178517
```

HFV7 rate deltas:

```text
bytes saved vs dense HFV1       24120
bytes saved vs HFV6 implicit    4
bytes over FEC6/PR110 baseline  12
rate delta vs FEC6/PR110        0.00000799030743747
```

The required component gain to beat FEC6/PR110 on CPU-axis rate arithmetic is
now about `7.99e-06`, down from `1.07e-05` for HFV6, `1.60e-05` for HFV5,
`2.86e-05` for HFV4, and `5.99e-05` for HFV3.

## ZIP anatomy

```text
archive bytes                   178529
members                         1
member name                     x
member compression              stored
member compressed bytes         178429
member uncompressed bytes       178429
central directory bytes         47
extra fields                    none
```

## Full in-process frame parity proof

- Output directory: `experiments/results/hfv7_exp_golomb_inflate_parity_20260521T193756Z`
- JSON: `experiments/results/hfv7_exp_golomb_inflate_parity_20260521T193756Z/sparse_foveation_inflate_parity.json`
- Markdown: `experiments/results/hfv7_exp_golomb_inflate_parity_20260521T193756Z/sparse_foveation_inflate_parity.md`

Hashes:

```text
f89e62935c0b84ffdb06b450f462c266b2c3dd87e7ff4e1fbbcce0cbeb14c7b2  sparse_foveation_inflate_parity.json
4422cba5240ddadb2c7a5a904fd6f3a5045e7d7ec9fce2bdca77fad9f6d32b14  sparse_foveation_inflate_parity.md
```

Proof result:

```text
pairs checked              600 / 600
frames checked             1200
dense_output_sha256        6d27ea355e0e1dbc66fbb9f1df4c31d28faa52e1afae58231ce0185511ae81d2
sparse_output_sha256       6d27ea355e0e1dbc66fbb9f1df4c31d28faa52e1afae58231ce0185511ae81d2
output_sha256_match        true
tensor_equal               true
max_abs_diff               0.0
mismatched_batches         0
sparse_sidecar_name        embedded_foveation_params.hfv7
sparse_inflate_py_sha256   07baa3cf4b8409b2710b212437bcdcbc56b511852807d75717ae7bd15da7e940
```

## Dispatch plan

Plan-only paired Modal dispatch was materialized and did not execute Modal.

Plan result:

- archive SHA matched expected
- no existing promotable CPU anchor was found
- no existing promotable CUDA anchor was found
- pair group: `pair_hfv7_exp_golomb_pr101_hfv1_sidecar_exact_eval_eaead36921bf`
- CPU lane id: `hfv7_exp_golomb_pr101_hfv1_sidecar_exact_eval_contest_cpu`
- CUDA lane id: `hfv7_exp_golomb_pr101_hfv1_sidecar_exact_eval_contest_cuda`
- CPU/CUDA execute commands are recorded in `paired_dispatch_plan.json`

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/build_hfv7_exp_golomb_sidecar_candidate.py \
  tools/prove_hfv2_sparse_inflate_parity.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv7_exp_golomb_sidecar_candidate.py \
  --output-dir experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_hfv2_sparse_inflate_parity.py \
  --sparse-archive experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/archive.zip \
  --sparse-submission-dir experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/submission_dir_hfv7_exp_golomb \
  --batch-pairs 8 \
  --output-dir experiments/results/hfv7_exp_golomb_inflate_parity_20260521T193756Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/build_hfv7_exp_golomb_sidecar_candidate.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/prove_hfv2_sparse_inflate_parity.py --status reviewed
```

Review tracker result:

- `tools/build_hfv7_exp_golomb_sidecar_candidate.py`: 13 entities reviewed
- `tools/prove_hfv2_sparse_inflate_parity.py`: 17 entities reviewed

## Current blocker

I did not execute paired Modal exact eval because `tools/claim_lane_dispatch.py
summary` still reports 13 active dispatch claims, including DP1 paired CPU/CUDA
auth-eval calls and NSCS06/Selfcomp Modal jobs. HFV7 is byte-closed,
in-process frame-parity proven, and plan-ready as a research candidate, but it
has the same profile-code compliance gate as HFV5/HFV6 plus an implicit 12-byte
trailer interpretation gate before any submission/promotion claim. Shell-level
`inflate.sh` output parity is still intentionally listed as a promotion blocker
unless paired exact eval lands first.
