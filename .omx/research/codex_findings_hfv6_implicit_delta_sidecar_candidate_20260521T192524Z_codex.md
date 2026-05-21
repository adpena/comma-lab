# Codex Findings: HFV6 Implicit-Delta Sidecar Candidate

- timestamp_utc: 2026-05-21T19:25:24Z
- lane: hfv6_implicit_delta_pr101_hfv1_sidecar
- status: LANDED_IMPLICIT_DELTA_BYTE_CLOSED_CANDIDATE_WITH_FULL_PARITY
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What landed

New builder:

- `tools/build_hfv6_implicit_delta_sidecar_candidate.py`

Updated proof tool:

- `tools/prove_hfv2_sparse_inflate_parity.py`

HFV6 reduces the HFV5 profile-coded foveation trailer from 24 bytes to 16 bytes
by storing only the active-pair deltas as raw uint8 trailing bytes after the
PR101/FEC6 selector payload:

```text
payload bytes   16
deltas          [64, 15, 47, 36, 1, 130, 209, 5, 1, 7, 3, 8, 5, 6, 8, 1]
```

The pair set is not recoverable from the selector payload alone: the FEC6
selector has 466 non-identity pairs, while the foveation sidecar applies to 16
pairs. The 16 pair deltas therefore remain archive-contained. The active row
still uses the HFV5 camera-geometry profile in runtime code, so HFV6 is
research-only until implicit-trailer/profile-code compliance is accepted.

## Candidate artifact

- Output directory: `experiments/results/hfv6_implicit_delta_sidecar_candidate_20260521T192355Z`
- Archive: `experiments/results/hfv6_implicit_delta_sidecar_candidate_20260521T192355Z/archive.zip`
- Submission runtime: `experiments/results/hfv6_implicit_delta_sidecar_candidate_20260521T192355Z/submission_dir_hfv6_implicit_delta`
- Manifest: `experiments/results/hfv6_implicit_delta_sidecar_candidate_20260521T192355Z/hfv6_implicit_delta_manifest.json`
- Paired dispatch plan: `experiments/results/hfv6_implicit_delta_sidecar_candidate_20260521T192355Z/paired_dispatch_plan.json`

Hashes:

```text
4ef8b9424ff32bd7404f2d58e555946ee48d255d622e8e8f40d86f674d6ac8dd  archive.zip
13daa00fafede39ac64bd452767e0d1190fb43711add7582dc0710ced6ad9233  hfv6_implicit_delta_manifest.json
963486b5ec1540601f5ef8eef6edf04eb82ad1388521dcdc62ab2ec1c3e1a2b1  submission_dir_hfv6_implicit_delta/archive_manifest.json
e61b0b891eb3fd15c2cc0125608849ea81b5d05e36c9c1aab5bb9c6f2dbbb40f  submission_dir_hfv6_implicit_delta/inflate.py
db4fc85535d78950e130fd574e2325a133edce2a69ab135827439a6204e60c0b  paired_dispatch_plan.json
```

## Byte result

```text
dense HFV1 archive bytes        202649
HFV5 profile archive bytes      178541
HFV6 implicit archive bytes     178533
FEC6/PR110 baseline bytes       178517
```

HFV6 rate deltas:

```text
bytes saved vs dense HFV1       24116
bytes saved vs HFV5 profile     8
bytes over FEC6/PR110 baseline  16
rate delta vs FEC6/PR110        0.00001065374325
```

The required component gain to beat FEC6/PR110 on CPU-axis rate arithmetic is
now about `1.07e-05`, down from `1.60e-05` for HFV5, `2.86e-05` for HFV4, and
`5.99e-05` for HFV3.

## ZIP anatomy

```text
archive bytes                   178533
members                         1
member name                     x
member compression              stored
member compressed bytes         178433
member uncompressed bytes       178433
central directory bytes         47
extra fields                    none
```

## Full parity proof

- Output directory: `experiments/results/hfv6_implicit_delta_inflate_parity_20260521T192417Z`
- JSON: `experiments/results/hfv6_implicit_delta_inflate_parity_20260521T192417Z/sparse_foveation_inflate_parity.json`
- Markdown: `experiments/results/hfv6_implicit_delta_inflate_parity_20260521T192417Z/sparse_foveation_inflate_parity.md`

Hashes:

```text
e769d4638cfe74e15232d9b0b2129353822aad040bd533c0d91799213a72afc4  sparse_foveation_inflate_parity.json
de54f05fbd4cfd15f92cea9d024fd645ba5296c67a6c25086f60d8fead32ae08  sparse_foveation_inflate_parity.md
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
sparse_sidecar_name        embedded_foveation_params.hfv6
```

## Dispatch plan

Plan-only paired Modal dispatch was materialized and did not execute Modal.

Plan result:

- archive SHA matched expected
- no existing promotable CPU anchor was found
- no existing promotable CUDA anchor was found
- pair group: `pair_hfv6_implicit_delta_pr101_hfv1_sidecar_exact_eval_4ef8b9424ff3`
- CPU/CUDA execute commands are recorded in `paired_dispatch_plan.json`

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/build_hfv6_implicit_delta_sidecar_candidate.py \
  tools/prove_hfv2_sparse_inflate_parity.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv6_implicit_delta_sidecar_candidate.py \
  --output-dir experiments/results/hfv6_implicit_delta_sidecar_candidate_20260521T192355Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_hfv2_sparse_inflate_parity.py \
  --sparse-archive experiments/results/hfv6_implicit_delta_sidecar_candidate_20260521T192355Z/archive.zip \
  --sparse-submission-dir experiments/results/hfv6_implicit_delta_sidecar_candidate_20260521T192355Z/submission_dir_hfv6_implicit_delta \
  --batch-pairs 8 \
  --output-dir experiments/results/hfv6_implicit_delta_inflate_parity_20260521T192417Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/build_hfv6_implicit_delta_sidecar_candidate.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/prove_hfv2_sparse_inflate_parity.py --status reviewed
```

Review tracker result:

- `tools/build_hfv6_implicit_delta_sidecar_candidate.py`: 10 entities reviewed
- `tools/prove_hfv2_sparse_inflate_parity.py`: 17 entities reviewed

## Current blocker

I did not execute paired Modal exact eval because `tools/claim_lane_dispatch.py
summary` still reports 13 active dispatch claims, including DP1 paired CPU/CUDA
auth-eval calls and NSCS06/Selfcomp Modal jobs. HFV6 is byte-closed,
full-parity proven, and plan-ready as a research candidate, but it has the same
profile-code compliance gate as HFV5 plus a stricter implicit-trailer parsing
gate before any submission/promotion claim.
