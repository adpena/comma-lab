# Codex Findings: HFV5 Profile-Coded Sidecar Candidate

- timestamp_utc: 2026-05-21T19:19:13Z
- lane: hfv5_profile_pr101_hfv1_sidecar
- status: LANDED_PROFILE_CODED_BYTE_CLOSED_CANDIDATE_WITH_FULL_PARITY
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What landed

New builder:

- `tools/build_hfv5_profile_sidecar_candidate.py`

Updated proof tool:

- `tools/prove_hfv2_sparse_inflate_parity.py`

HFV5 reduces the HFV4 embedded foveation trailer from 43 bytes to 24 bytes by
storing only:

```text
magic          4 bytes  HFV5
pair_count     1 byte   16
profile_id     1 byte   1
pair_deltas   18 bytes  delta-uvarint active-pair list
```

The active foveation row is derived from camera geometry in the runtime profile:

```text
alpha       5.5e-4
radius      sqrt(1164^2 + 874^2) * 0.78
power       1.4
cx          (1164 - 1) / 2
cy          874 * 0.45
```

This is byte-closed with respect to the selected profile id and pair list, but
it moves the active-row formula into runtime code. Treat HFV5 as research-only
until profile-code contest-compliance review accepts that interpretation.

## Candidate artifact

- Output directory: `experiments/results/hfv5_profile_sidecar_candidate_20260521T191742Z`
- Archive: `experiments/results/hfv5_profile_sidecar_candidate_20260521T191742Z/archive.zip`
- Submission runtime: `experiments/results/hfv5_profile_sidecar_candidate_20260521T191742Z/submission_dir_hfv5_profile`
- Manifest: `experiments/results/hfv5_profile_sidecar_candidate_20260521T191742Z/hfv5_profile_manifest.json`
- Paired dispatch plan: `experiments/results/hfv5_profile_sidecar_candidate_20260521T191742Z/paired_dispatch_plan.json`

Hashes:

```text
6d783ef691d150873c928e4377d7ddabdf2c9032b0d654e88bd87a3f64491e36  archive.zip
562bacc4ea67abdf0ca2692f4925fc390a6cbbec69f3f41d0ab07cc7e973ec06  hfv5_profile_manifest.json
aeecf2c6b35d87ef1b2923870f08174e7655e477c6f39e306d0ed6e573de212f  submission_dir_hfv5_profile/archive_manifest.json
b38f5d59ab1c5ddd412b793553cce9d80f6227ad4acdbe33e84e95332ce29a38  submission_dir_hfv5_profile/inflate.py
c26f4f22cc786935a2de86c9ff6f14ff3502913efb66f606c9bf5895d14856bc  paired_dispatch_plan.json
```

## Byte result

```text
dense HFV1 archive bytes        202649
HFV4 embedded archive bytes     178560
HFV5 profile archive bytes      178541
FEC6/PR110 baseline bytes       178517
```

HFV5 rate deltas:

```text
bytes saved vs dense HFV1       24108
bytes saved vs HFV4 embedded    19
bytes over FEC6/PR110 baseline  24
rate delta vs FEC6/PR110        0.0000159806148749
```

The required component gain to beat FEC6/PR110 on CPU-axis rate arithmetic is
now about `1.60e-05`, down from `2.86e-05` for HFV4 and `5.99e-05` for HFV3.

## Profile-row fidelity

The derived profile row is not float32-bit-exact against the dense active row:

```text
dense active row    [0.0005499999970197678, 1135.3681640625, 1.399999976158142, 581.5, 393.29998779296875]
profile row         [0.00055, 1135.3681714756674, 1.4, 581.5, 393.3]
absolute deltas     [2.98e-12, 7.41e-06, 2.38e-08, 0.0, 1.22e-05]
```

The row is not exact, but the post-warp rounded frame bytes are exact for all
600 pairs in the CPU parity proof below. The failed half-row probe is the
negative control: float16 active-row storage saved 10 bytes but regressed parity
on active pairs with `max_abs_diff=3.0`.

## ZIP anatomy

```text
archive bytes                   178541
members                         1
member name                     x
member compression              stored
member compressed bytes         178441
member uncompressed bytes       178441
central directory bytes         47
extra fields                    none
```

## Full parity proof

- Output directory: `experiments/results/hfv5_profile_inflate_parity_20260521T191803Z`
- JSON: `experiments/results/hfv5_profile_inflate_parity_20260521T191803Z/sparse_foveation_inflate_parity.json`
- Markdown: `experiments/results/hfv5_profile_inflate_parity_20260521T191803Z/sparse_foveation_inflate_parity.md`

Hashes:

```text
3c8367fd842cad32510219e5cebfdaf04680455110ca3b32251654526d48c73c  sparse_foveation_inflate_parity.json
cd554cc0a251d5ca41d2575ae1b61e7f5017f934c5601b5e744a1481bda588b7  sparse_foveation_inflate_parity.md
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
sparse_sidecar_name        embedded_foveation_params.hfv5
```

## Dispatch plan

Plan-only paired Modal dispatch was materialized and did not execute Modal.

Plan result:

- archive SHA matched expected
- no existing promotable CPU anchor was found
- no existing promotable CUDA anchor was found
- pair group: `pair_hfv5_profile_pr101_hfv1_sidecar_exact_eval_6d783ef691d1`
- CPU/CUDA execute commands are recorded in `paired_dispatch_plan.json`

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/build_hfv5_profile_sidecar_candidate.py \
  tools/prove_hfv2_sparse_inflate_parity.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv5_profile_sidecar_candidate.py \
  --output-dir experiments/results/hfv5_profile_sidecar_candidate_20260521T191742Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_hfv2_sparse_inflate_parity.py \
  --sparse-archive experiments/results/hfv5_profile_sidecar_candidate_20260521T191742Z/archive.zip \
  --sparse-submission-dir experiments/results/hfv5_profile_sidecar_candidate_20260521T191742Z/submission_dir_hfv5_profile \
  --batch-pairs 8 \
  --output-dir experiments/results/hfv5_profile_inflate_parity_20260521T191803Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/build_hfv5_profile_sidecar_candidate.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/prove_hfv2_sparse_inflate_parity.py --status reviewed
```

Review tracker result:

- `tools/build_hfv5_profile_sidecar_candidate.py`: 11 entities reviewed
- `tools/prove_hfv2_sparse_inflate_parity.py`: 17 entities reviewed

## Current blocker

I did not execute paired Modal exact eval because `tools/claim_lane_dispatch.py
summary` still reports 13 active dispatch claims, including DP1 paired CPU/CUDA
auth-eval calls and NSCS06/Selfcomp Modal jobs. HFV5 is byte-closed,
full-parity proven, and plan-ready as a research candidate, but it has an
additional compliance review gate before any submission/promotion claim:
profile-coded active-row derivation must be accepted as contest-compliant.
