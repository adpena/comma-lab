# Codex Findings: HFV9 Magic Explicit-Row Sidecar Candidate

- timestamp_utc: 2026-05-21T20:11:00Z
- lane: hfv9_magic_explicit_row_pr101_hfv1_sidecar
- status: LANDED_MAGIC_IDENTIFIED_EXPLICIT_ROW_CANDIDATE_WITH_SHELL_PARITY
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What landed

New builder:

- `tools/build_hfv9_magic_explicit_row_sidecar_candidate.py`

HFV9 is the explicit-format counterpart to HFV8. HFV8 paid for the active
foveation row inside `archive.zip`, but runtime format selection still depended
on the trailer length (`32` bytes). HFV9 pays four additional charged bytes for
an `HFV9` magic prefix:

```text
explicit magic bytes      4
explicit row bytes        20
Exp-Golomb delta bytes    12
total payload bytes       36
```

The FP11 wrapper determines the trailer boundary from `source_len` and
`selector_len`; the charged `HFV9` magic determines the trailer format. This is
not a byte win against HFV7 or HFV8. It is a compliance-fallback artifact for
the case where a reviewer rejects length-only HFV8 format discrimination.

## Candidate artifact

- Output directory: `experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z`
- Archive: `experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/archive.zip`
- Submission runtime: `experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/submission_dir_hfv9_magic_explicit_row`
- Manifest: `experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/hfv9_magic_explicit_row_manifest.json`
- Paired dispatch plan: `experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/paired_dispatch_plan.json`

Hashes:

```text
9a32b1311da1076b1659ff6652481383527279905c8a135eeedff6ec913888ac  archive.zip
e4fc0d438736929bdffc24703e292aa1e103bf28d24085bd5ce851c16613c75e  hfv9_magic_explicit_row_manifest.json
9ec296bb667123c279261a06454926d99cd8254e7afbea677a2d6181e3b94465  submission_dir_hfv9_magic_explicit_row/archive_manifest.json
5214d80049fba25d2de65d6d3bbc06e297abab6195fc48e96a7de03042a524fc  submission_dir_hfv9_magic_explicit_row/inflate.py
4961f9e2392965ae7c8351f64895551493b06cce8139eb23f5000ab9ec0c044e  paired_dispatch_plan.json
f66c5894f5a1647b01ae5efcfea2f936d143188d030b9f0c6d2d8f2dfc3727ab  embedded HFV9 payload
```

## Byte result

```text
dense HFV1 archive bytes        202649
HFV7 Exp-Golomb archive bytes   178529
HFV8 explicit-row archive bytes 178549
HFV9 magic explicit-row bytes   178553
FEC6/PR110 baseline bytes       178517
```

HFV9 rate deltas:

```text
bytes saved vs dense HFV1       24096
bytes over HFV8                 4
bytes over HFV7                 24
bytes over FEC6/PR110 baseline  36
rate delta vs FEC6/PR110        0.0000239709223124
```

## ZIP anatomy

```text
archive bytes                   178553
members                         1
member name                     x
member compression              stored
member compressed bytes         178453
member uncompressed bytes       178453
central directory bytes         47
extra fields                    none
```

## Shell parity proof

Primary proof artifact, using each packet's own runtime tree:

- Output directory: `experiments/results/hfv9_magic_explicit_row_shell_inflate_parity_source_runtime_20260521T200930Z`
- JSON: `experiments/results/hfv9_magic_explicit_row_shell_inflate_parity_source_runtime_20260521T200930Z/shell_inflate_parity.json`
- Markdown: `experiments/results/hfv9_magic_explicit_row_shell_inflate_parity_source_runtime_20260521T200930Z/shell_inflate_parity.md`

Proof hashes:

```text
cf16a9831c576fcdf2ba52e1d942681015cd8efc0c11395537aaff812a0e4e04  shell_inflate_parity.json
10e5208a84d21515a26b0c02150133eb1a2d35181b06f355c80ecf09b5b43511  shell_inflate_parity.md
```

Result:

```text
left archive              experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_seed_top16_component_hardpairs/archive.zip
left runtime              experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/inflate.sh
right archive             experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/archive.zip
right runtime             experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/submission_dir_hfv9_magic_explicit_row/inflate.sh
output raw bytes left     3662409600
output raw bytes right    3662409600
output raw sha left       23dde251c6a1153e61c69215b74a5b7599b55904d69467022aebb3eabd6c40f6
output raw sha right      23dde251c6a1153e61c69215b74a5b7599b55904d69467022aebb3eabd6c40f6
output_bytes_match        true
output_sha256_match       true
cmp_equal                 true
left inflate seconds      37.278
right inflate seconds     37.619
scratch_retained          false
```

## Dispatch plan

Plan-only paired Modal dispatch was materialized and did not execute Modal.

Plan result:

- archive SHA matched expected
- pair group: `pair_hfv9_magic_explicit_row_pr101_hfv1_sidecar_exact_eval_9a32b1311da1`
- CPU/CUDA execute commands are recorded in `paired_dispatch_plan.json`

## Current blocker

I did not execute paired Modal exact eval because `tools/claim_lane_dispatch.py
summary` still reports 13 active dispatch claims, including DP1 paired CPU/CUDA
auth-eval calls and NSCS06/Selfcomp Modal jobs.

HFV9 removes HFV8's length-only format-discriminator concern. It still has no
score claim and needs fresh strict pre-submission compliance review plus paired
contest CPU/CUDA exact eval before any promotion or submission claim.

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/build_hfv9_magic_explicit_row_sidecar_candidate.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv9_magic_explicit_row_sidecar_candidate.py \
  --output-dir experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
  --archive experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/submission_dir_hfv9_magic_explicit_row/archive.zip \
  --submission-dir experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/submission_dir_hfv9_magic_explicit_row \
  --inflate-sh inflate.sh \
  --label hfv9_magic_explicit_row_pr101_hfv1_sidecar_exact_eval \
  --expected-archive-sha256 9a32b1311da1076b1659ff6652481383527279905c8a135eeedff6ec913888ac \
  --run-id hfv9_magic_explicit_row_pr101_9a32b1311da1 \
  --pair-group-id pair_hfv9_magic_explicit_row_pr101_hfv1_sidecar_exact_eval_9a32b1311da1 \
  --lane-id-base hfv9_magic_explicit_row_pr101_hfv1_sidecar_exact_eval \
  --output-root experiments/results \
  --modal-bin .venv/bin/modal \
  --gpu T4 \
  --claim-agent codex:hfv9_magic_explicit_row_sidecar \
  --claim-notes "HFV9 magic explicit-row sidecar candidate; score_claim=false until shell parity and paired contest CPU/CUDA exact eval harvest." \
  --expected-runtime-tree-sha256 auto \
  --skip-axis-if-promotable-anchor-exists \
  --json-out experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/paired_dispatch_plan.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_shell_inflate_parity.py \
  --left-archive experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_seed_top16_component_hardpairs/archive.zip \
  --left-submission-dir experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir \
  --right-archive experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/archive.zip \
  --right-submission-dir experiments/results/hfv9_magic_explicit_row_sidecar_candidate_20260521T200823Z/submission_dir_hfv9_magic_explicit_row \
  --python-bin "$PWD/.venv/bin/python" \
  --output-dir experiments/results/hfv9_magic_explicit_row_shell_inflate_parity_source_runtime_20260521T200930Z
```
