# Codex Findings: HFV8 Explicit-Row Sidecar Candidate

- timestamp_utc: 2026-05-21T19:58:53Z
- lane: hfv8_explicit_row_pr101_hfv1_sidecar
- status: LANDED_EXPLICIT_ROW_BYTE_CLOSED_CANDIDATE_WITH_SHELL_PARITY
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What landed

New builder:

- `tools/build_hfv8_explicit_row_sidecar_candidate.py`

Lane registry:

- `hfv8_explicit_row_pr101_hfv1_sidecar` registered at L0, then marked L2
  after implementation and real-archive empirical evidence.

HFV8 is the compliance-safe counterpart to HFV7. HFV7 has the best rate result
for the active-pair delta sequence, but its active foveation row is a runtime
profile. HFV8 pays for that row inside `archive.zip` as five float32 values,
then appends the same 12-byte HFV7 Exp-Golomb delta stream:

```text
explicit row bytes        20
Exp-Golomb delta bytes    12
total payload bytes       32
```

This is not a byte win against HFV7. It is a compliance-fallback artifact:
pair-list information and active-row values are both archive-contained.

## Candidate artifact

- Output directory: `experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z`
- Archive: `experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z/archive.zip`
- Submission runtime: `experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z/submission_dir_hfv8_explicit_row`
- Manifest: `experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z/hfv8_explicit_row_manifest.json`
- Paired dispatch plan: `experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z/paired_dispatch_plan.json`

Hashes:

```text
b3e89b16b1107af1a661ea20c81f7cb689530f6591ac988b346cfa05c459eef8  archive.zip
fa0c8c78454218e8eb297599553f2ed3e261d3cfd3344b1e8bccfa56f25263f7  hfv8_explicit_row_manifest.json
1002db76db46449a08fd92ae39183b2f2adfd5b68e127be3915cae5b3d43cedb  submission_dir_hfv8_explicit_row/archive_manifest.json
4e768a04ecf4dbb288da1507acf0d9fffc99ff8cfb1d103210ce09f041b9ca48  submission_dir_hfv8_explicit_row/inflate.py
0594c6f2ac0dfed1e85e113321e55d751657dc87abf4bc357f499a84c79ef5e4  paired_dispatch_plan.json
d0fbfae65567b60f6d9bf5fc6be0d18f2be20472726a0564fb7329fb9b04aa96  embedded HFV8 payload
```

## Byte result

```text
dense HFV1 archive bytes        202649
HFV7 Exp-Golomb archive bytes   178529
HFV8 explicit-row archive bytes 178549
FEC6/PR110 baseline bytes       178517
```

HFV8 rate deltas:

```text
bytes saved vs dense HFV1       24100
bytes over HFV7                 20
bytes over FEC6/PR110 baseline  32
rate delta vs HFV7              0.0000133171790624
rate delta vs FEC6/PR110        0.0000213074864999
```

## ZIP anatomy

```text
archive bytes                   178549
members                         1
member name                     x
member compression              stored
member compressed bytes         178449
member uncompressed bytes       178449
central directory bytes         47
extra fields                    none
```

## Shell parity proof

Primary proof artifact, using each packet's own runtime tree:

- Output directory: `experiments/results/hfv8_explicit_row_shell_inflate_parity_source_runtime_20260521T195718Z`
- JSON: `experiments/results/hfv8_explicit_row_shell_inflate_parity_source_runtime_20260521T195718Z/shell_inflate_parity.json`
- Markdown: `experiments/results/hfv8_explicit_row_shell_inflate_parity_source_runtime_20260521T195718Z/shell_inflate_parity.md`

Proof hashes:

```text
77edca9850cf34d20c5287d9d92047e4643eb4af081e789ca8cf9f6c97b5024b  shell_inflate_parity.json
4f6d42438e92596731a7b98934227d1a9377e86eb8cb8d8fd18ab5718285448a  shell_inflate_parity.md
```

Result:

```text
left archive              experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_seed_top16_component_hardpairs/archive.zip
left runtime              experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/inflate.sh
right archive             experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z/archive.zip
right runtime             experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z/submission_dir_hfv8_explicit_row/inflate.sh
output raw bytes left     3662409600
output raw bytes right    3662409600
output raw sha left       23dde251c6a1153e61c69215b74a5b7599b55904d69467022aebb3eabd6c40f6
output raw sha right      23dde251c6a1153e61c69215b74a5b7599b55904d69467022aebb3eabd6c40f6
output_bytes_match        true
output_sha256_match       true
cmp_equal                 true
left inflate seconds      37.330
right inflate seconds     37.549
scratch_retained          false
```

## Implementation bug caught and fixed before landing

The first HFV8 generated runtime failed shell parity before writing the right
raw output because the builder checked for the substring `HFV8_ROW_STRUCT`,
which was already present in the inserted decoder function, and therefore did
not insert the actual struct definition. The fix changes the guard to look for
`HFV8_ROW_STRUCT = ` before insertion. The failed scratch directory was reduced
to 4 KB after cleanup; the passing proof above is from the regenerated
`20260521T195658Z` artifact.

## Dispatch plan

Plan-only paired Modal dispatch was materialized and did not execute Modal.

Plan result:

- archive SHA matched expected
- pair group: `pair_hfv8_explicit_row_pr101_hfv1_sidecar_exact_eval_b3e89b16b110`
- CPU/CUDA execute commands are recorded in `paired_dispatch_plan.json`

## Current blocker

I did not execute paired Modal exact eval because `tools/claim_lane_dispatch.py
summary` still reports 13 active dispatch claims, including DP1 paired CPU/CUDA
auth-eval calls and NSCS06/Selfcomp Modal jobs. HFV8 is byte-closed and
shell-parity proven, but it still needs implicit 32-byte trailer grammar review
before any submission/promotion claim.

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/build_hfv8_explicit_row_sidecar_candidate.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_hfv8_explicit_row_sidecar_candidate.py \
  --output-dir experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_shell_inflate_parity.py \
  --left-archive experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_seed_top16_component_hardpairs/archive.zip \
  --left-submission-dir experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir \
  --right-archive experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z/archive.zip \
  --right-submission-dir experiments/results/hfv8_explicit_row_sidecar_candidate_20260521T195658Z/submission_dir_hfv8_explicit_row \
  --python-bin "$PWD/.venv/bin/python" \
  --output-dir experiments/results/hfv8_explicit_row_shell_inflate_parity_source_runtime_20260521T195718Z
```
