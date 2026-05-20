# PR101 LFV1/HFV1 Seed Candidate: Top-16 Component-Hard Pairs

## Verdict

Built one no-score, byte-closed HFV1 seed candidate from existing PR101
component-response evidence. Official `inflate.sh` raw-output locality control
passed. This is still not an exact-eval score claim.

## Source Surface

- Source rows:
  `experiments/results/frame_exploit_segnet_posenet_20260514_pr101_cpu_top64_topmodes_codex/pair_component_rows.jsonl`
- Selection rule:
  top 16 identity-mode pairs by `component_score_no_rate`.
- Selected pairs:
  `[508, 546, 502, 518, 515, 79, 293, 507, 126, 537, 64, 545, 162, 531, 526, 163]`
- Selected frames:
  pair `k` maps to frames `2k` and `2k+1`.

This keeps SegNet and PoseNet sensitivity separated:

- PoseNet route: pair-indexed.
- SegNet route: frame-indexed after pair expansion.
- No component-response claim is made for the HFV1 seed itself.

## Candidate Artifact

- Manifest:
  `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/seed_top16_component_hardpairs_manifest.json`
- Manifest SHA-256:
  `fb1b2689f87df813755dd9b975c9aa981a793e4194fa08b7bc6d739643a7a16e`
- Archive:
  `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_seed_top16_component_hardpairs/archive.zip`
- Archive bytes:
  `202649`
- Archive SHA-256:
  `72cbd8197a2a8064cb54e7e56e1a5b892a89251c28091f22eba6eef8edff3efb`
- `foveation_params.bin` bytes:
  `24016`
- `foveation_params.bin` SHA-256:
  `f1dbcf02973957b4f4e30bd2638629051f4a986c3010bed32c954f500c6d1551`
- `x` member bytes:
  `178417`
- `x` member SHA-256:
  `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`

## Official Inflate Control

Command shape:

```bash
env PACT_PYTHON_BIN=/Users/adpena/Projects/pact/.venv/bin/python \
  time bash experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/inflate.sh \
  experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/official_inflate_control/data_seed_top16_component_hardpairs \
  experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/official_inflate_control/outputs/seed_top16_component_hardpairs \
  experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/official_inflate_control/file_list.txt
```

Result:

- `saved 1200 frames`
- Wall time: `38.22 real`
- Raw bytes from `wc -c`: `3662409600`
- Raw SHA-256 from `shasum -a 256`:
  `23dde251c6a1153e61c69215b74a5b7599b55904d69467022aebb3eabd6c40f6`

Raw comparison:

- Report:
  `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/official_inflate_control/seed_top16_component_hardpairs_raw_comparison.json`
- Report SHA-256:
  `750655bcd155c37eea1eb9e8c4bcc92248e0e88df093a44bdd51d6a62ce9dc83`
- Passed:
  `true`
- Changed frame count:
  `32`
- Changed frames match selected pair expansion:
  `true`

## Remaining Blockers

- `component_response_not_measured_for_hfv1_seed`
- `exact_cuda_auth_eval_missing`

## Next

Run a local component-response smoke for the HFV1 seed if a scorer-facing
component harness is available without violating no-scorer-at-inflate. Then
replace the uniform small radial seed with a sensitivity-weighted/orthogonalized
HFV1 search over the same PR110-canonical runtime root after PR110 merges.
