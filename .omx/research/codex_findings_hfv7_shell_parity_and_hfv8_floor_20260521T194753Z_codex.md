# Codex Findings: HFV7 Shell Parity + HFV8 Universal-Coding Floor

- timestamp_utc: 2026-05-21T19:47:53Z
- lane: hfv7_exp_golomb_pr101_hfv1_sidecar
- status: SHELL_INFLATE_PARITY_PROVEN_AND_HFV8_UNIVERSAL_RECODE_DEFERRED
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What landed

New reusable proof tool:

- `tools/prove_shell_inflate_parity.py`

The tool runs the actual upstream-style shell contract for two packets:

```text
inflate.sh archive_dir output_dir file_list
```

It extracts each archive, runs each side's `inflate.sh`, hashes the emitted raw
output, compares the byte streams with chunked file comparison, writes
JSON/Markdown proof files, and deletes extracted/raw scratch by default. This
closes the medium gap from the first HFV7 review: the prior proof was
in-process frame parity only.

## HFV7 shell-level parity proof

Primary proof artifact, using each packet's own runtime tree:

- Output directory: `experiments/results/hfv7_exp_golomb_shell_inflate_parity_source_runtime_20260521T194843Z`
- JSON: `experiments/results/hfv7_exp_golomb_shell_inflate_parity_source_runtime_20260521T194843Z/shell_inflate_parity.json`
- Markdown: `experiments/results/hfv7_exp_golomb_shell_inflate_parity_source_runtime_20260521T194843Z/shell_inflate_parity.md`

Proof hashes:

```text
dfbfc187cbe94e088ad4d9d8490586369918cca9e70a42056c10cff2dca8564c  shell_inflate_parity.json
29eda23cbc1c8ba85543954c85b89cad5f876ab954094dbd8cd40eb1b10a8844  shell_inflate_parity.md
```

Compared packets:

```text
left archive   experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_seed_top16_component_hardpairs/archive.zip
left runtime   experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/inflate.sh
left sha256    72cbd8197a2a8064cb54e7e56e1a5b892a89251c28091f22eba6eef8edff3efb
left bytes     202649

right archive  experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/archive.zip
right runtime  experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/submission_dir_hfv7_exp_golomb/inflate.sh
right sha256   eaead36921bfccaafa23b8315af97ac2a7b9526a64787f1a1067d477fe064c14
right bytes    178529
```

Result:

```text
file list entry          0.mkv
output raw bytes left    3662409600
output raw bytes right   3662409600
output raw sha left      23dde251c6a1153e61c69215b74a5b7599b55904d69467022aebb3eabd6c40f6
output raw sha right     23dde251c6a1153e61c69215b74a5b7599b55904d69467022aebb3eabd6c40f6
output_bytes_match       true
output_sha256_match      true
cmp_equal                true
left inflate seconds     37.862
right inflate seconds    37.892
scratch_retained         false
```

Secondary archive-isolation proof:

- Output directory: `experiments/results/hfv7_exp_golomb_shell_inflate_parity_tool_20260521T194603Z`
- JSON: `experiments/results/hfv7_exp_golomb_shell_inflate_parity_tool_20260521T194603Z/shell_inflate_parity.json`
- Result: same raw SHA `23dde251c6a1153e61c69215b74a5b7599b55904d69467022aebb3eabd6c40f6`, `cmp_equal=true`

The secondary proof ran both archives through the HFV7 runtime, isolating the
archive-surface change. An earlier manual shell-parity run also produced the
same raw SHA and `cmp_equal=true`. Its multi-GB scratch directories were deleted
after the reusable tool proof landed; only the small
`shell_inflate_parity_summary.txt` remains.

## HFV8 universal-coding floor review

Read-only sister review rechecked the HFV7 delta sequence:

```text
[64, 15, 47, 36, 1, 130, 209, 5, 1, 7, 3, 8, 5, 6, 8, 1]
```

HFV7 is already the best practical universal-code byte result found. To beat
HFV7's 12-byte payload, the next code must fit in at most 88 bits after all
signaling. The faithful families checked do not reach that byte threshold:

| family | best faithful count | bytes | verdict |
|---|---:|---:|---|
| raw uint8 / HFV6 | 128 bits | 16 | worse |
| Exp-Golomb | 96 bits | 12 | HFV7, best byte result |
| Rice | 108 bits | 14 | worse |
| general Golomb | 103 bits | 13 | worse |
| enumerative 16-of-600 | 104 bits | 13 | worse |
| enumerative over FEC6 nonidentity universe | 98 bits | 13 | worse and requires selector-universe authority |
| bounded composition, sum <=599 | 104 bits | 13 | worse |
| bounded composition, exact sum 546 | 96 bits | 12 | no byte win; exact sum must be archive-coded or becomes hidden payload |
| fixed geometric arithmetic | about 104.23 ideal bits | 14 | worse before coder overhead |
| fixed power-law arithmetic | about 96.82 ideal bits | 13 | worse after byte rounding and coder overhead |
| fitted mixture arithmetic | about 94.5-95.8 ideal bits | 12 | no byte win; fitted prior compliance risk |

Conclusion: do not implement HFV8 as a universal delta recode. A real sub-12-byte
HFV8 needs a legitimate archive-derived predictor/universe that reduces the
candidate set to at most 88 bits including signaling. Hardcoding the pair list,
the final pair `546`, a split point, a top-hardpair universe, or empirical
probabilities in `inflate.py` would move payload out of `archive.zip` and is not
contest-faithful unless the side information is charged in the archive.

## Remaining blockers

HFV7 is shell-parity proven and byte-closed, but not submission-ready:

- profile-coded active row is still runtime-side information pending compliance acceptance
- implicit 12-byte trailer interpretation still needs compliance acceptance
- paired contest CPU/CUDA exact eval has not run
- `tools/claim_lane_dispatch.py summary` still reports 13 active Modal claims

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/prove_shell_inflate_parity.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_shell_inflate_parity.py \
  --left-archive experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/archive_seed_top16_component_hardpairs/archive.zip \
  --left-submission-dir experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir \
  --right-archive experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/archive.zip \
  --right-submission-dir experiments/results/hfv7_exp_golomb_sidecar_candidate_20260521T193740Z/submission_dir_hfv7_exp_golomb \
  --python-bin "$PWD/.venv/bin/python" \
  --output-dir experiments/results/hfv7_exp_golomb_shell_inflate_parity_source_runtime_20260521T194843Z
```
