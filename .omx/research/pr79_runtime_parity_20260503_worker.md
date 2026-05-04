# PR79 Runtime Parity Forensics - 2026-05-03

## Scope

Implemented `experiments/compare_pr79_runtime_parity.py` as a local-only
forensic tool for comparing public PR79
`submissions/qpose14_r55_segactions_minp` inflate/runtime custody with
`submissions/robust_current` on identical archive bytes.

This does not run `upstream/evaluate.py`, does not dispatch remote GPU jobs,
and does not modify scorer files. Exact CUDA auth eval remains the score
truth.

## Commands

Focused verification:

```bash
python3 -m py_compile experiments/compare_pr79_runtime_parity.py
.venv/bin/python -m pytest src/tac/tests/test_compare_pr79_runtime_parity.py -q
```

Default forensic profile command:

```bash
.venv/bin/python experiments/compare_pr79_runtime_parity.py \
  --archive experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/pr79_s2_fixed_adaptive_actions/archive.zip \
  --pr79-checkout /tmp/pact_pr79_inspect \
  --output-json experiments/results/pr79_runtime_parity_20260503_worker/pr79_runtime_parity.json
```

Local dry forensic runs, with reports written to `/tmp` to avoid creating
extra repo artifacts:

```bash
.venv/bin/python experiments/compare_pr79_runtime_parity.py \
  --archive experiments/results/pr79_action_dictionary_repack_v2_20260503_codex/pr79_s2_fixed_adaptive_actions/archive.zip \
  --pr79-checkout /tmp/pact_pr79_inspect \
  --output-json /tmp/pr79_runtime_parity_check.json

.venv/bin/python experiments/compare_pr79_runtime_parity.py \
  --archive experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip \
  --pr79-checkout /tmp/pact_pr79_inspect \
  --output-json /tmp/pr79_public_runtime_parity_check.json
```

Observed dry-run summaries:

- S2 archive `5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68`:
  public PR79 parse failed with `brotli: decoder failed`; robust_current parsed
  `public_pr75_qzs3_qp1_segactions_fixed_slices`. This flags a
  public-runtime compatibility gap for the S2 action wire.
- Public PR79 archive
  `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`:
  public and robust parsed successfully; decoded member hashes, QP1 float32
  pose hash, and canonical action-record hash all matched; no gap classes.

Optional rendered-byte parity command, still scorer-free:

```bash
.venv/bin/python experiments/compare_pr79_runtime_parity.py \
  --archive <archive.zip> \
  --pr79-checkout /tmp/pact_pr79_inspect \
  --run-raw-parity \
  --file-list public_test_video_names.txt \
  --output-json <report.json>
```

## Detection Surface

The tool records:

- ZIP custody: member names, compressed sizes, uncompressed sizes, CRCs, and
  SHA-256 hashes, with fail-closed duplicate and zip-slip checks.
- Payload custody: exact `p`/`renderer_payload.bin*` bytes and SHA-256.
- Public PR79 slicing behavior: legacy fixed slices, P2, P3, and the public
  minp fixed-window path.
- Robust runtime slicing behavior: current `unpack_renderer_payload.py`
  profile, including P3/P4/P5/P6 and S1/S2 action codecs.
- Decoded member hashes for `masks.mkv`, `renderer.bin`,
  `optimized_poses.qp1`, and `seg_tile_actions.bin`.
- QP1 float32 pose hash parity, to catch the older QP1-to-fp16 materialization
  drift class.
- Canonical decoded action-record hash parity, to catch P3/P6/S1/S2/dictionary
  runtime drift before any scorer is involved.
- Runtime source manifests for public PR79 `inflate.py`/`inflate.sh` and
  robust_current `inflate.sh`/`inflate_renderer.py`/`unpack_renderer_payload.py`.
- Optional raw `.raw` output SHA-256 parity when explicitly requested.

Known score context is included only as non-recomputed context:
public PR79 body formula score `0.31372571308675656`; robust_current PR79 S2
exact T4 replay score `0.31453355357318635`.

## Interpretation

If public and robust decoded members, QP1 float32 pose hashes, action-record
hashes, and optional raw-output hashes all match, this tool rules out
archive/runtime byte drift as the explanation for the public-body formula
score versus exact-T4 replay gap. It still cannot prove scorer parity because
it deliberately avoids `upstream/evaluate.py`.

If public PR79 cannot parse the archive but robust_current can, the report
flags `public_pr79_parse_or_decode_failed`. This is expected for robust-only
codec extensions such as S2 adaptive actions; those archives may be exact-eval
legal under our runtime but are not public-branch inflate-equivalent.
