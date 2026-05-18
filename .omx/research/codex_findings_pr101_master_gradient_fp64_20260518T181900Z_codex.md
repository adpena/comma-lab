# Codex Findings: PR101_lc_v2 Master-Gradient fp64 Diagnostic

Date: 2026-05-18T18:19:00Z

Actor: Codex

Task: `codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518::ITEM_3`

## Verdict

The second projection-supported grammar, `pr101_lc_v2`, now has end-to-end fp64 diagnostic materialization evidence through `tools/extract_master_gradient.py`.

This is diagnostic CPU subset evidence only. It is not `[contest-CPU]` or `[contest-CUDA]` authority because it used `--axis '[diagnostic-CPU]'`, `--n-pairs-used 8`, local CPU hardware, and `--no-anchor-write`.

## Archive

- Archive: `experiments/results/public_pr_archive_release_view/public_pr101_intake_20260505_auto/archive.zip`
- Inflate path: `experiments/results/public_pr_archive_release_view/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/inflate.py`
- Charged archive bytes: `178258`
- Charged archive SHA-256: `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- Gradient subject domain: `zip_inner_member_payload`
- Gradient subject bytes: `178158`
- Gradient subject SHA-256: `5f1948f9572e65f71c614d2ff15764ee416522e25cb1b06c8b1299c1306e8aaf`
- Detected grammar: `pr101_lc_v2`

## Empirical Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/extract_master_gradient.py \
  --archive experiments/results/public_pr_archive_release_view/public_pr101_intake_20260505_auto/archive.zip \
  --inflate-py experiments/results/public_pr_archive_release_view/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/inflate.py \
  --upstream-dir upstream \
  --axis '[diagnostic-CPU]' \
  --output-npy .omx/state/master_gradient_pr101_lc_v2_diagnostic_8pair_20260518.npy \
  --per-pair-output-npy .omx/state/master_gradient_pr101_lc_v2_diagnostic_8pair_per_pair_20260518.npy \
  --device cpu \
  --n-pairs-used 8 \
  --n-pairs-total 600 \
  --preserve-per-pair \
  --compute-dtype float64 \
  --storage-dtype float64 \
  --no-anchor-write \
  --verbose
```

## Tensor Artifacts

These `.npy` sidecars are ignored state artifacts and are intentionally not committed.

| Path | Shape | Dtype | SHA-256 | Notes |
|---|---:|---|---|---|
| `.omx/state/master_gradient_pr101_lc_v2_diagnostic_8pair_20260518.npy` | `(178158, 3)` | `float64` | `31088a20b302424f550d97a25925eb3cd5298bbb0300e1c064ee297db44f6938` | finite; absmax `3.60958110832e-05`; nonzero `323904` |
| `.omx/state/master_gradient_pr101_lc_v2_diagnostic_8pair_per_pair_20260518.npy` | `(178158, 8, 3)` | `float64` | `e90809edf3110419dd0624d66d94ab74d1363bf692b883fb7f78dd851c0840d2` | finite; absmax `6.1124075728e-05`; nonzero `2591232` |

## Operating Point

- `d_seg=0.0009`
- `d_pose=0.00175394`
- `rate=0.004748`
- `score=0.3393`
- `dS/d_seg=100.0`
- `dS/d_pose=37.75`
- `dS/d_byte=6.659e-07`
- Forward plus 16 backward passes completed in `57.55s`

## Scope After This Run

Projection-supported grammar families with diagnostic fp64 materialization now include:

- `a1_finetuned`
- `pr101_lc_v2`

Remaining ITEM_3 work is projector design or explicit fail-closed closure for packed/length-prefixed families:

- `pr106_format0d`
- `pr106_ff_packed_hnerv`
- `hnerv_lc_v2_length_prefixed`
- `pr107_apogee_length_prefixed`
