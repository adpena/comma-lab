# Test spec — smarter segmentation main ROI

## Verification targets

1. ROI analysis helper emits bounded, even-sized metadata and always produces a main ROI.
2. Metadata-driven packaging succeeds and archive layout matches inflate expectations.
3. Inflation succeeds and shape/frame-count checks pass.
4. One full authoritative local CPU evaluation is recorded.
5. Promotion happens only if the measured score beats 3.33.
6. Results/state/writeups are updated with explicit current_workflow vs rule_faithful separation.

## Minimum commands

- `uv run python submissions/robust_current/analyze_roi.py --help`
- `python3 -m py_compile submissions/robust_current/analyze_roi.py`
- `bash -n submissions/robust_current/compress.sh`
- `bash -n submissions/robust_current/inflate.sh`
- `bash submissions/robust_current/compress.sh`
- `uv run comma-lab eval-submission robust_current --device cpu`

## Failure conditions

- Main ROI missing or unstable
- Archive/inflate mismatch
- Unbounded aux ROI growth
- Claimed improvement without full local measured summary
