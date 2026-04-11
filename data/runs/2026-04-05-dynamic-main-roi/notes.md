# dynamic main ROI notes

## design intent

- Main ROI must stay primary.
- Auxiliary ROI must never replace the central driving corridor.
- Staticness analysis should inform the mask, but not blindly erase apparently static structure that still matters to SegNet/PoseNet.
- Learned/semantic analysis is allowed only on the compression side in this phase.

## first implementation targets

- `submissions/robust_current/analyze_roi.py`
- metadata-driven path in `submissions/robust_current/compress.sh`
- compatibility check in `submissions/robust_current/inflate.sh`

## evidence discipline

Do not write a promotion claim into `reports/results.jsonl` until a full local CPU evaluation summary exists under `reports/raw/2026-04-05-dynamic-main-roi/`.
