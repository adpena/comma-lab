# ddm_ffm1 NEXT_IF_RESUMED

1. At `jd8q3_window` endpoint harvest, consume `PREDICTIONS.md` before reading the result. Classify the Q3 projection-vs-conditioning prediction as supported, falsified, or ambiguous using the registered `seg_retention` and pose-held thresholds.
2. When touching subset-gate calibration next, run the queued scorer-free strong-consistency replay: prefix, strided, seeded-random, and stratified estimators over banked n600 per-pair/per-block rows, with d_seg, d_pose, and rate reported separately.
3. Do not launch flow-matching training, open a scorer job, or scale gate noise floors from the paper's iid Bernstein rate. Those rows are folded in `RECEIPT.md`.
4. Do not claim any score movement from this literature-only arm.
