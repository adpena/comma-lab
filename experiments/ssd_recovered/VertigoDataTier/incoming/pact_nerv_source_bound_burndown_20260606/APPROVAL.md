# APPROVAL

Verdict:

```text
HiNeRV long training: blocked
SNeRV long training: blocked
```

## Minimum blockers before launch

HiNeRV:

1. Parse-back archive selection available and required.
2. Target-region hard-birth actuator passes scoped smoke.
3. Joint Seg/Pose trust-region telemetry passes.
4. Section value-per-byte rows emitted.
5. Receiver proof and full-video MLX replay available for selected archive.

SNeRV:

1. Official MFU/HFR/TUB full source-forward closure.
2. Trained checkpoint/state ingestion bound.
3. Temporal encoder/output2 decoder mapping closed.
4. LF/HF learned replacement measured in score units per byte.
5. Receiver source-forward replay bound.

## Top 10 implementation tasks

1. Shared parse-back archive selection.
2. HiNeRV target-region birth actuator.
3. Joint Seg/Pose trust region.
4. SNeRV full TUB source-forward closure.
5. Section value-per-byte ledger.
6. Full-video MLX replay gate.
7. HiNeRV scoped output-head/feature-grid gradient report.
8. SNeRV LF/HF score-unit section profiler.
9. QAT timing ablation.
10. PR95 stage-faithful selection smoke.

## Top 10 risks

1. Local proxy selection overclaims archive quality.
2. Target coverage hides target-region zero hard support.
3. SegNet edit damages PoseNet at high pose marginal.
4. SNeRV receiver primitive proof confused with official source-forward proof.
5. Byte cap applied before distortion birth.
6. Modelsize knobs act on capacity but not actual archive bytes.
7. QAT suppresses class birth.
8. EMA smooths away rare class islands.
9. MLX/Torch/NumPy scorer drift hides wrong update direction.
10. Metadata/JSON silently bloats archive sections.

## Approval condition

A long run becomes approved only when `tools/validate_nerv_long_run_gate.py` can consume:

- parse-back archive selection manifest,
- target-region birth smoke,
- joint Seg/Pose trust-region rows,
- section value-per-byte ledger,
- receiver source-forward proof,
- full-video MLX replay artifact.
