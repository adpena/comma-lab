# Codex Session Summary 20260523T203247Z

## Landed

- Canonicalized IAS1 runtime-parity top4 exact CPU/CUDA results as result-review
  packets and appended their conservative evidence rows to
  `reports/cathedral_autopilot_evidence.jsonl`.
- Added paired exact-auth calibration ingestion to
  `tac.optimization.inverse_steganalysis_acquisition`.
- Added CLI ingestion for paired exact-auth calibration packets in
  `tools/build_inverse_steganalysis_action_functional.py`.
- Added regression tests proving exact-auth regression demotes the measured IAS1
  config to zero expected gain while preserving false authority.

## Empirical Anchor

- CPU `[contest-CPU Linux x86_64]`: `0.19380912393883232` vs frontier
  `0.19202828295713675`.
- CUDA `[contest-CUDA T4]`: `0.2279696105246996` vs frontier
  `0.20533002902019143`.
- Shared archive:
  `2d0850789483e17c7ee68ae8bfe1e33489d1981416f71266cf8a66b19a87e549`,
  `181232` bytes.
- Paired calibration penalty: `0.024420422463531932`.

## Outstanding

- The calibration currently applies to the measured IAS1 bundle candidate. It
  does not yet learn per-cell inheritance rules for future inverse-action cells.
- Additional exact pairs are needed before safely generalizing this penalty into
  a family-wide trust-region prior.
