# Codex Findings: Grayscale LUT Export Recovery

- UTC: 20260522T123316Z
- Lane: lane_codex_grayscale_lut_export_recovery_hardening_20260522
- Evidence grade: local export artifact only
- Score authority: false
- Promotion eligible: false
- Rank or kill eligible: false

## Result

The timed-out Modal A100 grayscale-LUT run had a recoverable `best.pt`, and the
local export-only path produced byte-closed archive artifacts without retraining
or auth-score claims.

- Source checkpoint:
  `experiments/results/lane_substrate_grayscale_lut_lut_bits_5_modal_a100_dispatch_20260521T185859Z_modal/harvested_artifacts/lane_substrate_grayscale_lut_results/output/best.pt`
- Export directory:
  `experiments/results/grayscale_lut_lut_bits_5_export_recovery_20260522T121308Z`
- Archive ZIP:
  `experiments/results/grayscale_lut_lut_bits_5_export_recovery_20260522T121308Z/archive.zip`
- Archive ZIP SHA-256:
  `99203f6b0858e8bd54bbc8b88b0a1583ed49f4c75d75590c1ce1951ecfcfda13`
- Payload `0.bin` SHA-256:
  `78878f6f21f5e666de23b8df16e58fbd3b9b116e7ffd93448ed175aa51dfe95e`
- Payload `0.bin` bytes: `1808596`

No contest CPU/CUDA auth eval was run in this step. The artifact is a recovered
archive candidate, not a score claim.

## Canonical State

Lane maturity was updated through `tools/lane_maturity.py`:

- `impl_complete`: true, evidence `experiments/train_substrate_grayscale_lut.py`
- `real_archive_empirical`: true, evidence recovered `archive.zip`
- `strict_preflight`: true, evidence `src/tac/tests/test_grayscale_lut_export_recovery.py`
- `deploy_runbook`: true, evidence `scripts/remote_lane_substrate_grayscale_lut.sh`
- Computed level: `L2`

Contest CPU, contest CUDA, three-clean-review, and score/promotion gates remain
false.

## Verification

- Export command used `--export-only-checkpoint`, `--epochs 0`, `--device cpu`,
  and `--skip-auth-eval`.
- Output wrote `0.bin`, `archive.zip`, and `provenance.json`.
- The focused recovery tests remain the guard surface:
  `src/tac/tests/test_grayscale_lut_export_recovery.py`.

## Next Action

Run exact auth eval on the recovered archive only through the normal claimed
contest-axis path. If that score is noncompetitive, keep the archive as a
recovery proof and feed the timeout lesson into future grayscale-LUT paid
recipes: soft deadline, early-stop export, and no Modal hard-timeout reliance.
