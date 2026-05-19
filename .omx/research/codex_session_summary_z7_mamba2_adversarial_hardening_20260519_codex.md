# Codex Session Summary - Z7-Mamba-2 Hardening - 2026-05-19

## Changed

- Hardened `experiments/train_substrate_time_traveler_l5_z7_mamba2.py`.
- Hardened `scripts/remote_lane_substrate_time_traveler_l5_z7_mamba_2.sh`.
- Updated the Z7-Mamba authorization recipe to match the actual reference
  runtime and remote-driver identity contract.
- Added focused unit/subprocess tests for static same-byte controls, bool
  parsing, fail-closed guards, runtime vendoring, remote claim closure, and
  scorer eval-mode preservation.

## Evidence

- Z7-Mamba focused tests: `65 passed, 11 warnings`.
- Z7 handoff verifier tests: `3 passed`.
- Handoff verifier on tiny Mamba packet: no runtime-custody blocker; blocked
  only on expected proxy-loss fields.
- Tiny real-video CPU packet with inflate verify:
  `experiments/results/z7_mamba2_codex_tiny_e2e_20260519T131059Z`.
- Local remote-driver smoke with isolated claims ledger closed terminal
  no-score success.

## Authority

This landing does not claim a contest score and does not make Z7-Mamba
promotion-eligible. It converts prior ambiguous blockers into either
implemented guards or explicit evidence gates.

