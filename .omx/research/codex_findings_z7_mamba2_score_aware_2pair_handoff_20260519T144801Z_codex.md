# Codex Findings: Z7-Mamba2 Score-Aware 2-Pair Handoff Artifact

timestamp_utc: 2026-05-19T14:48:01Z
actor: codex_session_019de465
lane_id: lane_z7_mamba2_score_aware_2pair_handoff_20260519
evidence_grade: z7_exact_eval_handoff_no_spend
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false

## Summary

The Z7-Mamba2 local implementation blocker was real: `score_aware` mode tried
to import `load_differentiable_scorers` from
`tac.differentiable_eval_roundtrip`, but the canonical loader lives in
`tac.scorer`. After routing the trainer to the canonical scorer loader, the
tiny score-aware recurrent/static same-byte packet builds and the handoff doctor
reaches the expected evidence-only blocker shape.

## Artifact

- Run directory:
  `experiments/results/z7_mamba2_score_aware_2pair_20260519T144552Z`
- Stats JSON:
  `experiments/results/z7_mamba2_score_aware_2pair_20260519T144552Z/z7_mamba2_full_main_export_stats.json`
  - sha256: `a2e612de5f3e8921ff401e09728b4c2d934fbde2d41d883526fb9ff635506424`
- Handoff JSON:
  `experiments/results/z7_mamba2_score_aware_2pair_20260519T144552Z/handoff/z7_exact_eval_handoff_20260519T144801Z.json`
  - sha256: `f624000f4283c6cf3d3b2c1aebc7de0ff777ab668fdf801e397bee562a831028`
- Recurrent archive:
  `experiments/results/z7_mamba2_score_aware_2pair_20260519T144552Z/archive.zip`
  - bytes: `1357696`
  - sha256: `03370f6da6034cbb2aad7d71f876b757b2280b029798f0cc4b350f882a21c47f`
- Static-control archive:
  `experiments/results/z7_mamba2_score_aware_2pair_20260519T144552Z/static_capacity_control/archive.zip`
  - bytes: `1357696`
  - sha256: `43d79b4a7f8a01d9129f7a153dd44a976416fa4f84c49a504d2e0d00ece5eed0`

## Evidence

- `loss_mode=score_aware`
- `score_aware_scorer_loss_used=true`
- `config.num_pairs=2`
- `mamba2_backend_active=reference_torch`
- score-aware scorer load: `1.3210062920115888s`
- one CPU epoch: `3.9640512499026954s`
- total wall: `6.7059769169427454s`
- recurrent inflate raw sha256:
  `3dd5f8374aaadb25605c18978438ab54839a97d46fc5b66cb0fe8c273187f753`
- static-control inflate raw sha256:
  `e04fbaeab4994b11a6c0c64efcfcdf6d9ba63b30c8c00f49ea5b437c73893e68`
- static/recurrent runtime outputs differ by `261729` bytes while archive ZIP
  byte counts match.

## Handoff Verdict

The handoff doctor sets:

- `ready_for_exact_eval_handoff=false`
- `ready_for_paid_dispatch=false`
- `provider_dispatch_attempted=false`
- `result_review_blockers=["z7_exact_handoff_current_packet_not_600_pairs"]`

This burns down the earlier implementation blockers
`loss_mode_not_score_aware` / `score_aware_scorer_loss_not_used` for the tiny
local packet. The remaining blocker is evidence scale: rerun the same path at
`--max-pairs 600 --batch-size 600`, then dispatch paired recurrent/static
contest-CPU and contest-CUDA exact eval under normal lane-claim custody.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check experiments/train_substrate_time_traveler_l5_z7_mamba2.py src/tac/tests/test_z7_mamba2_score_aware_trainer_wiring.py`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools:upstream .venv/bin/python -m pytest -q -p no:cacheprovider src/tac/tests/test_z7_mamba2_score_aware_trainer_wiring.py src/tac/tests/test_z7_mamba2_substrate_full_landing.py src/tac/tests/test_z7_mamba2_scaffold.py src/tac/tests/test_verify_z7_exact_eval_handoff.py src/tac/tests/test_probe_z7_temporal_coherence_vs_static_capacity_disambiguator.py`
  - result: `83 passed, 12 warnings`
