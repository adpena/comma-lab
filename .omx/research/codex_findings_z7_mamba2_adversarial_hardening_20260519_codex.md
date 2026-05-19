# Codex Findings - Z7-Mamba-2 Adversarial Hardening - 2026-05-19

## Scope

Operator directive: burn down remaining Z7-Mamba blockers. This pass reviewed
the Z7-Mamba-2 trainer, runtime export, static-capacity disambiguator, remote
driver, authorization recipe, and exact-eval handoff schema.

## Findings Burned Down

1. Static same-byte control had a hidden 65,535-byte ceiling.
   - Previous path used only ZIP comment padding.
   - Hardened path now uses deterministic no-op metadata padding inside the
     single `0.bin` Z7MCM2 payload, with ZIP comments only for final residual
     bytes.
   - `_full_main` now fails closed if static-control emission fails.

2. Mamba stats were not handoff-schema-compatible.
   - Stats now emit `submission_runtime_dir`, top-level `loss_mode`,
     `score_aware_scorer_loss_used`, and static-control fields expected by
     `tools/verify_z7_exact_eval_handoff.py`.
   - Tiny proxy run now verifies without repo-root runtime-custody scanning;
     it blocks only on expected proxy-loss gates.

3. Remote dispatch custody was too soft.
   - Remote driver now requires an active lane claim and job id.
   - `trap cleanup EXIT` is installed before claim verification.
   - Claim verification uses `summary --live-only --format json`.
   - Terminal rows distinguish claim verification failure, trainer failure,
     and no-score success.

4. False-authority surfaces are now explicit.
   - Trainer defaults to `reference_torch`, matching inflate replay.
   - `mamba_ssm` is opt-in research until state/export replay is implemented.
   - `ego_source` accepts only `frame_delta_proxy`; scorer-derived sources fail
     closed until implemented.
   - `--batch-size` fails closed unless it equals `--max-pairs`, because the
     current loop is whole-sequence.
   - Non-finite loss fails before backward/export.
   - Score-aware loss keeps frozen scorers in eval mode when the wrapper trains.

## Verification

- `65 passed, 11 warnings`:
  `src/tac/tests/test_z7_mamba2_scaffold.py`,
  `src/tac/tests/test_z7_mamba2_substrate_full_landing.py`, and the Z7-Mamba
  readiness test.
- `3 passed`: `src/tac/tests/test_verify_z7_exact_eval_handoff.py`.
- `bash -n scripts/remote_lane_substrate_time_traveler_l5_z7_mamba_2.sh`.
- `git diff --check` on scoped Z7-Mamba files.
- Tiny real-video CPU packet:
  `experiments/results/z7_mamba2_codex_tiny_e2e_20260519T131059Z`.
  It produced equal recurrent/static archive ZIP byte counts, changed runtime
  output under inflate verify, and stayed `score_claim=false`.
- Local remote-driver claim smoke used an isolated temporary claims ledger and
  closed with `completed_z7_mamba2_remote_driver_no_score_claim`.

## Remaining Gates

No known implementation blocker remains in the reviewed Z7-Mamba control path.
Remaining blockers are evidence gates, not silent implementation failures:

- run score-aware full/timing packet with `loss_mode=score_aware`;
- keep static/recurrent same-byte pair and inflate-difference evidence;
- run `tools/verify_z7_exact_eval_handoff.py` clean on the ratified packet;
- only then dispatch paired CPU/CUDA exact eval under lane-claim custody;
- if `mamba_ssm` is reintroduced, first implement byte-faithful state/export
  replay or keep it training-only.

