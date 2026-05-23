# Codex Findings - Inverse Scorer Inflate Parity Gate

UTC: 2026-05-23T19:11:38Z

## Scope

Follow-on adversarial review and hardening for the IAS1 inverse-scorer candidate
receiver chain. The prior chain proved descriptor consumption but intentionally
left full-frame inflate parity and contest auth eval blocked.

## Findings And Fixes

- Descriptor receiver evidence was not enough to prevent future false authority
  if a downstream consumer treated `receiver_contract_satisfied=true` as
  runtime readiness. The verifier now has a separate
  `inverse_scorer_cell_inflate_parity_probe_v1` proof and clears only
  `candidate_inflate_output_parity_missing` when that proof is strict.
- The parity proof is bound to the candidate archive, source/template archive,
  candidate member SHA, and IAS1 descriptor packet offset/bytes/SHA. Truthy
  nested authority fields fail closed.
- The chain can now either consume precomputed inflate output directories or run
  an `inflate.sh` runtime against the source and candidate archives, compare the
  output trees, and retain compact telemetry while leaving raw work dirs
  disposable unless explicitly retained.
- The queue and DAG path preserves parity metadata and exposes runtime-driven
  parity commands through `tools/run_inverse_scorer_cell_candidate_chain.py`.
- `tools/operator_briefing.py` now surfaces compact IAS1 chain readiness rows
  from recent chain manifests, so ignored `experiments/results` artifacts do not
  become invisible evidence. Gate #28 in `tools/all_lanes_preflight.py` rejects
  inverse-scorer rows that claim score/dispatch authority or hide missing parity
  or exact-auth blockers.
- A concurrent/selective-runtime patch that parses IAS1 after DQS1 in generated
  decoder-q runtimes was reviewed and covered with a focused test so the signal
  is not orphaned.

## Verification

- `src/tac/tests/test_inverse_scorer_cell_materializer.py`: 25 passed.
- Focused queue/operator/decoder-q slice: 14 passed.
- `src/tac/tests/test_all_lanes_operator_briefing_gate.py` plus operator
  briefing readiness test: 30 passed.
- Targeted `ruff check`, `py_compile`, and `git diff --check` passed.

## Remaining Gates

- This is still not score authority. Exact `[contest-CPU]` or `[contest-CUDA]`
  auth eval remains required before any score, promotion, rank, kill, or
  dispatch-readiness claim.
- The real IAS1 candidate should next run the actual source/candidate
  `inflate.sh` parity path against a full file list, with raw-output custody
  summarized into tracked `.omx/research` and bulky outputs handled by storage
  retention policy.
