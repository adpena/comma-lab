# L5 v2 False-Authority Hardening - 2026-05-16

## Context

Read-only L5 v2 adversarial review found two authority leaks:

1. The committed TT5L side-info proof was a two-frame local parser/inflate
   proof, but the canonical helper could return it as satisfied evidence for
   `byte_closed_temporal_sideinfo_consumption`.
2. PR106 PacketIR stack projection trusted matrix row `valid=true` bits instead
   of revalidating exact-axis custody through the shared exact-eval contract.

Both bugs could let local or stale evidence influence L5 v2 staircase readiness
without contest-scale paired custody.

## Fix

- `l5_v2_canonical_sideinfo_gate_evidence()` now loads the proof and reruns
  semantic validation before returning gate evidence.
- The side-info semantic gate now requires contest-scale full-frame custody:
  `proof_scope` in the contest full-frame set, `n_pairs_hashed=600`,
  `total_frames=1200`, `file_list_sha256`, and distinct source/candidate
  raw-output aggregate SHA-256s, mirrored in the inflated-output manifest.
- PR106 PacketIR paired rows are normalized into
  `validate_exact_eval_evidence(...)`; rows missing `n_samples`, score formula
  closure, runtime SHA, hardware, command, log path, artifact path, device
  fields, current runtime match, or source-artifact false-authority flags are
  blocked and do not produce stack-cell candidates.
- Operator briefing preflight now rejects internal L5 ready flags that are true
  while top-level score/rank/exact-dispatch authority remains false.
- The TT5L proof markdown SHA was corrected and the artifact was demoted to a
  local consumption proof until a contest-scale proof lands.

## Evidence

Focused tests added/updated:

- `test_l5_v2_packetir_paired_rows_revalidate_exact_eval_custody`
- `test_l5_v2_sideinfo_consumption_rejects_toy_manifest_scope`
- `test_operator_briefing_dispatch_gate_rejects_l5_authority_leak`

This is not a score claim. It deliberately makes the current L5 v2 briefing
more conservative until the full-frame side-info proof and runtime-bound paired
PacketIR exact candidates exist.
