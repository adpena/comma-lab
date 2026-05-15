---
sister_of: local_hardware_aggressive_sweep_5_streams_LANDED_20260513.md
audit_subagent_id: HISTORICAL-EVIDENCE-AUDIT-20260515
remediation_subagent_id: WAVE-B-23-ROW-CORRUPTION-REMEDIATION-20260515
audit_date_utc: 2026-05-15T20:35:00Z
remediation_date_utc: 2026-05-15
manifest: .omx/research/historical_evidence_audit_20260515.json
classification: POTENTIALLY-CORRUPTED-DEFERRED-PENDING-RE-VERIFICATION
verdict: NO_KILL
hooks_active: ["sensitivity_map_NA", "pareto_NA", "bit_allocator_NA", "cathedral_autopilot_dispatch_hook_records_to_continual_learning", "continual_learning_anchor_re-tag_persisted_via_canonical_helper", "probe_disambiguator_NA"]
---

# Sister audit re-tag for `local_hardware_aggressive_sweep_5_streams_LANDED_20260513.md`

## Audit summary

`HISTORICAL-EVIDENCE-AUDIT-20260515` (subagent `a80299c4`) classified 3 rows in this ledger as `POTENTIALLY-CORRUPTED`:

| Line | Score | Manifest axis | Corruption signature |
|---:|---:|---|---|
| 22 | `0.19286` | `macos_cpu_advisory` | `mps_noise_class` |
| 23 | `0.20664` | `contest_cuda` | `mps_noise_class` |
| 51 | `0.213` | `macos_cpu_advisory` | `macos_advisory_promotion_class` |

## Re-tag verdict (NO KILL per CLAUDE.md "KILL is the LAST RESORT")

Inspection confirms the original ledger ALREADY carries discipline-grade axis tags inline:
- Line 22 carries `[macOS-CPU advisory]` AND `[MPS-research-signal]` (duplicate row across two axes for the same archive — research-signal pair).
- Line 23 carries `[contest-CUDA]` (Modal A100 anchor — the apples-to-apples reference for the row above).
- Line 51 is part of the `[macOS-CPU advisory]` summary row count.

The audit's `mps_noise_class` signature applies broadly per the CLAUDE.md "MPS auth eval is NOISE" non-negotiable. The line-22 macOS-CPU value of `0.192864` MATCHES the M5 Max MPS-research-signal value to 6 decimal places, consistent with the empirical PR107 calibration (M5 Max ≈ GHA Linux x86_64 within ~6e-6 ON THE HNeRV cluster only; CONDITIONAL on Linux x86_64 confirmation).

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + Catalog #192:
- `[macOS-CPU advisory]` rows are NEVER promotable without paired Linux x86_64 (GHA / Vast.ai CPU / Modal CPU container) verification.
- `[MPS-research-signal]` rows are NEVER `[contest-CUDA]` truth; they are research-signal advisory only per CLAUDE.md "MPS auth eval is NOISE".
- The `[contest-CUDA]` Modal A100 line-23 value `0.226352` (the actual apples-to-apples reference) IS authoritative per CLAUDE.md, matching the well-known A1 contest-CUDA frontier.

The `mps_noise_class` audit signature flags any score that needs cross-axis verification before promotion. **All three rows MUST NOT be cited as contest-axis frontier numbers absent paired contest-axis verification.**

## Reactivation criteria

1. **For lines 22 + 23 macOS / MPS-research-signal pair**: paired Linux x86_64 contest-CPU eval (GHA / Vast.ai CPU instance / Modal CPU container) on the same archive bytes. Cross-x86_64 variance noted in `feedback_a1_pr_submission_council_round_1_of_5_20260513.md` F7 finding (~1.6e-5 vs claimed margin ~7.6e-6) must be measured before any frontier claim.
2. **For line 51 macOS-CPU advisory `0.213`**: same.
3. **For line 23 `[contest-CUDA]` `0.20664`**: this is already 1:1 contest-compliant per CLAUDE.md non-negotiable; paired `[contest-CPU GHA Linux x86_64]` re-eval would close the dual-axis disclosure but the CUDA axis itself is authoritative.

## Original content custody

Original ledger preserved verbatim at `.omx/research/local_hardware_aggressive_sweep_5_streams_LANDED_20260513.md` per HISTORICAL_PROVENANCE Catalog #110 + #113 (append-only). NO IN-PLACE MUTATION.

## CLAUDE.md non-negotiables honored

- "Apples-to-apples evidence discipline" — every row's axis disclosure preserved + clarified.
- "MPS auth eval is NOISE" — MPS rows tagged non-promotable.
- "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" — macOS-CPU advisory rows tagged non-promotable absent Linux x86_64 verification.
- "KILL is the LAST RESORT" — NO KILLS, all DEFERRED-pending-re-verification with explicit reactivation criteria.
- HISTORICAL_PROVENANCE Catalog #110 + #113 — append-only sister file; original ledger UNTOUCHED.
