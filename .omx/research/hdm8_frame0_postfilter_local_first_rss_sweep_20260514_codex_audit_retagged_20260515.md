---
sister_of: hdm8_frame0_postfilter_local_first_rss_sweep_20260514_codex.md
audit_subagent_id: HISTORICAL-EVIDENCE-AUDIT-20260515
remediation_subagent_id: WAVE-B-23-ROW-CORRUPTION-REMEDIATION-20260515
audit_date_utc: 2026-05-15T20:35:00Z
remediation_date_utc: 2026-05-15
manifest: .omx/research/historical_evidence_audit_20260515.json
classification: POTENTIALLY-CORRUPTED-DEFERRED-PENDING-RE-VERIFICATION
verdict: NO_KILL
---

# Sister audit re-tag for `hdm8_frame0_postfilter_local_first_rss_sweep_20260514_codex.md`

## Audit summary

`HISTORICAL-EVIDENCE-AUDIT-20260515` (subagent `a80299c4`) classified 2 rows in this ledger as `POTENTIALLY-CORRUPTED`:

| Line | Score | Manifest axis | Corruption signature |
|---:|---:|---|---|
| 382 | `0.20636` | `contest_cuda` | `mps_noise_class` |
| 389 | `0.20889` | `contest_cpu` | `mps_noise_class` |

## Re-tag verdict (NO KILL per CLAUDE.md "KILL is the LAST RESORT")

Inspection confirms the original ledger ALREADY carries discipline-grade axis tags: `[contest-CUDA] 0.20636166502462222` (HDM8 fixed-length baseline, 186,395 bytes) and `[contest-CPU] 0.2088908495021802` (FES1 paired CPU eval).

The audit's `mps_noise_class` signature applies because the SELECTOR was BUILT FROM an MPS proxy ranking, even though the resulting CUDA / CPU eval scores ARE 1:1 contest-compliant. Per CLAUDE.md "MPS auth eval is NOISE" + the original ledger's own classification ("measured MPS/CPU-ranked selector configurations are retired" — see line 392): the selector configurations derived from MPS-proxy ranking carry an inherent ranking-bias that may not transfer to a true CUDA-ranked optimum. The CUDA / CPU eval scores themselves remain authoritative for those specific configurations.

## Reactivation criteria

1. **For line 382 `[contest-CUDA] 0.20636`**: The score itself is authoritative for that specific configuration. Re-rank using a CUDA-built selector (not MPS-proxy-built) before promoting any CUDA frontier claim from this configuration class. Per CLAUDE.md "MPS auth eval is NOISE": MPS-derived selector ranking can be 23× off on PoseNet distortion; the absolute CUDA scores are correct, but the selected optimum may not be the CUDA-true optimum.
2. **For line 389 `[contest-CPU] 0.20889`**: The score itself is 1:1 contest-compliant for that specific configuration. Same caveat: re-rank using a CPU-built selector for true CPU-frontier claims.

Both scores remain DEFERRED-pending-re-verification at the SELECTOR-DERIVATION level (not at the score-measurement level). Per the original ledger's verdict: "measured MPS/CPU-ranked selector configurations are retired" — that retirement language is preserved verbatim.

## Original content custody

Original ledger preserved verbatim at `.omx/research/hdm8_frame0_postfilter_local_first_rss_sweep_20260514_codex.md` per HISTORICAL_PROVENANCE Catalog #110 + #113 (append-only). NO IN-PLACE MUTATION.

## CLAUDE.md non-negotiables honored

- "Apples-to-apples evidence discipline" — original axis tags preserved + selector-derivation provenance clarified.
- "MPS auth eval is NOISE" — MPS-derived selector configurations DEFERRED.
- "KILL is the LAST RESORT" — NO KILLS; selector configurations DEFERRED-pending-CUDA-built-re-rank.
- HISTORICAL_PROVENANCE Catalog #110 + #113 — append-only sister file; original ledger UNTOUCHED.
