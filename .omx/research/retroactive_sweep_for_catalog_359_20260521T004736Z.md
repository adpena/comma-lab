<!-- SPDX-License-Identifier: MIT -->
# Retroactive verdict-taint sweep for Catalog #359 — 2026-05-21T00:47:36Z

Per Catalog #348 (event-driven retroactive verdict-taint sweep) 4-field contract.

## 1. Bug-class symptom signature

Canonical equation #26 (`procedural_codebook_from_seed_compression_savings_v1`) misapplied to residual-correction stacking-extension contexts via `update_equation_with_empirical_anchor` calls whose `inputs.in_domain_context` strings do NOT match either `_INCLUDED_CONTEXTS` (intermediate-transform codebook positions) NOR `_EXCLUDED_CONTEXTS` (direct DWT detail / wavelet coefficient byte substitution). The equation's existing `validate_context_is_in_domain` validator returns `False` for unknown contexts (line 180 of `procedural_codebook_savings.py`) — "not refused, not endorsed" — and the misapplication slips past silently. Anchors land at residual_zscore far outside 2σ threshold (38.8 for pair #1; 101.18 for pair #2).

## 2. Pre-fix window

- Canonical equation #26 registered: 2026-05-20T22:37:45Z (commit `5c1af7ba6`)
- Catalog #359 strict gate landing: 2026-05-21T00:47:36Z
- Pre-fix window: 2 hours 10 minutes (3 anchor_appended events landed in this window)

## 3. Historical KILL / DEFER / FALSIFY search results

Searched the pre-fix window for KILL / DEFER / FALSIFY verdicts that depended on canonical equation #26 misapplication:

| Anchor | Window status | Verdict effect | Reassessment |
|---|---|---|---|
| `first_empirical_anchor_wave_3_dwt_smoke_20260520T232240Z` (commit `f25f8cc1b`) | PRE-CUTOFF preserved | Yesterday's DWT-detail-subband KL=1.638 nats anchor; **NOT residual-hybrid** — direct substitution context that the `_EXCLUDED_CONTEXTS` already covers; CORRECT empirical falsification | KEEP — no reassessment needed |
| `second_empirical_anchor_wave_3_magic_codec_pair_1_dwt_residual_smoke_20260520T234707Z` (commit `debbc5833`) | PRE-CUTOFF preserved | Pair #1 zscore=38.8 falsification; **IS residual-hybrid misapplication** | **RE-EVAL-priority MEDIUM**: per the Catalog #359 landing's §1 + §5 (Issue A) verdict, the empirical receipt (+0.036805 ΔS) is HARD-EARNED data on the residual-hybrid class; the FALSIFIED verdict against canonical equation #26 prediction (-0.00200) was a MISAPPLICATION-level falsification not a paradigm-level refutation. The residual-correction stacking paradigm remains DEFERRED-PENDING-NEW-SISTER-EQUATION per CLAUDE.md "Forbidden premature KILL"; the next operator-routable subagent should land sister equation `procedural_predictor_plus_residual_correction_savings_v1`. Anchor preserved per Catalog #110/#113 APPEND-ONLY |
| `third_empirical_anchor_wave_3_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_20260521T002120Z` (commit `a986efa99` + sister codex) | PRE-CUTOFF preserved | Pair #2 zscore=101.18 falsification; **IS residual-hybrid misapplication** | **RE-EVAL-priority MEDIUM**: same reasoning as pair #1; the +0.054055 ΔS empirical receipt IS HARD-EARNED data on the (sparse_packet_ir SRL1, fec6 null-byte) residual-hybrid surface. The cascade-PIVOT verdict `PAIR_2_FALSIFIED_CASCADE_FURTHER_NARROWS_PIVOT_TO_PAIR_4_OR_DP1_ONLY` from the landing memo remains correct (pair #4 + DP1-only are the operator-routable next steps); the canonical equation misapplication does NOT change the cascade pivot recommendation |

NO KILL verdicts depend on the misapplication; the falsifications were already correctly classified per Catalog #307 as IMPLEMENTATION-level (not paradigm-level) in the original landing memos.

## 4. Per-finding RE-EVAL-priority assignment

| Finding | RE-EVAL priority | Action |
|---|---|---|
| Pair #1 + Pair #2 historical anchors | MEDIUM | No action required at the canonical equation #26 surface — anchors preserved per APPEND-ONLY discipline; the Catalog #359 gate prevents FUTURE misapplications; pair #1 + pair #2 cascade pivots (to pair #4 / DP1-only) recommended in the original landing memos remain correct |
| Sister canonical equation registration | LOW | DEFERRED-PENDING-RESEARCH per the adversarial review §13 op-routable #3; not blocking |
| Pair #1 + Pair #2 smoke script `update_equation_with_empirical_anchor` call sites | LOW | The smoke scripts wrote anchors directly without calling `validate_context_is_in_domain`; Catalog #359 catches the misapplication at the persisted-artifact surface AFTER append; future smokes should call the canonical helper `refuse_residual_hybrid_context_misapplication` BEFORE append per the runtime per-call surface; sister design candidate |
| Stacking analysis memo §7 4-pair matrix predictions | LOW | Predictions inherit the misapplied equation #26 prediction; per the adversarial review §5 Issue E this is a derivative bug, not a new bug class; sister design candidate to rewrite the matrix with the sister equation once landed |

**End of retroactive sweep memo.**
