# Retroactive sweep for Z8 Phase B `decompose_M_contest_per_subband` landing (2026-05-30)

Per Catalog #348 `check_new_gate_landing_includes_retroactive_sweep_evidence`.
4-field canonical contract.

## (1) Bug-class symptom signature

Pre-Phase-B, callers needing per-subband Mallat wavelet decomposition of a
per-pixel scorer-sensitivity map (`PerPixelSensitivityMap` from Phase A
`extract_M_pixel`) had two failure paths:

* **Failure A** (Phase C non-dyadic): callers passing a non-dyadic
  `level_shape` to `decompose_M_contest_per_level` correctly received
  `MallatDyadicMismatchError` per the Phase C canonical contract
  (commit `300702cdf` line 1586-1601); the error explicitly named Phase B
  (`decompose_M_contest_per_subband`) as the reactivation criterion.
* **Failure B** (no canonical per-subband helper): callers wanting actual
  per-subband {LL, LH, HL, HH} decomposition (per Daubechies wavelet
  multi-resolution analysis) had to roll their own — risking
  per-substrate-trainer cargo-cult variations of the canonical Mallat
  §7.7 separable construction.

Phase B closes BOTH failure paths by landing the canonical per-subband
helper that consumes the canonical Daubechies primitive at
`tac.symposium_impls.daubechies_wavelet_codec` and emits 4 canonical
`PerPixelSensitivityMap` subbands per the canonical Mallat 1989 §7.7
construction.

## (2) Pre-fix window

* **Phase A** landed commit `8a95c9cc5` (2026-05-30 earlier): `extract_M_pixel`
  + `PerPixelSensitivityMap` + broadcast adapter.
* **Phase D** landed commit `5a5311c00` (2026-05-30 earlier): wire-in
  consuming Phase A.
* **Phase C** landed commit `300702cdf` (2026-05-30 earlier today):
  `decompose_M_contest_per_level` + `MallatDyadicMismatchError` reactivation
  criterion.
* **Pre-fix window**: ~2 hours between Phase C landing and Phase B
  landing (THIS sweep). No empirical bug-class incidents in this window —
  no callers exercised the non-dyadic projection path yet because the
  Phase C surface is newer than Z8 M8 ScoreAwareLevelLoss consumers.
* **Pre-Phase-C window**: ~2 hours from Phase A → Phase C. No per-subband
  callers existed.

## (3) Historical KILL / DEFER / FALSIFY search

Searched canonical memory + research surfaces for KILL / DEFER / FALSIFY
verdicts related to per-subband Mallat wavelet decomposition at the
sensitivity-map surface:

* `grep -ri 'decompose_M_contest_per_subband\|per_subband.*sensitivity\|per_subband.*mallat\|per_subband.*wavelet' .omx/research/ ~/.claude/projects/-Users-adpena-Projects-pact/memory/*.md 2>/dev/null` — ZERO matches outside today's design context.
* `grep -ri 'wavelet.*subband.*sensitivity.*killed\|wavelet.*sensitivity.*falsified' .omx/research/ ~/.claude/projects/-Users-adpena-Projects-pact/memory/*.md 2>/dev/null` — ZERO matches.
* Sister Z8 `mallat_dwt_adapter` (commit `5f74a50a0`) carries the canonical
  M5 milestone PROCEED-unconditional verdict; no FALSIFICATION events
  there for the per-subband construction itself.
* Sister Z8 `binding_contract` `WaveletPartition` Protocol carries the
  canonical contract that Phase B's `SubbandSensitivityDecomposition`
  sister-extends to the sensitivity-map surface.

**Verdict**: zero historical KILL / DEFER / FALSIFY verdicts invalidated
by Phase B landing. The bug class Phase B closes is structural
(`MallatDyadicMismatchError` reactivation) not empirical.

## (4) Per-finding RE-EVAL-priority assignment

| Affected finding | RE-EVAL priority | Reason |
|---|---|---|
| Sister Z8 `mallat_dwt_adapter` M5 milestone | NONE | Z8 adapter operates on NHWC architecture surface; Phase B operates on `(N_pairs, H, W)` sensitivity-map surface. Sister-disjoint; no reactivation of M5. |
| Phase C `MallatDyadicMismatchError` reactivation criterion | CLOSED-VIA-Phase-B | Phase C's error message explicitly named Phase B as the reactivation; Phase B lands; reactivation criterion CLOSED. |
| Sister Z8 Phase E `ScoreAwareLevelLoss` (in-flight) | LOW | Phase E currently consumes Phase C; Phase B is available as future enhancement. Op-routable #1 surfaces this. |
| Future Z8 M9-M12 cascade | LOW | Operator-routable per op-routable #2. |
| Future cathedral consumer wrapper for per-subband sensitivity | LOW | Operator-routable per op-routable #4; deferred per iterate-not-force. |

No historical evidence invalidated. All operator-routable next-steps are
sister-extensions, not reactivations of killed work.

## Empirical anchor

This sweep memo IS the Catalog #348 evidence companion to the landing memo at:

`.omx/research/z8_phase_b_decompose_m_contest_per_subband_mallat_wavelet_hierarchy_landed_20260530.md`

per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
canonical 2-landing pattern (fix + self-protection) extended at the
META-meta surface by Catalog #348 retroactive-sweep discipline.

mission_predicted_contribution = `apparatus_maintenance`.
