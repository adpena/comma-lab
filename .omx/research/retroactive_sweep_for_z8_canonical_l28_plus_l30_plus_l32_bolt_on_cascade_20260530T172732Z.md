<!-- SPDX-License-Identifier: MIT -->
# Retroactive sweep for Z8 canonical L28+L32 PR-family quick-wins bolt-on cascade — 2026-05-30T17:27:32Z

Per Catalog #348 (`check_new_gate_landing_includes_retroactive_sweep_evidence`).
NOT a new STRICT gate landing — this is a canonical PR-family bolt-on cascade
on the existing Z8 substrate. Sweep applies the 4-field contract per Catalog
#348 standard.

## Bug-class symptom signature

The canonical PR-family L28 + L32 bolt-ons exist in CLAUDE.md HNeRV parity
discipline L28 + L32 anchors but were NOT yet bound to the Z8 substrate
(operator's `[[pr-or-greater-parity-synergy-binding-integration-not-hnerv-specific-meta-class-lesson-correction]]`
standing directive empirically anchored 2026-05-30 — binding-depth discipline
applies the canonical L1-L32 ingredients SIMULTANEOUSLY).

Symptom: canonical PR-family marginal score improvements (L28 ~-0.0001 to
-0.0005; L32 ~5-10% archive bytes saved) NOT yet realized at the Z8
substrate; the Z8 substrate predicted ΔS band reflects only L1-L7 + L29
ingredients bound at M11 landing.

## Pre-fix window

- 2026-05-29: operator BINDING META-class correction "the parity lessons are
  not hnerv parity, they are PR parity or greater" (commit landed as
  standing directive in CLAUDE.md context).
- 2026-05-30 16:08:04 UTC: Z8 M11 L1 smoke landed at commit `2f8570755`
  with canonical PR-family L1-L7 + L29 ingredients but WITHOUT L28 + L32.
  Baseline anchor: `evaluator_final_score=43.62` /
  `archive_bytes_total=92408` / `evaluator_compressed_size_bytes=92516`.
- 2026-05-30 16:56:39 UTC: this lane spawned (operator-routed Yousfi-cascade
  TOP-1 post-M11) — see `feedback_pr_or_greater_parity_synergy_binding_integration_not_hnerv_specific_meta_class_lesson_correction_20260529.md`
  + `feedback_complexity_loc_unconstrained_push_boundaries_within_contest_compliance_standing_directive_20260530.md`.

## Historical KILL/DEFER/FALSIFY search results

Per Catalog #348 sweep contract — search for historical KILL/DEFER/FALSIFY
verdicts that may be invalidated by this canonical PR-family binding:

```bash
$ grep -rli "L28\|L30\|L32" .omx/research/*killed*.md .omx/research/*deferred*.md .omx/research/*falsification*.md 2>/dev/null | head -10
(no results)
```

**No historical KILL/DEFER/FALSIFY verdicts** reference L28 / L30 / L32
canonical equations. The canonical PR-family bolt-on cascade is a NEW
binding (not a reactivation of a previously-killed surface).

Sister L1-L32 canonical equations registered 2026-05-28 per the CLAUDE.md
HNeRV parity expansion (commit history: `feedback_canonical_equations_and_models_registry_formalization_landed_20260519.md`
+ subsequent L14-L32 backfill); pre-2026-05-28 there were no L28/L30/L32
canonical equations to KILL/DEFER/FALSIFY.

**Sister cross-substrate audit**: the L28 canonical helper `write_rgb_pair_to_raw`
extension is backward-compatible (default OFF kwarg) so sister substrates
(NSCS06 v8 / DP1 / Slot GGG / Cascade C' / etc.) are NOT impacted. Any
sister substrate that wants L28 opt-in must explicitly pass
`apply_pr98_l28_channel_postprocess=True` to the canonical helper.

## Per-finding RE-EVAL priority assignment

Per Catalog #348 sweep contract — for each finding affected by this canonical
PR-family binding, assign RE-EVAL priority:

| Finding | Affected? | RE-EVAL priority | Rationale |
|---|---|---|---|
| Z8 M11 L1 macOS-CPU end-to-end smoke (baseline 43.62) | YES (baseline reference) | LOW | Baseline is canonical reference; THIS landing extends with L28+L32 marginal; baseline not invalidated, just superseded by new post-L28-L32 anchor |
| Z8 M10 inflate consumes real trained weights per Catalog #369 | YES (inflate site receives L28 wire-in) | LOW | L28 is opt-in kwarg added to canonical helper; M10 inflate.py opts in explicitly; no Catalog #369 contract violation (real-trained-weight consumption preserved; the postprocess adds canonical PR98 channel offsets at uint8-cast stage) |
| Z8 M9 canonical quadruple training (M9 archive emit uses brotli q=9) | YES (q=9 → q=11) | LOW | Brotli q=11 is a deterministic invariant-preserving compression upgrade per Brotli RFC 7932; archive decompresses to byte-identical content; no M9 contract violation |
| Canonical equations L28 / L30 / L32 registry status | YES (NEW empirical anchor for L28 + L32; L30 DEFERRED) | MEDIUM | Append canonical equation EmpiricalAnchor for L28 + L32 per Catalog #344; register canonical anti-pattern for L30 float32 cargo-cult per Catalog #344 sister discipline; auto-recalibration per Catalog #371 fires when 3+ anchors accumulate |
| Sister substrates' inflate runtime usage of `write_rgb_pair_to_raw` | NO (backward-compat via default-OFF kwarg) | NONE | Verified by dedicated regression test `test_l28_postprocess_default_off_preserves_backward_compatibility` (byte-identical output) |
| Canonical PR111 + PR110 + PR107 + PR101 + PR98 + PR103 SILVER + PR100 family | YES (canonical PR-or-greater parity binding-depth realization) | LOW | The canonical PR-family bolt-ons are now bound into the Z8 class-shift substrate per operator's binding-depth standing directive; PR-family canonical references unchanged but now applied as canonical PR-or-greater parity binding at the class-shift surface |
| Cathedral autopilot ranker | NO (Tier A observability-only canonical equation anchors; no score-mutation routing) | NONE | Canonical equation anchors flow through `update_equation_with_empirical_anchor` per Catalog #344; sister cathedral consumer `canonical_equation_lookup_consumer` (Catalog #344 + #335 auto-discovery) surfaces the new L28+L32 anchors at Tier A observability per Catalog #341 markers — no score-mutation routing |

## Conclusion

ZERO historical KILL/DEFER/FALSIFY verdicts invalidated by this canonical
PR-family bolt-on cascade. The L30 deferral is NEW classification per
Catalog #290 FORK_BECAUSE_PRINCIPLED_MISMATCH with substantive reactivation
criteria (NOT a kill verdict per CLAUDE.md "Forbidden premature KILL").

Sister cross-substrate impact: NONE (L28 opt-in default OFF preserves
backward compatibility per dedicated regression test).

Canonical apparatus mutations land in same commit batch per CLAUDE.md
"Strict-flip atomicity rule" + Catalog #344 canonical-2-landing pattern.
