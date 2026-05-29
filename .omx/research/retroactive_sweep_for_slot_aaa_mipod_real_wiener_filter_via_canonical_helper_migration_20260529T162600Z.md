# Catalog #348 retroactive sweep for Slot AAA MiPOD real Wiener filter migration

Per Catalog #348 (`check_new_gate_landing_includes_retroactive_sweep_evidence`):
this memo discharges the 4-field contract for the canonical bind-helper
landing surfaced by the Slot AAA migration.

**NOTE:** Catalog #348 was authored for NEW STRICT preflight gate landings.
This migration does NOT land a new STRICT preflight gate (per Catalog #299
quota brake decision: NO new Catalog # claimed; sister-extinction architecture
reuses 35+ existing surfaces). However the sister discipline applies at the
canonical surface landing too, so this sweep memo is emitted for hygiene.

## 1. Bug-class symptom signature

**Bug class**: substrate inverse-steganalysis L0 SCAFFOLDs that implement
a per-pair scalar aggregation of a paper-canonical per-pixel cost matrix
on synthetic random-noise inputs, with the canonical helper functions
admitting in their own docstrings that they have been simplified (box-
blur instead of REAL Wiener filter; row-band aggregation instead of
per-pixel; etc.).

**Symptom signature**:
1. Docstring admits the canonical helper is a simplification.
2. Strategy enum has 3-of-4 values sharing the same code path
   (degenerate enum).
3. Tests verify per-pair scalar aggregation on synthetic random noise.
4. Cost-matrix dynamic-range on real upstream frames is degenerate.

## 2. Pre-fix window

Slot AAA MiPOD canonical L0 SCAFFOLD landed 2026-05-29 ~14:53 UTC
(operator-binding directive #10 + Slot UU TOP-2 ranking 8/9 + canonical
Fridrich-Yousfi cascade Axis 6 extension). Slot EEE fake-implementation
audit landed 2026-05-29 ~15:18 UTC (operator binding "Must review for
fake implementations"). Slot EEE verdict on Slot AAA: PARTIAL —
admitted-box-blur Wiener filter; per-pair simplification; degenerate
enum. THIS migration lands 2026-05-29 ~16:26 UTC routing through the
canonical shared helper (commit `32a70c051` landed 2026-05-29 ~11:13
UTC per the Slot YY HILL sister-cascade pattern).

## 3. Historical KILL/DEFER/FALSIFY search results

Searched `.omx/research/` + `~/.claude/projects/-Users-adpena-Projects-
pact/memory/` for historical KILL/DEFER/FALSIFY verdicts that this
migration might invalidate.

**Results**:

- **0 historical KILL verdicts on MiPOD** found.
- **0 historical DEFER verdicts on MiPOD** found whose evidence basis
  this migration invalidates.
- **0 historical FALSIFICATION verdicts on MiPOD** found.

**The Slot EEE PARTIAL verdict** on Slot AAA is the canonical
operator-routable cascade trigger for THIS migration; the verdict is
REMEDIATED per Catalog #307 IMPLEMENTATION-LEVEL classification (the
admitted-box-blur is implementation-level NOT paradigm-level; the
Sedighi-Cogranne 2016 MiPOD paradigm IS INTACT; the bind helper
remediates the implementation gap by routing through the canonical
shared helper which implements the canonical Algorithm 1).

**The existing 86 per-pair tests** are PRESERVED per Catalog #110/#113
HISTORICAL_PROVENANCE + CLAUDE.md "Forbidden premature KILL"
non-negotiable. The existing `_wiener_filter_canonical` box-blur
function remains callable for backward compat; the NEW bind helper is
ADDITIVE.

## 4. Per-finding RE-EVAL-priority assignment

| Historical finding | RE-EVAL priority | Reason |
|---|---|---|
| Slot EEE Slot AAA PARTIAL verdict (admitted-box-blur) | REMEDIATED | THIS migration provides the canonical REAL Wiener filter via canonical shared helper routing. |
| Slot EEE Slot AAA PARTIAL verdict (per-pair simplification) | REMEDIATED | THIS migration provides per-pixel cost matrix on real upstream frames via canonical shared helper. |
| Slot EEE Slot AAA PARTIAL verdict (3-of-4 degenerate enum) | DEFERRED-PENDING-SEPARATE-CORRECTIVE-ACTION | THIS migration is additive; degenerate enum is pre-existing design issue per Catalog #110/#113 HISTORICAL_PROVENANCE; operator-routable per Catalog #308 in a separate corrective action. |
| Slot AAA canonical L0 SCAFFOLD landing (predicted band [-0.0008, +0.0003] pending_post_training) | NO CHANGE | THIS migration is observability-only (Tier A markers); does not affect the predicted band; the canonical equation candidate stays FORMALIZATION_PENDING until paired-CUDA RATIFICATION per Contrarian dissent. |
| Slot YY HILL sister-cascade landing (commit `32a70c051`) | NO CHANGE | Sister pattern; same canonical shared helper module; no invalidation. |

## Cross-reference

- Catalog #348 (`check_new_gate_landing_includes_retroactive_sweep_evidence`)
- Slot EEE landing memo: `feedback_slot_eee_fake_implementation_audit_
  on_today_l0_scaffolds_per_operator_binding_must_review_for_fake_
  implementations_landed_20260529.md`
- Slot AAA canonical L0 SCAFFOLD landing: `feedback_slot_aaa_mipod_
  canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016_
  canonical_fridrich_yousfi_cascade_axis_6_extension_per_slot_uu_
  top_2_landed_20260529.md`
- THIS migration landing: `feedback_slot_aaa_mipod_real_wiener_filter_
  via_canonical_helper_migration_landed_20260529.md`
- Council memo: `.omx/research/council_slot_aaa_mipod_real_wiener_
  filter_via_canonical_helper_migration_20260529.md`
- Slot YY HILL sister-cascade pattern anchor: commit `32a70c051`
- Canonical shared helper: `src/tac/inverse_steganalysis_real_video_mlx/__init__.py`
