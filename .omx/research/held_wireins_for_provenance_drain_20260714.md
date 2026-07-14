# HELD wire-ins — owed to the DSL + canonical_equations legs once the provenance arm drains (2026-07-14)

The live arm `provenance_canonicalize_fix_all_fakes` EXCLUSIVELY owns `src/tac/witness_dsl/` and
`src/tac/canonical_equations/` this session (it is restructuring them for the #501 fake-audit). Two reviewed
findings this turn produced DSL-lever + canonical-equation legs that are **HELD** — deferred by ownership
coordination, NOT silently dropped. This ledger is the queryable record so the triality legs are wired the
moment provenance drains (main picks these up, coordinating serially). Trajectory legs already landed in
`sub015_DAG_*` (FEED-bregman-review, FEED-fable-AMC).

## 1. Bregman squared-Hessian dual-metric correction (from bregman_v9_all_surfaces; #500/#501/#504)
- **canonical equation (owed):** the dual metric is `‖Δη‖² = Δθᵀ H² Δθ` (squared-Hessian), NOT the ordinary
  Hessian metric `Δθᵀ H Δθ = Δηᵀ H⁻¹ Δη`. Fisher-natural dual REQUIRES the typed `H⁻¹` solve. Anchor: the
  measured 600/600 false-equality (err ~9e-13). Held module: `src/tac/information_geometry/bregman_v9_surfaces.py`
  + `src/tac/canonical_equations/bregman_v9_surfaces_20260714.py` (untracked, in working tree, held).
- **DSL leg (owed):** `argmax_native_vjp_fidelity_v1` must NOT be changed to a no-solve `‖Δη‖` shortcut
  (name-preserving fake); if a dual-metric lever is adopted it carries the `H⁻¹` solve.
- Memo: `.omx/research/codex_premise_falsification_bregman_dual_euclidean_20260714_codex.md`. Memory:
  `dual_metric_no_solve_is_squared_hessian_not_fisher_natural_20260714`.

## 2. Fable AMC per-row tiered code bit-allocation (from Apple warm-start; #406/#336)
- **canonical equation (owed):** `amc_perrow_tiered_code_bitalloc_v1` — law: pair-local code rows ⇒ per-pair
  d_seg composition is EXACTLY ADDITIVE ⇒ measured-response (per-pair-KKT) allocation DOMINATES proxy-saliency
  tiers. Anchors: the Fable measurement artifact + the 07-13 n600 custody (byte-identical 6/6 sha). Advisory
  axis only until an exact contest-CPU row exists.
- **DSL leg (owed, HELD spec):** `TieredCodeQATLever` (train-time; flags born-through-DSL per never-invent-flags).
  Tool `tools/apply_amc_saliency_tiered_bitalloc_witness.py` is committed (7c99b52b75); DSL Lever is N/A for the
  measurement tool (FEED-07l precedent), the QAT lever is the held train-time surface.
- Full spec: `.omx/research/fable_amc_saliency_codex.md` §8. Pays only at a competitive witness checkpoint.

**Trigger:** when `codex_status.py` shows `provenance_canonicalize_fix_all_fakes` no longer RUNNING, wire both
canonical equations + the held module/lever through the serializer (coordinating with whatever provenance
landed in those dirs), then delete this ledger.
