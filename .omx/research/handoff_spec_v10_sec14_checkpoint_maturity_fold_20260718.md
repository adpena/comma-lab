# HANDOFF — SPEC_v10 §14 fold: the `_dev`/`_prod` checkpoint-maturity axis (2026-07-18)

**Why a handoff note instead of an on-branch commit:** the SoT branch
`claude/p0_521_spec_v10_capstone_20260717` is CHECKED OUT in another worktree (`git branch -a`
shows the `+` marker), so committing to it from this worktree would desync that worktree's
index. Per the task contract, the fold text is delivered here for main to land on the branch
as a dated sub-fold of §14 (do NOT create a new spec doc — §14.1 is the naming SSoT this
extends).

**Paste-ready fold text (append after §14.5):**

---

### §14.6 CHECKPOINT-MATURITY AXIS — `_dev` / `_prod` (operator 2026-07-18; extends §14.1)

Maturity is a SECOND axis, ORTHOGONAL to the vehicle axis locked in §14.1. The vehicle name
says WHICH carrier (v9c2 / v9c3 / v10); the maturity suffix says HOW PROMOTABLE its scores
are. Composed name form: `<vehicle>_<dev|prod>[...]` — e.g. `v9c3_dev`, `v10_prod_bank_<date>`.

- **`_dev`** — free-iteration lane (v9c3 iterations live here). Kept ALONGSIDE prod; a dev
  bank NEVER overwrites a prod bank. Dev advisory AND exact rows are NON-PROMOTABLE to
  `.omx/state/canonical_frontier_pointer.json` BY DEFAULT — banked + labeled, never
  pointer-moving, until explicit operator dev→prod promotion.
- **`_prod`** — the capstone lane (v10's cold-start line when fired). Only its byte-closed
  exact rows (operator-GO) are pointer-eligible. Prod bank dirs are IMMUTABLE (write a new
  dated bank dir; never overwrite in place).
- **Untagged vehicle-shaped names** (e.g. `v9c2_defensive_bank_20260718`) default to the SAFE
  side: treated as dev / non-promotable. The live run `levelset_n600_witness_20260717T113932Z`
  (v9c2) is implicitly dev and is NOT renamed — the convention binds NEW run/bank names.
- **Legacy pre-convention names** (no vehicle token; e.g. the standing frontier lanes) are
  grandfathered at the pointer surface only (refusing them would clobber the live anchors).

APPARATUS (landed on main 2026-07-18): `src/tac/checkpoint_maturity.py` (parse +
`is_pointer_promotable` strict-prod-only + `pointer_promotion_verdict` + `bank_dir_name` +
`assert_bank_dir_writable` prod-immutability guard) and the fail-closed pointer-promotion
gate inside `tac.canonical_frontier_pointer.refresh_canonical_frontier_from_local_state`
(refusals recorded in `refresh_provenance.checkpoint_maturity_refusals`; prior anchor kept;
covers CLI, upstream-refresh, and dispatch auto-refresh routes). 28 dedicated tests;
preview-refresh MEASURED inert on the standing pointer state. Consequence for §14.5(b): a
v9c3 (`v9c3_dev`) iteration can byte-close and exact-eval freely — its rows bank and label
but structurally cannot move the pointer; the v10 line (`v10_prod`) is the promotion lane.

---

Pointer 0.19108 UNMOVED (apparatus). Memory:
`checkpoint_maturity_dev_prod_axis_orthogonal_to_vehicle_20260718.md`. DAG: FEED-maturity
(2026-07-18) in `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
