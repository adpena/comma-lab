# ddm_mpg1 — archive-pin patches now rebind the receiver manifest

<!-- # FORMALIZATION_PENDING: apparatus-only filesystem custody invariant; no witness dynamics or new score equation was measured -->

## Outcome

The defect is fixed and self-protected. Every production `patch_inflate_pins*` producer now refuses a tree whose `MANIFEST.sha256` cannot name every rewritten file, updates the `inflate.py` row through one canonical writer, validates the completed listing with the existing seal-inputs validator, and restores the original `inflate.py` plus manifest if the transaction fails. Repeating a completed patch leaves `MANIFEST.sha256` byte-identical.

The FIX landing is commit `3e4645313`. The GUARD landing carries Catalog #420 and is the second landing.

## Producer and caller census

- `experiments/ddm_sj1_joint_admission.py:577` — current pattern-based producer. Callers: the local stage at line 721, `experiments/ddm_fe1_admit_and_build.py:527`, and `experiments/ddm_rp1_build.py:164`.
- `experiments/ddm_cl2_hpac_prior_capacity_ladder.py:249` — historical literal-bound producer; local caller at line 499.
- `experiments/ddm_cl3_rc1_rung_price.py:260` — historical live-literal producer; local caller at line 444.

The live Catalog #420 census is **0 violations / 7,085 production Python files**, with 6 signature-bearing candidates parsed, 3 producers, and 5 call sites. Because the live count is zero, `preflight_all()` is strict in this landing.

## Fix form

`src/tac/receiver_manifest.py:33` owns the canonical manifest byte writer extracted from `experiments/ddm_ntb2_public.py::regenerate_manifest`; the older public-proof entry point delegates to it. The helper preserves the input tree's archive-row policy, so both valid forms accepted by the contract remain valid: manifests that list `archive.zip` and manifests that deliberately exclude it.

`src/tac/receiver_manifest.py:59` checks target-row coverage before a producer mutates a file. `src/tac/receiver_manifest.py:77` refuses unrelated stale or omitted rows, renders the full replacement outside the runtime tree, atomically replaces the manifest, and calls `tac.decode_wall_clock.validate_receiver_manifest` (`src/tac/decode_wall_clock.py:170`). This is the existing seal-inputs validation used by `tac.candidate_seal` first-measurement identity checks.

All three producers call the helper themselves. Their five callers therefore cannot forget a separate follow-up write. The three stage censuses now expect exactly `archive.zip`, `inflate.py`, and `MANIFEST.sha256` to move.

## Guard and tests

Catalog #420 is `src/tac/preflight.py:95333`. It scans production Python definitions and call sites for `patch_inflate_pins` / `patch_inflate_pins_live`. A producer is protected only when its body calls `rebind_receiver_manifest` imported specifically from `tac.receiver_manifest`; an unresolved call site is a violation. Same-line `# MANIFEST_REBIND_OK:<rationale>` waivers are supported, and placeholder rationales are rejected.

The focused verification is **111 tests passed**. The charter's six required behaviors are explicit: rewrite updates the row; no-op preserves manifest bytes; missing row refuses before mutation; two rewritten files rebind both rows; the existing validator accepts the result; and that validator detects a stale pre-fix tree. Additional controls reject unrelated stale-row laundering, prove the old `ntb2` entry point delegates to the shared writer, exercise strict/warn guard modes, and prove live count zero. Ruff is clean on every changed Python file.

## RECALL EVIDENCE

Sources searched and exact query surfaces:

- Full research corpus: `rg -n "MANIFEST.sha256" .omx/research docs tools src experiments`, plus focused reads of `ddm_sj1_t4_token_predistortion_pass7_20260912.md`, `ddm_ntb2_public.py`, `ddm_pc3_public.py`, `ddm_hpr1_public.py`, and the pr18 manifest-validation tests.
- Producer/caller corpus: `git grep -n "def patch_inflate_pins"` and `git grep -n "patch_inflate_pins"`.
- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json` filtered for manifest/pin/archive terms. No equation changed this apparatus plan.
- Research index and DAG: content searches for `manifest`, `pin patch`, and `archive pin` over `.omx/research/CANONICAL_RESEARCH_INDEX*` and `.omx/research/sub015_DAG_*`.
- Live authority: `.omx/state/main_hot_state.md`, including move 49 and the named MANIFEST guard debt.

Beyond the charter seeds, recall found two historical sibling producers (`cl2`, `cl3`), two external callers (`fe1`, `rp1`), the `ntb2` canonical byte-generation rule, the pr18 contract's optional archive-row semantics, and the existing seal-inputs validation call. That changed the plan from an sj1 point-fix into a three-producer class fix, and from assuming one 49-row shape into preserving whichever valid archive-row policy the input manifest already uses.

## Boundaries

No scorer, Modal call, paid dispatch, candidate build, or long job ran. No file under `upstream/`, any PR tree, any sealed tree, or `/Volumes/` was edited. The only volume access was a read-only comparison of a promoted tree's listing policy. `candidate_seal.py` and `decode_wall_clock.py` were not edited. No payload was materialized or discarded.

This apparatus landing does not move or claim a score. **composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600] (move 49) unchanged.**

## LIVE-HYPOTHESES

- None inside this charter. The implementation and structural guard are both exercised at live count zero.

## DEAD-ENDS

- Hand-editing pass 6 or any retained candidate tree is closed: those trees are custody evidence and were read-only by charter.
- A caller-by-caller manifest write is closed: it recreates the exact omission class at the next caller; the producer owns the transaction.
- One hard-coded 49-row writer is closed: valid receiver manifests may include or exclude `archive.zip`, and the rebind preserves the input policy.
- Treating the existing seal validator as prevention is closed: it detects the stale tree only after materialization; Catalog #420 prevents the producer class.
