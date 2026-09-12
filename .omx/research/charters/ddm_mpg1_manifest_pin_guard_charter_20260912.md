# ddm_mpg1 — land the class guard: `patch_inflate_pins` must rebind MANIFEST.sha256 (two-landing; charter, MAIN 2026-09-12; codex sol medium)

## The defect (MEASURED by sj1 pass 7, memo `.omx/research/ddm_sj1_t4_token_predistortion_pass7_20260912.md`)
The pin patcher that rewrites `ARCHIVE_SHA256` / `ARCHIVE_BYTES` in a candidate tree's `inflate.py` does not update the tree's `MANIFEST.sha256`
row for `inflate.py`, so a candidate tree carries a stale manifest and fails the seal's manifest validation. Pass 6's candidate tree carries the
defect; pass 7 fixed its own instance by hand. Find the producer (`git grep -n "def patch_inflate_pins"` — record file:line) and every caller.

## Deliverable (two landings per CLAUDE.md "Bugs must be permanently fixed AND self-protected against")
1. FIX: `patch_inflate_pins` (or its caller, whichever owns the tree write) rebinds the `MANIFEST.sha256` row(s) for every file it rewrites, using
   the repo's canonical manifest writer (grep for the tool that regenerates `MANIFEST.sha256` from OUTSIDE the tree; do not hand-roll a second
   manifest format); idempotent; refuses if the manifest lacks a row for the rewritten file.
2. GUARD: a test (≥ 6 cases: rewrite updates the row; no-op leaves the manifest byte-identical; missing row refuses; second file rewritten also
   rebinds; the manifest validator accepts the result; a stale manifest is DETECTED by the existing seal-inputs manifest validation) + a STRICT-eligible
   preflight check `check_inflate_pin_patch_rebinds_manifest` in `src/tac/preflight.py` scanning for pin-rewrite call sites without the manifest
   rebind (same-line `# MANIFEST_REBIND_OK:<rationale>` waiver, placeholder rejected), wired warn-only into `preflight_all()` with live count reported;
   strict-flip in the same landing only if live count is 0. Catalog number via `tools/claim_catalog_number.py claim`; CLAUDE.md catalog row per the
   pointer-backed catalog doc.
3. Memo `.omx/research/ddm_mpg1_manifest_pin_guard_20260912.md` (`# FORMALIZATION_PENDING:<rationale>`), tests green (`ruff` clean), two visible
   review passes per .py, serializer commits with post-edit shas, `[no-triality] [p0-ledger-ok]`, NO co-author trailer, NO AI attribution.
   A Git-object write denial (rc 17) is NOT a stop: keep files in the working tree, leave the bundle; MAIN lands; commit LAST.

## Boundaries
Never edit `upstream/`, the PR tree, sealed trees, contract code (`candidate_seal.py`, `decode_wall_clock.py`), or any candidate tree under
`/Volumes/` (pass 7/8's trees are live custody — read-only; do NOT "fix" pass 6's tree). $0, no launches. Checkpoint as `ddm_mpg1`.

## OPTIMAL FORM
`# OPTIMAL_FORM_NA: apparatus fix + guard on an existing tool; reference form is the repo's canonical manifest writer and the existing seal-inputs manifest validator, both cited by path in the memo.`

Final message: file:line of the producer, the diff summary, test count, catalog number, live count and strictness, commit rc, and
`composition S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600] (move 49)` unchanged.

<!-- # FORMALIZATION_PENDING: charter for an apparatus guard; no measured row -->
