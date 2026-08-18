# ddm_ft1 — fire_modal_auth_eval.py determinization debt (#1105, two-landing)

**Origin:** task #1105 (pq2 blocker + rv2 FO-2). The canonical Modal fire path
tools/fire_modal_auth_eval.py (65e15db4e9 lineage) is the ONLY allowed fire mechanism
(memory hand_assembled_dispatch_is_the_error_factory_20260817) but has two named defects:
(a) NO CPU-axis mode — #1111's packet freeze requires an exact-byte contest-CPU authority row
(~$0.15) and pq1's sealed CPU fire-order cannot execute through the canonical tool;
(b) modal_endpoint_close discards str-typed artifacts (the rv2 FO-2 finding) — receipts lost.

**RECALL FIRST (m44):** rv2 memo FO-2 · pq1/pq2 memos (the sealed CPU fire-order + blocker) ·
the fire-ladder cures (no-pipe · axis-pairing waiver · re-pinned runtime · entry-point smoke) —
CPU mode must preserve ALL of them · #1054 receipts (the one prior CPU-axis row: contest-CPU on
frontier bytes, decode 831.5s, fired how?) — inherit that path's working pieces · t1h r1-r4
failure ladder (the paired-axis gate semantics).

**Standing laws:** NO Modal fires from this arm (build + dry-run only; MAIN fires) · NO Metal ·
NO n600 scorer · serializer commits w/ POST-EDIT sha · .py = 2 review-tracker passes, never
REVIEW_GATE_OVERRIDE on .py · upstream/ READ-ONLY · no AI attribution · .venv/bin/python.

## The two landings (each fix + self-protection, per the two-landing law)
L1 — CPU-axis mode: --axis cpu (and explicit dual mode) on the canonical tool, preserving the
  paired-axis gate semantics (single-axis still requires --single-axis-waiver-reason; CPU-axis
  single fires get the same waiver discipline), staged-tree custody identical to the T4 path,
  Linux-x86_64 contest-CPU worker route, [contest-CPU] evidence tagging fail-closed. Guard: a
  test that REFUSES an axis-untagged result row + a dry-run receipt proving the CPU route
  composes the same FIRE_MANIFEST custody fields as T4.
L2 — modal_endpoint_close str-artifact fix: persist ALL artifact types byte-for-byte (ALWAYS
  KEEP THE PAYLOAD applies to receipts); regression test with a str-typed artifact fixture +
  both-direction control (drops before fix, retained after).
Deliverable: memo + tests green + a DRY-RUN receipt (no paid fire) demonstrating the CPU-mode
command pq1's sealed fire-order would execute; MAIN fires the real row at packet freeze.

## OPTIMAL FORM
Reference: the live T4 fire path at its proven form (rr4/fx1 fires) extended, never forked —
a second hand-rolled CPU dispatcher is the error-factory anti-pattern and FORBIDDEN. SCOPE
reduction legal (dual-mode may land as sequential single-axis calls). MECHANISM reduction
forbidden: a CPU mode that skips the receiver entry-point smoke or the archive-sha stage
verification is a TOY. Provenance pins: tools/fire_modal_auth_eval.py @ HEAD · 65e15db4e9 ·
#1054 CPU-row receipts · pq1 sealed CPU fire-order (dec5402577 bundle).

## PRIOR-LAW PREDICTION
The #1054 CPU row proves the worker side exists; the debt is tool-surface plumbing. Predict:
L1+L2 land in one arm-session with zero worker changes; falsifier: if the CPU route requires
worker-image changes, STOP and report the delta (MAIN adjudicates before any image rebuild).
