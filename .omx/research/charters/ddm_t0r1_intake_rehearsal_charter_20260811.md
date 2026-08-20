# ddm_t0r1 — T0 INTAKE DRESS REHEARSAL on pass-3 objects (scorer-free; the chain-never-ran-end-to-end cure)

## Mission (pn1/Carmack law: the chain that never ran end-to-end fails on terminal day)

Rehearse the ENTIRE T0 intake chain of the reseal skeleton
(`.omx/research/ddm_js1_reseal_skeleton_20260811.md`) against the ps135 pass_03 objects — same
SHAPES as the terminal objects — clearly labeled REHEARSAL throughout, in a separate store
(`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/t0_rehearsal_pass03/`). NEVER touch
the scorer, the live solve (pid 26406, its dirs are READ-ONLY to you), or any terminal-labeled
store.

## Ordered work

1. **ROLE CENSUS (the real payoff):** enumerate hr2's 7 typed terminal roles
   (src/tac/witness_dsl/hr1_prestage.py binder) and attempt to bind EACH from the pass-3
   artifacts (/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/leg_a/passes/pass_03/ + starts/
   + state.json + the gen3_resume safe_run receipt) + the ps135 store at large. Output the
   DEMAND LIST: which roles bind cleanly · which exist under a different name/shape (adapter
   needed — spec it) · which have NO producer in the solve's emission at all (⇒ the terminal
   run will NOT emit them either — name the missing producer so MAIN can decide whether to add
   emission BEFORE terminal or drop the role). Suspect: the 'sensitivity map' role (ps135
   stage-C deliverable — verify whether ANY pass emits it) and 'convergence receipts' (may be
   derivable from the passes/ sequence — spec the derivation if so).
2. **BINDER RUN:** execute the hr2 content binder end-to-end on whatever binds; REHEARSAL label
   in every manifest; stream-hash custody; typed-unresolved for the genuinely absent (no fake
   paths).
3. **m37 RECEIPTS:** produce same-parent freshness receipts (ip1 port 2) for the pass-3
   archive↔receipt pair and any fit/selector objects present.
4. **ACTIVATION JOIN REHEARSAL:** already positive-controlled by ip1 on the v752 config — just
   RE-VERIFY the CLI runs green from a clean invocation here (tools/report_terminal_activation_join.py),
   no new control needed.
5. **T0 RUNBOOK:** a short executable checklist (commands, in order, w/ expected outputs) the
   terminal-day MAIN executes verbatim — the rehearsal's residue as an artifact.

## Boundaries

Scorer-FREE. Solve dirs READ-ONLY (copy, never move; the payload law binds — retain rehearsal
outputs). No Modal. Review gate honored for any .py; prefer NO new .py (adapters = spec rows
unless trivially small). Serializer commits, post-edit shas, [no-triality] [p0-ledger-ok],
--no-co-author; BLOCKED-GIT ⇒ fire-order for MAIN handoff. Durable memo
`.omx/research/ddm_t0r1_intake_rehearsal_20260811.md` w/ NEXT_IF_RESUMED + DEAD-ENDS.

## OPTIMAL FORM

Pins: reseal skeleton (committed) · hr2 memo/apparatus (436edf452c) · ip1 (a61c550890) ·
pass_03 receipt sha 93f8d7b4b668919d2357a02cde2a96fc0488ec7e2ac00a250f509d27dbef4c6e. SCOPE =
all 7 roles + all 5 ordered items. PRIOR-LAW PREDICTION (derived from the hr1 'typed
unresolved' design + the solve's observed emission): ≥2 of the 7 terminal roles have NO
current producer in the solve's per-pass emission (candidates: sensitivity map · convergence
receipts as a distinct object) — the demand list will require either a terminal-emission patch
or role adaptation BEFORE terminal day. FALSIFIER: all 7 bind cleanly from pass-3 objects ⇒
the intake is proven ready as-built and the rehearsal closes with the runbook only.
