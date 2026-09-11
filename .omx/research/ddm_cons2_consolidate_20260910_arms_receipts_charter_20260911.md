# ddm_cons2 — consolidate the 2026-09-10 arms' 132 typed receipts and their lane/equation/DSL/task dispositions (charter, MAIN 2026-09-11; operator full-authority GO; Opus)

## Why
cons1 (`.omx/research/ddm_cons1_consolidate_closed_arms_20260911.md`, commits d33e59a3a…171cf81fd) consolidated the nine arms that closed on
2026-09-11 and refused, correctly, to absorb out-of-scope debt: **132 typed receipts of the 2026-09-10 arms** remain untracked (rlc2–rlc5,
pr12–pr17, ffi1–ffi6, cust1, cpx1–cpx3, swp2/swp3, sr4, gdc1–gdc4, ls1/ls2, ntb1, mxo1/mxo2, gpp1's predecessor, obx1, pc3's predecessor,
vr7/vr8, tc2–tc4, bnd2/bnd3, eb2, md1–md4 re-pins), plus their lane rows, probe outcomes, equations and task transitions where memos
carry `# FORMALIZATION_PENDING`. The consolidation monitor still reads CONSOLIDATE-NOW (pile_files 241; 828 residual files, of which the
staging trees, bundle-verify trees, serializer patch copies and logs are deliberately untracked and stay so).

## Deliverable ($0; no launches; no pointer/seal writes; never a live arm's files — ddm_hpr1 is LIVE)
1. Enumerate the 2026-09-10 arms from `.omx/research/ddm_*_20260910*.md` and `.omx/research/ddm_*_20260910/` dirs; for each, commit the durable
   typed receipts and memos (not gitignored logs, not serializer patch copies, not CLAIMS_/CALL_LEDGER_ snapshots, not recall dumps > 1 MB —
   list every skip with its reason); batches ≤ 25 files; post-edit shas; no co-author trailer; `[no-triality] [p0-ledger-ok]`.
2. Lane rows + probe outcomes for each (levels per evidence; `lane_maturity.py audit|mark|validate`); reactivation criteria in notes.
3. Canonical equations/anti-patterns for the 09-10 MEASURED laws still pending formalization — at minimum: the lane-surprise atlas bound
   (ls1/ls2: oracle 17,534 B receiver-visible, 1,035 B short), the tail composition sub-additivity by pair overlap (sj/rlc composition memo),
   the fitted-scalars-in-receiver-code rule-118 anti-pattern (move 41 retraction), the derived-listing-in-identity-digest anti-pattern (pr14/pr18),
   the container-break one-sample lottery (sd 34.8 B), the first-order token price is a ranking not a charge (0.15–0.27× / 1.28×), the
   inherited-leg-cannot-be-inherited-twice contract fact (ntb2 chain break). Each with an EmpiricalAnchor (memo path + sha + numbers),
   producers and consumers; no orphan equations; read the registry invariants first (`tools/list_canonical_equations.py --json`).
4. Design-state `Lever` factories for 09-10 levers with a verdict (gdc1–gdc4 generator doors, ls1 atlas, mxo1 corrector, tc contexts,
   bnd recode) so the activation ledger carries their state.
5. Task-ledger transitions; memo `.omx/research/ddm_cons2_consolidate_20260910_arms_receipts_20260911.md` with the table and the monitor
   before/after. Checkpoint as `ddm_cons2`.

## Boundaries / OPTIMAL FORM / prior negatives
As cons1's charter (`.omx/research/ddm_cons1_consolidate_closed_arms_20260911_charter_20260911.md`): reference form = cons1's landing; every
registration anchored and consumed (mh1: 68.6 % of findings reach no consumer); one canonical ledger, no new lists (m36); skip ExFAT stubs;
never land a live arm's files; a Git-object denial is not a stop. Corrections cons1 made stand (obx2 gate 3.199e-4; mxo3's memo under mxo2's
name; a refit has no secant).
