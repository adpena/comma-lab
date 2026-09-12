# ddm_cons3 — consolidate the arms that closed after cons1/cons2 on 2026-09-11 (charter, MAIN 2026-09-12; operator full-authority GO; Opus)

## Why
cons1 (`.omx/research/ddm_cons1_consolidate_closed_arms_20260911.md`) and cons2 (`…ddm_cons2_consolidate_20260910_arms_receipts_20260911.md`)
consolidated everything closed up to their cut. Since then the day landed four pointer moves and closed nine more arms whose typed
receipts, lane rows, equations, DSL levers and task rows are not consolidated: **pc3** (move 45, CAP1 predictor refit; rung prices expire at
a pointer move), **ntb2** (move 46, frame-embedding even rounding; first-measurement chain; frame_quad LOSES on the refit prior),
**hpr1** (moves 47/48: retrained prior −887 B, shape rung +812 B vs its control FALSIFIED, even rounding on the refit prior −248 B,
q re-solve −9,440 B held-out CLOSED, step-4 rounding +58 B), **pr18** (receiver BEHAVIOUR digest v2; consumer-fix rows), **pr19**
(`measured_t4_identity_class_envelope` risk mode; transitive inheritance REFUSED; the code-and-row-together LAW), **tmx1** (tail-mixer
refit LOSES +20 B; mass-conditioned ranking REFUTED; refit law n=2), **obx2 design A closure** (noise 9.40× smooth at matched RMSE;
pre-burn step-0 render-floor gate 3.199e-4, measured 7.22× over), **rbf1** (post-render pixel actuators pay a quadratic pose tax), **gpp1 /
mxo3** (resumed; closed), **sr5 unit 2** (placeholder hash ≠ certificate; sampled prefilter never decides equality; `df -h` prints GB under
"Gi"; 40 GiB reserve underived, measured need ~21–24 GiB), **swp4** (PR #140 restage on move 47, 91/93), **pc4** (RETIRED before spawn),
**cons2 itself** (its own receipts). The monitor reads CONSOLIDATE-NOW (pile_files 102; ssd_only_code 33 blobs). MAIN measured a new
defect today that dpi1 is curing (warm-start init drops the bit-depth state; memory
`warm_start_init_dropped_the_bit_depth_state_refits_relearn_depths_from_8_bits_20260912`) — register it as an ANTI-PATTERN now with MAIN's
receipts; its price row is dpi1's and stays pending.

## Deliverable ($0; no launches; no pointer/seal writes; NEVER a live arm's files — ddm_ren1 and ddm_dpi1 are LIVE)
1. Enumerate the arms above from `.omx/research/ddm_*_2026091{1,2}*.md` and their dirs, take the COMPLEMENT of cons1's and cons2's tables
   (read both memos' §1 tables first; do not redo a row they own), and commit every durable typed receipt and memo not yet tracked
   (skip gitignored logs, serializer patch copies, CLAIMS_/CALL_LEDGER_ snapshots, recall dumps > 1 MB, ExFAT `._` stubs — list every skip
   with its reason); batches ≤ 25 files; post-edit shas; no co-author trailer; `[no-triality] [p0-ledger-ok]`; `REVIEW_GATE_OVERRIDE=1`
   only for non-.py; any .py gets two visible review passes.
2. Lane rows + probe outcomes per arm (`tools/lane_maturity.py audit|mark|validate`; levels per evidence; reactivation criteria in notes;
   lane ids never carry `v<digit>` tokens).
3. Canonical equations / anti-patterns for the MEASURED laws above, each with an EmpiricalAnchor (memo path + sha + the numbers), producers
   AND consumers (no orphans; read `tools/list_canonical_equations.py --json` invariants first). Required at minimum: the refit-by-fitted-capacity
   law at n=2 (hpr1 −887 B / tmx1 +20 B; the mass-conditioned ranking as the anti-pattern); the identity-class risk envelope + no-transitive-
   inheritance contract fact (pr19) and the code-and-row-together law; the behaviour-digest scope rule (pr18); obx2's noise-vs-smooth 9.40× and
   the step-0 render-floor gate; rbf1's quadratic pose tax; pc3's "rung prices expire at a pointer move"; sr5's four custody laws; the
   warm-start-drops-state anti-pattern (dpi1 pending price). Where a memo lacks `# FORMALIZATION_PENDING` or a cite, add the cite.
4. Design-state `Lever` factories (`tac.witness_dsl`) for the levers with a verdict (pc3 predictor refit; ntb2 frame_even/frame_quad; hpr1
   past_dilation/conv_a_dilation/q-re-solve/step-4 rounding; tmx1 mixer refit; obx2 design A; rbf1 pixel actuators) so the activation ledger
   carries their state; the triality drift detector must pass.
5. Task-ledger transitions (cite CONTENT never bare ids); memo `.omx/research/ddm_cons3_consolidate_late_20260911_arms_20260912.md` with the
   table arm → records → lane → equations → DSL → task, and the monitor before/after (`ssd_only_code` blobs: commit or certify each with a
   rationale). Checkpoint as `ddm_cons3`.

## Boundaries / OPTIMAL FORM / prior negatives
As cons2's charter (`.omx/research/ddm_cons2_consolidate_20260910_arms_receipts_charter_20260911.md`): reference form = cons2's landing
(every registration anchored AND consumed; one canonical ledger, no new lists; skip ExFAT stubs; a Git-object denial is not a stop).
Corrections that stand: cons1's (obx2 gate 3.199e-4; mxo3 under mxo2's name; a refit has no secant); cons2's four backwards-reading rows;
MAIN's append-only gs3 corrections (cap 154,507 B; 7.22×; receiver rung 8,365 B short; bz2d ×1.157 ratio retracted). Never edit `upstream/`,
the PR tree, sealed trees, or contract code. Read `docs/operating_manual_craft_handoff.md`. Final message: the table, commit shas, skips with
reasons, monitor before/after, every boundary, and the frontier line
`composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)` unchanged.
