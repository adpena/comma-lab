# ddm_bhw1_winwin_cone_rescreen — apply fcd1's B/H/W law to the families that were closed by TRADE-space search (task #1322; owning memo `ddm_fcd1_field_for_coder_diagonal_20260829.md`, commit 6df3b4ea9b)

## MANDATE

Operator 2026-08-29 (standing GO, verbatim): *"Recover and respawn and continue with all"* +
*"be creative and weird and think divergently."*
Routed finding: fcd1 measured an EXCEPTION to the sharp-optimum law
(`ddm_gestalt_the_three_laws_20260823.md`; five arms had concluded "the HPAC field+model optimum
is sharp in every direction"). Among the 227,671 positions where shipped tokens disagree with the
HPAC coding argmax, exactly **5,268 are GT-BENEFIT** (coding argmax == GT, token wrong) and
editing them toward GT **SHRANK the archive −3,756 B** at −5.703872 bits/edit — rate AND
token-truth improved together. The mechanism: the coder was paying a surprise premium to transmit
labels the model correctly disbelieved. **Every prior family searched rate↔distortion TRADES, so
none of them ever probed the win-win cone.** This arm re-screens the CLOSED families with the
B/H/W classification that exposed it. Cheap: the fields and receipts are retained; the
classifier is landed (`experiments/ddm_fcd1_field_for_coder_diagonal.py::classify_pool`).

## SCOPE

1. **Inventory the trade-space-closed families** whose objects are retained and re-screenable —
   seeded (verify each at source, do not trust these labels): dg2 diagonal
   (`ddm_dg2_diagonal_distortion_verdict_20260824.md`, refused 686×) · jf1/jf2 terminal diagonal
   (`ddm_jf2_terminal_diagonal_harvest_20260826.md`, 738374ded2) · oe1 zero-stored causal
   (`ddm_oe1_online_escape_member_20260822.md`) · ld1 lossy Lane
   (`ddm_ld1_lane_lossy_drop_exchange_20260822.md`) · ae1 anti-predicted excess. For each record:
   object retained? (path+sha) · was the search TRADE-space or win-win-aware? · re-screenable at $0?
2. **Run B/H/W per re-screenable family**: tokens vs that family's coding argmax vs DALI GT →
   {B = coding-argmax==GT & token wrong (the win-win cone) · H = model-miss · W = wash}. Use the
   landed `classify_pool`; where the family's coding argmax was never persisted, say so plainly
   and cost the re-derivation rather than inventing one (the producer
   `experiments/ddm_df1_drop_field.py` is hardwired to the shipped DX2 stream — adapting it is a
   MECHANISM extension, declare it).
3. **Price the top cone by REAL joint re-encode** (fs2 law, never entropy/additive): for the
   family with the largest B pool, build the GT-benefit-edited field and re-encode. Report exact
   marginal bytes vs that family's own base AND vs the gb1 pointer.
4. **Fork per family**: cone empty (B≈0) → the family's trade-space closure STANDS and is now
   win-win-verified (a strengthening, record it) · cone non-empty with negative marginal bytes →
   a NEW live rate opening; hand MAIN a typed row with the distortion legs UNMEASURED and
   explicitly named (fcd1's exact posture — B/H labels are NOT realized SegNet flips).
5. **Typed exit**: per-family {B/H/W counts, bits/edit, real marginal B, verdict} + the law's
   revised scope statement.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm. NO distortion claim from token labels —
  fcd1's own dead-end: B/H token labels ≠ realized SegNet flips (only a scorer run decides).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; receipts to `/Volumes/APDataStore/pact/ddm_bhw1_winwin_cone_rescreen/`.
- Axis honesty: `[macOS-CPU frozen-scorer advisory]`, score_claim=false, promotable=false.
- Do NOT touch the fcd1/fcd3 consumer store or the fcd3 scorer lane (fcd3 owns it).
- Storage: check free space before materializing any field; AP tier per the disk rules.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- fcd1 (`ddm_fcd1_field_for_coder_diagonal_20260829.md`): same-move compensation dead 45.18× ·
  entropy/average/additive pricing dead · B/H labels ≠ realized flips · stale native corrector
  refuses jt21 (use the Python `FreeCorrector` path).
- fcd2 (`ddm_fcd2_distortion_legs_execute_20260829.md`): the fcd1 UNION refused at the pose gate
  (42.96× base) — so a large cone is NOT automatically admissible; any cone found here inherits
  the same distortion adjudication and must NOT be presented as a win.
- The sharp-optimum law's five arms (dg2 · jf1 · oe1 · ld1 · ae1): each is a REAL negative in
  trade space. This arm does not overturn them; it asks a DIFFERENT question of the same objects.
- `ddm_jt23_coder_collection_compose_verdict_20260826.md`: the coder axis is CLOSED at 0 B — this
  arm is a FIELD question, not a coder race; do not re-race coders.

## OPTIMAL FORM

- Family exemplar: fcd1 itself is the reference form — memo
  `ddm_fcd1_field_for_coder_diagonal_20260829.md` + commit 6df3b4ea9b, reference receipts
  `BYTE_ONLY_RESULT.json` + `PREPARE.json` in
  `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`, classifier
  `experiments/ddm_fcd1_field_for_coder_diagonal.py::classify_pool` (:221). Run that landed form
  on other objects — no new mechanism.
- SCOPE reductions declared per family (a family whose coding argmax was never persisted is
  screened at REDUCED scope and labelled so). MECHANISM reductions FORBIDDEN (real coder for any
  byte claim; real GT table — the DALI-lineage GT, per the #1142 wrong-objective cure).
- **PRIOR-LAW PREDICTION (falsifiable):** the win-win cone exists because a coder pays a surprise
  premium for labels its model disbelieves — a property of the FIELD×MODEL pair, not of fcd1's
  particular search. So ≥1 other retained family should show B > 0 with negative marginal bytes.
  FALSIFIER: every re-screenable family returns B ≈ 0 (< 0.1% of its disagreement set) or
  non-negative marginal bytes — that would scope the win-win cone to the fcd1 object alone
  (INSTANCE, not a law), and the sharp-optimum law survives everywhere else. Count it plainly.

## DELIVERABLE

`.omx/research/ddm_bhw1_winwin_cone_rescreen_20260829.md` — typed rows: (1) family inventory w/
re-screenability + object shas; (2) per-family B/H/W table; (3) real-coder marginal for the top
cone; (4) per-family verdict + the law's revised scope; (5) any new live opening handed to MAIN
with distortion explicitly UNMEASURED; (6) NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
Commit via the serializer. End with the own-vehicle frontier line.
