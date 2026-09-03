# ddm_qn1_qbr1_n600_realization_ticket — the CONDITIONAL-N600-BUY made READY: a ticket generator that binds the burn's winning cell's exact QBF1 archive to a same-object, payload-retaining n600 realization (BR2 protocol) the moment the sealed adjudication says OPTIMIZATION_LIVE — so the critical path never waits on a human to assemble the next fire order

## MANDATE

Operator standing GO; Opus subagents authorized 2026-09-03 ("for now"). The QBR1 six-cell burn (chain driver
`experiments/ddm_qbr1_cell_chain.py` fa80f8256 live; sealed order in
`/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/SEALED_MAIN_FIRE_ORDER.json`) is the sub-0.12 critical path
(`ddm_gs3_gestalt_after_submission_20260903.md` + addendum: born-field accuracy by OPTIMIZATION is the live door).
Its pre-registered adjudication (`ADJUDICATION_SCHEMA.json`: per-seed treatment win = treatment S_hat < control
S_hat on n32 stratified Horvitz-Thompson; pose corner = d_pose_hat < (0.12 − rate_exact − 100·d_seg_hat)²/10;
OPTIMIZATION_LIVE = ≥2/3 seeds win AND ≥2/3 treatment pose corners pass; `no_n600_buy_before_sign_repeats`)
ends in the row **CONDITIONAL-N600-BUY** (`ddm_qbr1_born_fairform_burn_prep_20260902.md` NEXT_IF_RESUMED):
"build a same-object retained n600 ticket rather than transferring n32 or BR2 distortion". Nobody has built that
ticket generator. When the adjudication lands (~+15 h) MAIN must be able to fire it in one command.

## SCOPE

1. `experiments/ddm_qn1_qbr1_n600_realization_ticket.py`: reads `ADJUDICATION_RESULT.json` (or, for the dry run,
   a synthetic result over cell 1's existing milestones); refuses unless the typed outcome is
   `OPTIMIZATION_LIVE_DISTORTION_ROUTE` (or `--dry-run`); selects the winning TREATMENT cell per the schema (best
   S_hat among seeds that won); binds its step-5000 exact QBF1 archive bytes (sha256 + bytes from the cell's
   RESULT.json / milestone materialization under `runs/<seed>/treatment_zero_native/`) and its decoded field;
   writes a SEALED n600 fire order in the BR2 protocol (`experiments/ddm_br2_born_object_scorer_realization.py`
   97846a07b: `realize --output --resume-from --scorer-claim-id --launch-authorized`, 30-pair chunks, every render
   chunk + decoded field + results RETAINED, AP free-bytes precondition ≥ 1.5 GB, unique scorer claim
   placeholder, expected wall ≈ 485 s per BR2's measured realization), plus the exact score-law recompute
   (100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489) and the pre-registered falsifier
   (d_seg ≤ 0.01 ∧ d_pose ≤ 1.25e-4 at ≤ 137,986 B opens the first byte-feasible distortion path — from qxr1's
   fire order; cite it). The ticket is machine-readable (schema `ddm_qn1_n600_realization_fire_order.v1`) and
   its argv arrays are verbatim.
2. `--dry-run` against the LIVE cell 1's completed milestones (control cell, plumbing only — say so in the
   receipt; never present it as a treatment): proves the archive/field bindings, the chunk plan, and the
   refusal paths (missing result, INCONCLUSIVE_MIXED, pose corner failed, claim placeholder).
3. Tests (`tests/test_ddm_qn1_qbr1_n600_realization_ticket.py`): outcome refusals, winner selection per schema,
   verbatim argv, sha binding, dry-run receipt schema.
4. Do NOT run the realization (MAIN owns the scorer lane; the burn owns Metal). Deliver the generator + a
   dry-run receipt + the exact MAIN command.

## HARD CONSTRAINTS

- `upstream/` and `submissions/semantic_joint_ctxmix/` READ-ONLY. NO scorer, NO Modal, NO Metal/MPS. Never write
  under `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/` or `authorized_configs/`; the dry-run writes
  ONLY under `/Volumes/VertigoDataTier/pact/ddm_qn1_qbr1_n600_realization_ticket/`.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 review passes; ruff clean.
- Never invent a flag: grep BR2's and the burn script's `add_argument` first; bind only real ones.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_br2_born_object_scorer_realization_20260831.md` — the payload-retaining n600 protocol; DISTORTION-REFUSED
  1,046× on the OLD born object: this ticket realizes the NEW (burn-trained) field, never transfers that number.
- `ddm_qxr1_realization_adjudication_20260902.md` — the falsifier and the "identical-by-construction ⇒ derive,
  don't fire" rule: refuse if the winning cell's consumed state is byte-identical to a scored ancestor.
- `ddm_bz2d_distortion_verdict_20260830.md` — 1.157× token→argmax amplification; pose 152× worse on the fork:
  never inherit distortion; the ticket's whole point is a fresh same-object row.
- memory `m110` — pose absolute budget ≤ 1.25e-4 (∂S/∂d_pose 626.5).

## OPTIMAL FORM

- Family exemplar: BR2's realization fire order and receipts, reference
  `experiments/ddm_br2_born_object_scorer_realization.py` (commit 97846a07b) and qxr1's `FIRE_ORDER.json`
  (`/Volumes/APDataStore/pact/ddm_qxr1_qxo1_born_realization_binding/FIRE_ORDER.json`); the burn's
  materialization contract `experiments/ddm_qbr1_born_fairform_burn_prep.py` (commit 42d322db5).
- SCOPE reductions: none (the ticket is full n600). MECHANISM reductions FORBIDDEN: no n32→n600 transfer of
  distortion; no synthetic archive; no chunk plan that discards payloads.
- **PRIOR-LAW PREDICTION (falsifiable):** the dry run binds cell 1's step-5000 archive and field byte-exactly and
  every refusal path fires on its synthetic negative. FALSIFIER: any binding or refusal that does not — count it.

## DELIVERABLE

`.omx/research/ddm_qn1_qbr1_n600_realization_ticket_20260903.md` — the ticket schema, the dry-run receipt, the
exact MAIN command, RECALL EVIDENCE, NEXT_IF_RESUMED, LIVE-HYPOTHESES, DEAD-ENDS. Commit via the serializer. Cite
`docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
