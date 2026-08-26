# ddm_jf2_terminal_diagonal_harvest — finalize jf1's seven epoch-60 fits: physical model+stream bytes vs 127,292 B (the diagonal cell's actual verdict, never measured)

## MANDATE

hv2's FIRE-NOW head (memo `.omx/research/ddm_hv2_harvest_consumption_sweep_20260826.md`,
rank 1): the jf1 joint field+model refit trained SEVEN k-arms (null, k002500, k005000,
k010000, k020000, k040000, k060000) to TERMINAL epoch-60 QAT-stage-end checkpoints
(`.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/training/*/model.checkpoints/qat_stage_end_epoch_0060.pt`)
— and the byte diagonal was only ever computed at EPOCH-2 (`BYTE_DIAGONAL_SCOPE_E0002.json`;
its positive control FAILED by 7,554 B — harness #1221's finding w/ its ledger row). The
#1215 2×2 law says the diagonal (field AND model move together) is the only cell that could
win; #1239 entered it via a SOLVE and refused at 686× — but jf1's TRAINED entry is
unmeasured at terminal. This arm buys the verdict: pack each epoch-60 checkpoint through
the real coder → physical (model bytes, stream bytes, total) per arm → compare vs the
127,292 B joint token subsystem AND vs the epoch-2 diagnostic → BYTE_DIAGONAL terminal
receipt. LIVE-HYPOTHESIS (hv2): epoch-60 may reverse the epoch-2 byte-negative. FALSIFIER:
every terminal arm remains byte-negative vs 127,292 B → the trained-diagonal family closes
at TERMINAL scope with receipts, joining #1239's solve-refusal as the cell's second leg.

## SCOPE

1. RECALL: jf1 memo `.omx/research/ddm_jf1_joint_field_model_refit_20260823.md` +
   instrument `experiments/ddm_jf1_joint_field_model_refit.py` (commit `b5c8edf63b`) +
   the receipts tree above (verify all 7 terminal checkpoints + launcher `ok` receipts as
   hv2 demanded) + `BYTE_DIAGONAL_SCOPE_E0002.json` (the epoch-2 baseline to diff against)
   + `SCORER_FIRE_ORDER.json` (jf1's own recorded fire-order — honor or supersede it
   explicitly, never silently).
2. HARVEST: for each of the 7 arms, materialize the terminal physical rows — pack the
   epoch-60 state through the REAL coder path the instrument defines (rsf1 law: real coder
   bytes only, no entropy estimates) → per-arm (model B, stream B, total B). Write
   `BYTE_DIAGONAL_TERMINAL.json` next to the E0002 receipt (append-only tree; do not
   modify E0002). ALWAYS KEEP THE PAYLOAD: retain packed payloads + sha256s to
   `/Volumes/APDataStore/pact/ddm_jf2_terminal_diagonal_harvest/`.
3. VERDICT: per-arm total vs 127,292 B and vs the E0002 figures. If ANY arm lands under
   127,292 B: its distortion legs are UNMEASURED — do NOT project; emit the wj1-target
   consumer row (hv2 FIRE-NOW #4) + a named next-measurement (realized d_seg/d_pose through
   the real decode, n600) for MAIN routing. If all negative: close at TERMINAL scope,
   receipts per arm, and mark hv2's FIRE-NOW #3/#4 rows resolved-by-this-verdict in the
   canonical ledger (tools/canonical_task_status.py, actor ddm_jf2).
4. If the git-object write denial recurs (codex-sandbox class, hit pc2+hv2 on 08-26):
   retain a serializer-authored bundle + verified shas exactly as pc2 did — MAIN lands it.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal, NO scorer fires from the arm (any scorer follow-up is a
  named row for MAIN). Serializer commits w/ post-edit shas; `.py` = 2 review passes.
- No naive/toy/generic (operator 08-26): the pack path must be the instrument's REAL coder
  path — a lesser packer cannot produce the verdict; per-arm rows, no averaging (fs3 law:
  average ≠ marginal).
- jf1's receipts tree is append-only; new artifacts get new filenames.

## PRIOR NEGATIVE SIGNAL

- #1221 (its harness row + the E0002 receipt): epoch-2 positive control failed by 7,554 B —
  the baseline this harvest diffs against, not a family closure.
- #1239 (harness row + its memo in the ledger): the SOLVE entry to the diagonal refused at
  686× (pose 93.3% of damage) — a different mechanism; this arm measures the TRAINED entry.
- #1215/#1227 (sy2 memo lineage): object-change law — the diagonal is the last unentered
  cell; both legs' receipts complete the 2×2.

## OPTIMAL FORM

- Family REFERENCE exemplars w/ provenance pins (receipt-backed): the jf1 instrument,
  commit `b5c8edf63b` (its own pack/coder path is the reference form) · the gb1 verdict
  rigor (`.omx/research/ddm_gb1_groupbin8_verdict_20260824.md`, commit `884bb65f1e`) ·
  pc2's MAIN-landing pattern for git-blocked custody (memo
  `.omx/research/ddm_pc2_pose_carrier_live_remainder_20260826.md`).
- SCOPE reductions declared (none expected — 7 arms is the whole population). MECHANISM
  reductions FORBIDDEN: no entropy estimates, no partial-arm verdicts presented as family
  rows, no distortion projection from byte rows.
- **PRIOR-LAW PREDICTION (falsifiable):** the E0002 trend + the sharp-optimum law predict
  terminal totals REMAIN byte-negative vs 127,292 B for all 7 arms (the diagonal's trained
  entry joins the solve entry in refusal). FALSIFIER: any arm under 127,292 B — then the
  diagonal is byte-ALIVE at terminal and the distortion measurement becomes the campaign's
  next named fire.

## DELIVERABLE

`.omx/research/ddm_jf2_terminal_diagonal_harvest_20260826.md` — 7-row terminal byte table
(vs 127,292 and vs E0002) + BYTE_DIAGONAL_TERMINAL.json path + payload shas + ledger
receipts + GESTALT-DELTA line. Serializer commit (or bundle per §4). End with the
own-vehicle frontier line.
