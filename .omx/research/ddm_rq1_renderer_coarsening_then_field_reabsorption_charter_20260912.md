# ddm_rq1 — RATE FIRST: coarsen the shipped renderer's realization (no training), then let the field RE-ABSORB the bias by pre-distortion; exact price on move 48 (charter, MAIN 2026-09-12; operator full-authority GO + "be creative and divergent"; Opus)

## The bet, and why its prior is not ren2's
ren2 trained the renderer at a FIXED field and lost at every checkpoint: the coded field is the renderer's pre-image, so any weight change
breaks it (memo `.omx/research/ddm_ren2_renderer_refit_in_place_on_the_coded_field_20260912.md`). This charter inverts the order. The
semantic member is 29,862 B (16.7 % of the archive; body 36,130 B laid out by shapes, `keep_percent`, and the DEPTH TABLE, never by
values — ren1 §4.4). The field, not the renderer, is where the campaign's seg was bought: jg1 measured 95.9 % of seg debt is render→
re-segment loss, and sj1's multipass PRE-DISTORTION writes "wrong" labels so the receiver's own render lands on the right argmax
(`experiments/ddm_sj1_multipass_token_predistortion.py`, docstring). sj1 pass 6 measured that when 365 planes were re-rendered the
closed repair family came back 2.12× STRONGER (memory `repair_family_exhausts_per_object_20260910`). So: take rate from the renderer's
realization first (depth table 4→3 on int4 tensors and/or deeper FiLM pruning — NO training; the same float weights, coarser grid), pay
seg at the fixed field, then run the pre-distortion passes against the coarsened renderer and measure how much of the debt the field
re-absorbs, at what token-byte cost. The exchange: 1 B = 6.658589531221714e-7 S; 1 flipped cell over 600 frames of 196,608 px =
8.48e-7 S (a cell is worth 1.27 B). Fire bar: net ΔS < −2e-5 vs move 48 (S 0.13638261682704697; d_seg 0.00010345; d_pose 4.59e-6;
179,111 B). Pre-registered expectation: unknown sign; the variant table decides whether Stage B runs.

## Stage A ($0, local; the rate/debt table; STOP rule inside)
1. Reconstruct the shipped renderer exactly as ren2 did (restored init: depth table `{frame_embed: 3, blocks.0.film: 3, rest: 4}`,
   keep_percent 1, kept rows [11,13]/[34,119]/[30,189], scale rule + 5.96e-08 floor; control must pack to member 29,862 B sha 786950a5…
   and reproduce n600 d_seg 1.03387e-4 on ren1's instrument; STOP if not). Reuse ren1's/ren2's landed producers (commits 2a0dc8f24,
   0215c504d); do not rewrite them.
2. Realize coarsened variants from the SAME float weights with the shipped packer (`pack_prune_mixed_candidate` → SM1S → Brotli
   q10/lgwin16 + CK2, the shipped container — ren2's correction): (a) depth 4→3 on all fourteen 4-bit tensors; (b) 4→3 on the eight
   largest only; (c) 4→3 on `head`+`blocks.3` only (rw1's most sensitive); (d) keep_percent pruning one notch deeper on the FiLM rows;
   (e) the shipped depths with the per-axis scale re-fit (control for the packer). For each: member bytes by REAL encode (twins), and
   n600 d_seg through R at the FIXED field on the frozen CPU scorer (ren1's instrument, DALI GT lineage), plus the flipped-cell count
   and its class split. Table: rate bought (B, S) vs seg debt incurred (cells, S) vs the pre-distortion repair fraction sj1 measured
   at first passes on a fresh object (cite jg3/jg5/pass-3 receipts; read them, do not quote from memory).
3. STOP rule: if for EVERY variant the seg debt exceeds the rate gain even at a 100 % repair fraction, Stage B does not run; report.
   Otherwise pick the variant maximizing (rate gain − debt × (1 − r_repair)) with r_repair taken from the measured pass-1/2 fractions.

## Stage B (the field re-absorbs; sj1's machinery, no new mechanism)
4. Swap ONLY the semantic member in a copy of move 48's runtime tree (receiver code unchanged; the member's values are the counted
   change). Run `ddm_sj1_multipass_token_predistortion.py` passes against the coarsened renderer on all 600 pairs (shards via
   `experiments/ddm_sj1_pass_shards.sh`; per-stage checkpoints; `--resume`; heavy steps through `tools/launch_detached_process.py
   --output-dir /Volumes/VertigoDataTier/pact/ddm_rq1/<stage> --nice 0 --done-receipt …`), with the seg-Lagrange admission at the
   exchange rate (`experiments/ddm_sj1_joint_admission.py` — read pass 6's charter and memo for the exact rule and STOP bars). Record
   per pass: cells repaired, tokens moved, token-stream bytes (real RLC1 encode, twins), d_seg, the reach decomposition.
5. Carrier re-solve on the edited field (canonical unforked `up2.solve_pair_realized`; pose read on the RESOLVED pose), then the exact
   archive on the move-48 base (new member + edited field + re-solved carrier), twins, cold n600 public parse-back with the decoded
   output's scorer numbers equal to the in-loop numbers, manifest regenerated from outside the tree, census, smokes, retention with shas
   (≤ 8 GiB; Vertigo ~52 GiB free, reserve 40 GiB never lowered).
6. S from components vs move 48; if net ΔS < −2e-5, stage SEAL INPUTS (receiver code unchanged ⇒ normal seal inheriting move 48's
   MEASURED leg once; the raw changes ⇒ MAIN fires the contest-CUDA eval and the CPU sibling). NO seal call, NO Modal, NO packet.
7. Memo `.omx/research/ddm_rq1_renderer_coarsening_then_field_reabsorption_20260912.md`: the variant table, the pass table, the price,
   margins in units of the 34.8 B lottery and of the instrument's 0.06 % reproduction, every boundary; serializer commits (two visible
   review passes per .py; `[no-triality] [p0-ledger-ok]`; NEVER a co-author trailer or AI attribution). Checkpoint `ddm_rq1`; lane id
   `ddm_rq1_renderer_coarsening_then_field_reabsorption_20260912` (claim it).

## Boundaries (binding)
Never edit `upstream/`, `submissions/semantic_joint_ctxmix/`, sealed trees, `src/tac/candidate_seal.py`, `src/tac/decode_wall_clock.py`, or
receiver CODE. No Modal, no `authorize_*`, no `fire_modal_auth_eval.py`. Local disk is not a tier; every payload on
`/Volumes/VertigoDataTier/pact/ddm_rq1/` with sha-verified retention. ren1/ren2/sj1/dpi1 directories read-only. No ScheduleWakeup; waits are
background receipt-only until-loops. Label MEASURED / DERIVED / INFERRED / ASSUMED. Read `docs/operating_manual_craft_handoff.md`.

## OPTIMAL FORM
Reference forms: the shipped packer on the shipped float weights (no toy quantizer); sj1's multipass machinery exactly as pass 6 ran it (its
admission rule, its shard layout, its DALI-lineage authority); ren1's n600 instrument. Declared deltas: the depth table / keep_percent
(the rung itself); pass count (SCOPE — stop at the pass whose reach falls below pass 6's 1 %/pass bar or at budget, and say which).
Provenance pins: HEAD (record); ren1 memo 063b5dd1…; ren2 memo (record sha); sj1 pass-6 memo + charter (record shas); move 48 seal
`.omx/research/ddm_hpr1_20260911/v2/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json` (archive d830edd3…, tree
`/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime`); field a92e7d90….

## Prior negatives accounted (operator 2026-08-15)
- ren2: gradient steps at fixed field lose — this arm trains nothing; the field moves, the renderer only coarsens.
- rw1: one int4 step moves 240–455 cells; a 4→3 re-grid moves every weight by up to half a step — the debt is EXPECTED to be large;
  the bet is the repair fraction, so measure the debt before spending a pass.
- the cross / NR1: "small" does not predict distortion; this is not a small body — it keeps the 118 KB field and asks it to absorb.
- sj1 pass 5/6: the family exhausts per object and re-opens on re-render; the pass bar (1 %/pass) and the exchange-rate admission bind.
- fe1: member re-encode fee is value-dependent; measure by real encode at the shipped container (q10/lgwin16 + CK2).
- Pose re-solve mandatory after every field/render change (op 09-10); frame-0 repair in admission.
- gt lineage: DALI table authority (`up2.verify_gt_lineage` fails closed); never PyAV for the verdict.

Final message: control row, the variant table (rate bought vs debt), Stage-B pass table if run, the exact price with margins, seal-inputs
path if staged, commit shas, retained bytes, every boundary, ending with
`composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)` unchanged (MAIN fires).
