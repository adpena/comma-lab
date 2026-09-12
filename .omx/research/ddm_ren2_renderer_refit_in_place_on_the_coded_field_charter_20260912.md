# ddm_ren2 — refit the shipped renderer IN PLACE on the CODED field, restored init, bounded render delta, terminal resolved pose (charter, MAIN 2026-09-12; operator full-authority GO; Opus)

## Why this is the remaining rung
ren1 (`.omx/research/ddm_ren1_renderer_provenance_and_refit_in_place_20260911.md`, sha 063b5dd1…; commits 2a0dc8f24, d64b3301e)
measured: the shipped SM3R renderer (66,339 params, mode 6, mixed 3/4-bit, keep_percent 1) was trained on GT-ORACLE tokens and has
NEVER been fit to any coded field; the shipped field a92e7d90… differs from DALI GT at 9,179 sites. Its gate PASSES on both legs: the
carrier's post-re-solve pose residue at 0.5 LSB camera RMS is k_post 0.0078 (noise) / 0.0190 (smooth) against k_payable 0.135451 at
move 48 — pose absorption is a CAPACITY at this amplitude, not the 170–220 / 13.82 cliff — so a refit must clear a d_seg cut of
≥ 1.006 % (noise-spectrum render change) or ≥ 2.682 % (smooth) plus the member's re-encode delta, with the terminal re-solve's
2–4 B carried (§4.1b). The two zero-pose-exposure alternatives ren1 ranked above it are now MEASURED closed (tmx1 +20 B; dpi1 +37 B:
the HPAC prior sits on its joint knee). ren1 also confirmed MAIN's init-drop law on this object (§3A): the warm-start init flattens the
16-entry depth table to `quant_bits: 4`, drops `keep_percent` and the prune mask, and binds to an old archive — the measured reason
ft1 refit a DIFFERENT object (its +31.23 % is not a verdict on the shipped renderer) and rw1 searched without the depth table in the loop.

## MAIN's decision on the loss shape (ren1 §4.6 item 1)
Seg objective in the loop; pose enters as the TERMINAL RESOLVED pose (the measured law: at ≤ 0.5 LSB camera RMS the carrier re-solve
removes 140–291× of the pose leg for 2–4 B). The "both scorers" intent is satisfied by MEASURING pose after re-solve at EVERY priced
checkpoint and refusing any checkpoint outside the bound — pose is never trained, never unmeasured.

## Deliverable
1. **Restore the init** (ren1 §3A, "What the next unit should run" items 1–2): per-tensor depth table `{frame_embed: 3, blocks.0.film: 3,
   rest: 4}`, `keep_percent = 1` + the prune mask (measured kept rows [11,13]/[34,119]/[30,189]), the per-axis fp16 scale rule and its
   5.96e-08 floor, `provenance.archive_sha256` re-bound to move 48's d830edd3…; diff the restored init's key set against the shipped
   deployed object and record it (MEASURED). **Control at step 0**: the restored init must pack (`pack_prune_mixed_candidate` → SM1S →
   brotli) to the shipped member bytes (29,862 B, sha 786950a5…) and score d_seg 1.03387e-04 / d_pose 4.58676e-06 on ren1's instrument
   within its 0.06 %/0.07 % reproduction; if not, STOP.
2. **Trainer** `src/tac/pr130_lift/train_semantic_quantized_resumable.py` with `--init <restored>`, `--weight-qat-q3q4`,
   `--film-row-dropout --film-row-dropout-protect-top`, canonical EMA, exact-R eval-roundtrip, per-stage checkpoints, `--resume-from`.
   Conditioning input = the CODED token plane a92e7d90… (never GT-oracle); seg target = the GT partition (gt_n600 lineage, DALI per the
   e2e declaration — state which and why). Budget: ren1's shipped schedule is 18,000 steps; a SCOPE reduction (e.g. 6,000 steps, declared)
   is legal; a MECHANISM change (loss form, quantizer, prune rule) is not. Launch via `tools/launch_detached_process.py --output-dir
   /Volumes/VertigoDataTier/pact/ddm_ren2/<stage> --nice 0 --done-receipt … -- <cmd>`; heavy waits as background until-loops on receipts.
3. **Build the amplitude bound** (ren1 §4.6 item 2): at every checkpoint, measure the render delta vs the shipped frame_1 render in
   camera LSB RMS (seeded RANDOM n ≥ 120 pairs at intermediate checkpoints; n600 at any checkpoint that is priced) and its smooth/noise
   split by ren1's decomposition; REFUSE any checkpoint above 0.5 LSB RMS; prefer the noise end. Record the curve.
4. **Price every admissible checkpoint end to end, the FIRST one first** (ren1 §4.4): (a) member re-encode by real pack+SM1S+brotli
   (value-dependent; fe1's +70 B fee does not extrapolate — measure); (b) exact n600 d_seg through R on the frozen CPU scorer (ren1's
   instrument, control reproduced); (c) terminal carrier re-solve with the canonical unforked `up2.solve_pair_realized` → post d_pose and
   carrier bytes; (d) S from components against move 48 (S 0.13638261682704697; d_seg 0.00010345, d_pose 4.59e-6, 179,111 B):
   net ΔS < −2e-5 is the fire bar. Report the required-cut table (1.006 %/2.682 %) against the measured cut.
5. **If a checkpoint clears the bar**: byte-close on the move-48 base (new member + re-solved carrier; everything else identical), twins,
   full cold n600 public parse-back; the decoded output's frozen-scorer d_seg/d_pose must equal the in-loop numbers (raw identity is NOT
   expected — say so); manifest regenerated from outside the tree and validated; literal census; smokes; retention with shas ≤ 8 GiB.
   Stage SEAL INPUTS (receiver code unchanged ⇒ normal seal inheriting move 48's MEASURED leg once; the SCORE needs MAIN's T4 fire since the
   raw changes) — NO seal call, NO Modal, NO packet: MAIN fires the contest-CUDA eval and the CPU sibling.
6. Memo `.omx/research/ddm_ren2_renderer_refit_in_place_on_the_coded_field_20260912.md`: the init diff, control row, the amplitude curve,
   per-checkpoint price table, fire verdict with the margin in units of the 34.8 B lottery AND of the instrument's 0.06 % seg reproduction,
   every boundary; serializer commits (two visible review passes per .py; `[no-triality] [p0-ledger-ok]`; NEVER a co-author trailer or AI
   attribution). Checkpoint as `ddm_ren2`; lane id `ddm_ren2_renderer_refit_in_place_coded_field_20260912` (claim it).

## Boundaries (binding)
Never edit `upstream/`, `submissions/semantic_joint_ctxmix/`, sealed trees, `src/tac/candidate_seal.py`, `src/tac/decode_wall_clock.py`, or
receiver CODE (the member's VALUES are the only counted change; the receiver must decode it unchanged). No Modal, no `authorize_*`, no
`fire_modal_auth_eval.py`. Never lower a storage reserve; local disk is not a tier (Vertigo ~53 GiB free, reserve 40 GiB; APDataStore
~51 GiB). Do not touch ren1's or dpi1's directories (read-only). Never wrap a Modal-querying tool in `timeout` (you run none).
Read `docs/operating_manual_craft_handoff.md`; label MEASURED / DERIVED / INFERRED / ASSUMED.

## OPTIMAL FORM
- Reference form: the shipped stage-08 trainer as ren1 reconstructed it (§1.1) with its shipped quantizer REALIZED in the loop (the two
  levers) and the restored init; the real coder for the member; ren1's n600 frozen-scorer instrument; the canonical re-solve. Declared
  deltas: step budget (SCOPE), conditioning on the coded field instead of GT-oracle (the rung itself — the object's first fit to what it
  actually receives), the amplitude bound (a constraint, not a mechanism change). No toy bracket: every priced row is n600 exact through R.
- Provenance pins: HEAD (record), ren1 memo sha 063b5dd1…, ren1 probe RESULT.json 6d0de07f…, rows.jsonl 297df0ee…/67426e20…/1850fd2c…,
  move 48 seal `.omx/research/ddm_hpr1_20260911/v2/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json` (archive d830edd3…, member
  786950a5… 29,862 B, body 17e0fd0b… 36,130 B), field a92e7d90…, trainer file sha (record).

## Prior negatives accounted (operator 2026-08-15)
- ft1 +31.23 %: refit a DIFFERENT object (init drop); do not cite it as this object's verdict; do reproduce its failure mode as a
  falsifier (seg-only fine-tune without the depth table/prune mask in the loop should NOT be run here).
- rw1 (int4 grid, 0/150 steps): searched without the depth table; one code step moves 240–455 cells (10–19× the 23.6-cell bar) — so the
  refit's smallest action is coarse; report cells moved per checkpoint.
- fe1 member fee (+70 B flat at N ≤ 200 codes): a refit changes all 59,376 codes — measure, never extrapolate.
- obx2's smooth-vs-noise law INVERTS on this object (smooth 19.4× worse here): quote the spectrum of YOUR render delta; never import a
  ratio from another object.
- tmx1 +20 B / dpi1 +37 B: refits of sections already fit to the field lose; this object was never fit to the field (hpr1's positive
  genus) — a hypothesis, priced, not assumed.
- Pose re-solve mandatory after every render change (op 09-10); frame-0 repair in admission.
- rp1 r2: bind every input by sha; a flag and a constant can disagree silently (init_bits-class traps) — prove the realized quantizer from
  the packed bytes at step 0, not from the flags.

Final message: init diff, control row, amplitude curve, per-checkpoint price table, fire verdict + margins, seal-inputs path if staged,
commit shas, retained bytes, every boundary, and the frontier line
`composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)` unchanged (MAIN fires).
