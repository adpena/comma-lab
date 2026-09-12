# ddm_dpi1 — restore the HPAC prior's BIT-DEPTH state to the warm-start law, retrain on the current field, price on move 48 (charter, MAIN 2026-09-12)

## The measured defect this rung cures (MAIN, host 2026-09-12 ~00:50Z)
The reference refit law (cl2/jf1/hpr1: `tools/train_ddm_cl1_hpac_capacity.py`, 60-epoch cosine, seed 20260716, batch 8,
QAT fraction 0.5, λ 1.0, Metal) warm-starts from `/Volumes/VertigoDataTier/pact/ddm_hpr1/train_inputs/init.pt`
(sha cf411127…f383, the "epoch-634 EMA init"). That file carries **ZERO `*.bit_depth` keys** (zip-scan of its
data.pkl). The trainer registers every row's bit depth at `init_bits = 8.0`, then `load_state_dict(strict=False)`
and explicitly tolerates missing `.bit_depth` keys (`allowed_missing`, ~line 1193). So every refit under this law
starts its per-row depths at 8 bits and has 30 QAT epochs (lr_bits 0.01) to descend. The source checkpoint
`/Volumes/VertigoDataTier/pact/ddm_hv1_harvest_compose/retained/epoch_0634.pt` (sha 5007beae…47ec) DOES carry
10 `bit_depth` tensors (`conv_a.bit_depth`, `conv_b1.bit_depth`, …). The state was dropped when the EMA init
was cut. hpr1 measured the consequence without naming the cause: retrained priors sit at mean row depth
**5.888–5.890 bits** vs the shipped prior's **4.116**, identical across geometries, "a 60-epoch training-budget
effect" (hpr1 memo §4a/§4d, sha 26573f9b…). The packed price of that inflation on move 47's prior was
**+351 B** (12,262 vs 11,911 B) before ntb2's frame-even rounding took the refit prior to 11,629 B (move 48).
This is a SILENT DEFAULT in the harmful direction (the confound signature): the cure is to restore the state.

## Deliverable (rate-only; receiver UNCHANGED from move 48)
1. **Build the owned initializer** `init_depths.pt` = init.pt's `state_dict` + the 10 `bit_depth` tensors from
   epoch_0634.pt (same keys, same shapes; assert every row of every compressible module has a depth; record
   both source shas and the new file's sha). Prove the depths you loaded reproduce the shipped mean row depth
   (~4.116 bits) with `bit_depth_histogram` at epoch 0 (the trainer's own histogram field).
2. **Retrain under the law with ONLY this delta** (profile `hpr1_shape_rungs`, past_dilation 1, conv_a_dilation 1,
   cache a92e7d90… pinned by `--expected-cache-content-sha256`, `--init init_depths.pt --expected-init-sha256 <sha>`,
   seed 20260716, 60 epochs, QAT 0.5, λ 1.0, mps). Per-stage checkpoints on, `--resume-from` honoured. Launch
   through `tools/launch_detached_process.py --output-dir /Volumes/VertigoDataTier/pact/ddm_dpi1/launch_depths
   --nice 0 --done-receipt … -- <cmd>`. Cost anchor: hpr1's dil1 took ~52 s/epoch on this host (≈52 min).
   The host also runs ren1's three CPU resolves (15 threads); do not add CPU-heavy work beside them.
3. **Rail control first** (the falsifier): extend `experiments/ddm_hpr1_shape_price.py` with `control48` — re-encode
   move 48's shipped prior (frame-even already applied) through the rail and reproduce **179,111 B, sha
   d830edd3…** at Δ 0 (hpac 11,629 B, stream 118,896 B), twins agreeing. If the control does not pass, STOP.
4. **Price the candidate on the move-48 base**: treatment `retrain_depths_frame_even` = pack the new EMA QAT prior,
   apply ntb2's frame-even rounding exactly as `retrain_frame_even` does, encode the RLC1 token stream with the real
   coder over all 600 frames, twins, decoded field sha must equal a92e7d90… (output-lossless). Report the model leg
   (vs 11,629 B), the tail leg (vs 118,896 B), archive bytes vs 179,111 B, mean row depth at terminal, and the
   twin. Fire bar: net ΔS < −2e-5 ⇔ archive ≤ 179,080 B (−30.04 B at 6.658589531221714e-7 S/B).
5. **Held-out honesty**: the tail leg on 600 frames is in-sample for the prior; report the trainer's held-out
   estimate if the law exposes one, else say it does not, and cite tmx1's sample-to-field loss as the risk.
6. If it nets: stage SEAL INPUTS exactly as hpr1 did for move 47 (normal seal path: receiver unchanged ⇒
   inherits move 48's MEASURED t4_direct leg via pr18's behaviour digest; pr19 permits one inheritance of a
   measured leg): candidate runtime tree = move 48's tree with only the `hpac` member replaced, manifest
   regenerated from outside the tree and validated, literal census, raw identity n600 (one raw 2b762eba…),
   public smokes candidate + frontier, retention manifest with shas ≤ 8 GiB on the SSD tier. NO seal call,
   NO Modal, NO packet — MAIN runs `make_candidate_seal.py`, the CPU sibling, and the packet.
7. Memo `.omx/research/ddm_dpi1_hpac_depth_state_restored_warm_start_20260912.md`: the defect with receipts,
   epoch-0 histogram, terminal depths, control row, candidate rows, fire verdict, every boundary; serializer
   commits (two visible review passes per .py; `[no-triality] [p0-ledger-ok]`; NO co-author trailer ever).
   Checkpoint as `ddm_dpi1` via `tools/subagent_checkpoint.py`. Lane id
   `ddm_dpi1_hpac_depth_state_restored_warm_start_20260912` (claim via `tools/claim_lane_dispatch.py`).

## Boundaries (binding)
- Never edit `upstream/`, `submissions/semantic_joint_ctxmix/`, sealed trees, the frozen contract code
  (`src/tac/candidate_seal.py`, `src/tac/decode_wall_clock.py`), or the receiver. No Modal. No `authorize_*`, no
  `fire_modal_auth_eval.py`. Never lower a storage reserve. Local disk is not a storage tier: every payload on
  `/Volumes/VertigoDataTier/pact/ddm_dpi1/` (53 GiB free; reserve 40 GiB) with sha-verified retention.
- Do not change the law's other constants. The ONLY delta is the restored depth state. If you believe a second
  change is warranted, record it as a follow-on rung; do not run it here.
- Do not touch ren1's directories (`/Volumes/VertigoDataTier/pact/ddm_ren1`) or tmx1's.
- Read `docs/operating_manual_craft_handoff.md`; report MEASURED / DERIVED / INFERRED / ASSUMED explicitly.

## OPTIMAL FORM
- Reference form: cl2's law exactly as hpr1 ran it (`train/dil1` = the move-47 prior; config in its result.json),
  the real coder on 600 frames, the shipped receiver of move 48; every delta below is SCOPE-neutral: the single
  MECHANISM change is restoring a state the law's own source checkpoint carries (no toy bracket needed — it is the
  reference form with a lost input returned).
- Provenance pins: HEAD 221bc43c8; trainer sha 2ccd6153…; rail sha 448fbe6d…; hpr1 memo sha 26573f9b…; init.pt
  cf41112788757f877d8d729d4bd7900772817cda8e06f6fd7fcb0129dc1f8383; epoch_0634.pt 5007beae7af7789758092f12f49096e13692e2e59850c85eb4642cd6fad147ec;
  cache a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8; move 48 archive 179,111 B sha d830edd3…
  (`.omx/research/ddm_hpr1_20260911/v2/SEAL_ddm_hpr1_comp_even_on_refit_contest_cuda_v3.json` — read the full sha and
  tree path from it); hpr1 dil1 prior 12,262 B / move 48 hpac 11,629 B.

## Prior negatives accounted (operator 2026-08-15)
- tmx1 (today): a low-capacity refit LOSES (+20 B); in-sample gains vanish in transfer — report held-out or say none.
- hpr1 R1 confound: measure against the RIGHT control (move 48's prior through the same rail), never against an
  older pointer; the control PASSES at Δ 0 or nothing is priced.
- cl2 +0.446 secant: more CAPACITY does not pay; this rung adds none (values held at 20,416; depths only).
- ntb2 frame_quad LOSES on the refit prior: do not stack a second rounding.
- Container lottery sd 34.8 B: quote the margin over the fire bar in units of it.
- rp1 r2: bind every input by sha; the flag-vs-constant trap (`init_bits` stays 8.0 as a flag; the state overrides it
  through the checkpoint — prove which one the terminal depths descend from with the epoch-0 histogram).

Final message: the defect receipts, epoch-0 and terminal mean depths, control48 row, the candidate's three legs and
archive bytes with sha, fire verdict with margin in lottery units, seal-inputs path if staged, every boundary, and
the frontier line `composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)` unchanged (MAIN seals).
