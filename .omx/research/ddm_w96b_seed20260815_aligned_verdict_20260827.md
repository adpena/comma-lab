# ddm_w96b seed-20260815 aligned-W96 verdict — ≥5× gate FAILS at 1.186×; seed-20260816 decides the fork

Date: 2026-08-27 · Author: MAIN · Status: MEASURED (n60 evenly-strided, S1E instrument)
Axis: [Darwin-mps training / macOS-CPU frozen-scorer advisory] — score_claim=false,
promotable=false, contest_eval_run=false. No exact-authority claim anywhere in this memo.
verdict_scope: INSTANCE — one seed (20260815), one window (65 ep), one config (the w96a
aligned expected-flip-margin law). The family fork is chartered two-seed; this is seed 1 of 2.

## STORES CONSULTED

- Charter + fork rule: `.omx/research/ddm_w96a_aligned_config_renderer_window_20260826.md`
  (aligned/OFF ratio ≥5× at any seed → LIVE n600 buy; <2× at BOTH seeds → CLOSURE).
- OFF baselines (same instrument, same constants, prior burns): s20260815 ep65
  **+0.1554134085557307** · s20260816 ep65 +0.19887521337211275 · s20260816 best/ep75
  +0.14815737243836.
- Instrument: `tools/s1a_off_floor_adjudicator.py` (S1E composed-delta; constants
  GB1_HARD_D_SEG=0.00020139 · GB1_D_POSE=6.37e-6 · GB1_RENDERER_BYTES=30856 ·
  RATE_PER_BYTE=6.658589531221714e-7).
- Receipts (retained, CAS-verified):
  `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/training/aligned_seed_20260815/W96_flattened/TRAIN_RESULT.json`
  (complete=true, 65/65 epochs, stage ckpt `wd3_stage_end_epoch_0065.pt` sha
  856707a852225e35765318dca181eea5501d0cf826a62cadf603a8114d965349) +
  `.../evaluations/epoch_0065_n60.json`.

## THE MEASURED ROW (seed 20260815, aligned, ep65, n60 evenly-strided)

- hard_d_seg **0.00039800009108148515** · d_pose **0.0013011128176003695** · W96 packet
  **38,847 B** (allocation uniform_int4_degenerate).
- seg_damage = 100·(0.00039800009108 − 0.00020139) = **+0.0196610**
- pose_damage = √(10·0.0013011128) − √(10·6.37e-6) = 0.1140663 − 0.0079812 = **+0.1060851**
- rate_credit = (30856 − 38847)·6.658589531e-7 = **−0.0053209** (packet is byte-BIGGER
  than gb1's renderer reference — a rate PENALTY, not credit)
- **composed_delta = +0.1310670** vs renderer break-even.

## GATE ARITHMETIC

aligned/OFF ratio (same seed, same epoch, same instrument):
0.1554134085557307 / 0.1310670 = **1.186×**. The ≥5× LIVE gate FAILS. The row lands in
the <2× closure zone; per the charter, CLOSURE requires <2× at BOTH seeds — seed-20260816
is now the deciding measurement.

## MECHANISM DECOMPOSITION (what worked, what dominates)

1. **The aligned seg law WORKS**: hard_d_seg 3.98e-4 vs OFF-same-seed 8.15e-4 — a 2.05×
   seg-axis improvement. The expected-flip-margin objective did exactly what CE1's
   derivation predicted on its own axis.
2. **Pose dominates the composed delta**: +0.1061 of the +0.1311 (81%). d_pose 1.30e-3 is
   204× gb1's 6.37e-6. Pose descended 14.6× inside the window (0.019 → 0.0013) and was
   still unconverged at ep65 — a window-length question, but the charter's window is fixed.
3. **The rate leg is NEGATIVE**: the 38,847 B W96 packet exceeds the 30,856 B renderer it
   would replace. Under m124's two-way demand law this family currently ADDS bytes.
4. Training health clean: 65/65 epochs, ~2h wall, peak RSS 7.96 GiB, expected-flip margin
   4.436 → 2.044 (54%), no jetsam/watermark events (r7 watermark-pair fix held).

## FORK STATE + ROUTING

- Seed-20260816 aligned burn **LIVE** (launcher counter 680, supervisor pid 53976, trainer
  53986; fire order `aligned_seed_20260816_authorized_20260827`, config_sha256
  9a277e6880dc0dd8c871d1704aa0fe48f81e5790297079f77a8e9900e01f37f6; done receipt
  `w96b_aligned_seed_20260816`).
- At its endpoint: same S1E arithmetic on ITS ep65 n60 receipt. ≥5× → LIVE branch (n600
  buy). <2× → the aligned-config W96 family CLOSES at chartered scope, and fb2 route 1
  re-adjudicates with pose named as the dominant failure term (the seg law itself stands
  vindicated 2.05× — the sy2 object-change candidate for any reopening is a
  pose-carrying variant, not more seg work).
- Route 3 (rb1 four sealed configs) queues behind this Metal slot regardless of branch.

Pointer unmoved: gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600].
