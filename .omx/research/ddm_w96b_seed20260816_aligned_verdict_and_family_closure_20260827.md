# ddm_w96b seed-20260816 aligned verdict + FAMILY CLOSURE — both seeds <2×; aligned-config W96 closes at chartered scope

Date: 2026-08-27 · Author: MAIN · Status: MEASURED (n60 evenly-strided, S1E instrument)
Axis: [Darwin-mps training / macOS-CPU frozen-scorer advisory] — score_claim=false,
promotable=false, contest_eval_run=false. No exact-authority claim anywhere in this memo.
verdict_scope: FORMULATION — the chartered aligned-config W96 family (w96a charter: aligned
expected-flip-margin law × WD3 trainer × 65-ep window × gb1-body Stage-A adapter), measured
at BOTH registered seeds. Not a kill of the aligned seg LAW (vindicated ~2× on its own axis,
both seeds) and not a kill of every trained-renderer object (reactivation criterion below).

## STORES CONSULTED

- Charter + fork rule: `.omx/research/ddm_w96a_aligned_config_renderer_window_20260826.md`
  (≥5× at any seed → LIVE n600 buy; <2× at BOTH seeds → CLOSURE).
- Seed-1 verdict: `.omx/research/ddm_w96b_seed20260815_aligned_verdict_20260827.md`
  (composed +0.1310670, ratio 1.186×).
- OFF baselines (same instrument, same constants, prior burns): s20260815 ep65
  +0.1554134085557307 · s20260816 ep65 +0.19887521337211275 · s20260816 best/ep75
  +0.14815737243836.
- Instrument: `tools/s1a_off_floor_adjudicator.py` (S1E composed-delta; GB1_HARD_D_SEG=
  0.00020139 · GB1_D_POSE=6.37e-6 · GB1_RENDERER_BYTES=30856 · RATE_PER_BYTE=
  6.658589531221714e-7).
- fb2 route table: `.omx/research/ddm_fb2_route_table_gb1_20260826.md` + route-2-dead
  adjudication `.omx/research/ddm_bs3_route2_dead_adjudication_20260827.md`.
- sy2 composition law (m148): a closed leg reopens only when another leg CHANGES ITS OBJECT.

## RECEIPTS (retained, ALWAYS-KEEP-THE-PAYLOAD)

- `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/training/aligned_seed_20260816/W96_flattened/TRAIN_RESULT.json`
  (complete=true; stage_checkpoint sha 8a51328640914585268382d3ad590e7fe1a5e2f03d736b03c307ee72f98e9e65,
  1,090,867 B) + `.../evaluations/epoch_0065_n60.json` (archive 180,181 B,
  archive_repeat_byte_identical=true, parse-back packet_exact=true).
- Fire order r3: `aligned_seed_20260816_chunk30_20260827`, config_sha256
  13163dc82514aaad1f74c898e4ae00d8d7968e9e7d3b8970069a381157041915 (chunk_pairs 60→30);
  run rc=0 at 7,372 s, 65/65 epochs.
- r1/r2 incident record: two launches died <3 min at the MPS watermark (107.24 / 107.46 of
  107.52 GiB cap) inside the 60-pair full-autograd materialization chunk; fresh-allocator
  hypothesis REFUTED by r2. Cure = chunk_pairs 30 (halves the gradient-chunk live set),
  within the operator 116 GiB ceiling (m79). Chunk-60 stage dir retain-MOVED with
  RETENTION_CERT (`.../stage_controllers/stage_04_from_epoch_0000.chunk60_oom_retained_20260827/`).
  r1/r2 are instances 2–3 of the #1306 Metal-blind watcher class (ps-RSS ~7 GiB while
  Metal held 107+ GiB).

## THE MEASURED ROW (seed 20260816, aligned, ep65, n60 evenly-strided)

- hard_d_seg **0.0004010518314316869** · d_pose **0.0011792422737926245** · W96 packet
  **38,847 B** (uniform_int4_degenerate — identical to seed-1's packet size).
- seg_damage = 100·(0.00040105183 − 0.00020139) = **+0.0199662**
- pose_damage = √(10·0.00117924227) − √(10·6.37e-6) = 0.1085930 − 0.0079812 = **+0.1006117**
- rate_credit = (30856 − 38847)·6.658589531e-7 = **−0.0053209**
- **composed_delta = +0.1258988** vs renderer break-even.

## GATE ARITHMETIC → FAMILY CLOSURE

- seed-2 ratio vs OFF ep65 (matched window): 0.19887521 / 0.1258988 = **1.5796×**
- seed-2 ratio vs OFF best/ep75: 0.14815737 / 0.1258988 = **1.1768×**
- seed-1 ratio (prior memo): **1.186×**

Both seeds land in the <2× closure zone; the ≥5× LIVE gate failed at both. **Per the
chartered fork rule the aligned-config W96 family CLOSES.** The n600 buy does not fire.

## DECLARED INSTRUMENT DELTA (chunk-30, seed-2 only)

Seed-2 ran the materialization at chunk_pairs=30 (r1/r2 OOM cure); seed-1 ran chunk-60.
Contract safety verified at source before firing: `_birth_contract` (runner :2532) excludes
chunk_pairs; `_resume_config_identity` (:1374) binds only training-loop checkpoints;
`controller_binding` (:2120) embeds it → the chunk-60 stage dir was retain-moved, never
mixed. The chunked accumulation `(total/ceil(N/chunk)).backward()` is a slightly different
decomposition of the means-based objective (nonlinear-in-mean terms: √(10·pose_mse),
relu(mean−ceiling)); training physics unchanged at batch_pairs=1; the ep65 n60 verdict
instrument is no_grad and chunk-invariant. Outcome evidence the delta is benign: seed-2's
seg endpoint reproduces seed-1's within 0.8% across the chunk change.

## MECHANISM (consistent across both seeds — what the family closure MEANS)

1. **The aligned seg law WORKS and is seed-robust**: hard_d_seg 3.98e-4 / 4.01e-4
   (0.8% spread) vs OFF ~8.15e-4 — a ~2.03× seg-axis improvement, both seeds. Partial
   #1251 payment: the aligned config's seg endpoint is a narrow distribution.
2. **Pose is the structural failure, ~80% of the composed delta both seeds**: d_pose
   1.30e-3 / 1.18e-3 = 204× / 185× gb1's 6.37e-6. The W96 renderer trains pose in-loop
   (pose_exact_nonlinear, pose_start_step 0) and still lands 2 orders above the frontier
   carrier — the renderer-replacement object does not CARRY pose at this
   capacity/window, consistent with #1222 (PoseNet scores the FRAMES; the renderer is
   the pose carrier) and #1230 (pose = 65.3% of W72 damage).
3. **The rate leg is NEGATIVE at W96**: 38,847 B packet vs the 30,856 B renderer it
   would replace. Under m124's two-way demand law this family ADDS bytes at this width.
4. Training health clean both burns (65/65, ~2h wall each, no jetsam after the chunk-30
   cure).

## ROUTING (fb2 route table re-adjudication)

- Route 1 (aligned W96 R+M) — **CLOSED** (this memo). The named dominant failure term is
  POSE. Per sy2, reopening requires a changed OBJECT: a pose-carrying variant (e.g. a
  renderer co-trained with/against the carrier ξ-stream, or a solve-seeded pose head) —
  not more seg work, not more seeds, not more width.
- Route 2 (born-small resolved carrier B+C) — closed prior
  (`ddm_bs3_route2_dead_adjudication_20260827.md`, ~66× distortion-over-credit).
- Route 3 (rb1 four sealed configs, D56/F64 × 2 seeds, SAME aligned law at SMALLER
  widths) — **HELD FOR RE-DERIVATION before fire** (zero-grav-pull m146): both W96 rows
  bound its outcome. Seg/pose at D56/F64 are capacity-monotone ≥ W96's (D56/F64 already
  measured negative under the OFF law — `negative_confirmed_arms` in the w96a config);
  best-case rate credit at packet→0 is +0.0205, swamped ~5× by the measured W96 pose
  term alone. Unless rb1's own fire-order arithmetic names a term outside this bound,
  route 3 closes by arithmetic and the Metal slot goes to the pose-carrying object
  change. Adjudication is the next action after this memo lands; rb1 configs also carry
  chunk_pairs=60 (the r1/r2 OOM path) and would need the chunk-30 recompile if fired.

## REACTIVATION CRITERION

An aligned-family trained renderer re-enters ONLY as a pose-carrying object change
(sy2): measured d_pose within ~4× of gb1's 6.37e-6 at any advisory scope n≥60, OR a
composed_delta within the campaign gap (+0.028) of renderer break-even. No re-fire on
seg-axis improvement or seed/width variation alone.

Pointer unmoved: gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600].
