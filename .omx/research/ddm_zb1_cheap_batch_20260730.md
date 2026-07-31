# ddm_zb1 — CHEAP-BATCH ARM (fu1 rank-N): QA80 field pass + 6 $0 dispositions (2026-07-30)

**Task #792 · the fu1 cheap-batch arm.** Pointer honesty FIRST: submittable
**0.1910828242 [contest-CPU] UNMOVED**. Every number below is **[macOS-CPU advisory]**,
`score_claim=false`, `research_only`. verdict_scope tags are the narrowest the receipts support.
No AI attribution; commits are the operator's alone.

## STORES-CONSULTED (recall receipts; multi-pass grep, path[+sha])
- **dw1 memo** `.omx/research/ddm_dw1_qa75_distill_window_20260730.md` (67cfa5e685): finishing-distill
  REFUTED (formulation); B-control (plain 40ep continuation) = NEW BEST realized anchor.
- **dw1 verdict** `/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/verdict.json`: B_control
  endpoint_dseg_n600 **0.005114661**, endpoint_bytes 259,407, S 0.6842; A_distill a1_realization_gap
  refuse; C_head_relax advisory-non-deployable.
- **field-pass harness** `tools/ddm_b2b_segnet_field_pass.py` (4bdd72a2f7): injectable frozen CPU-torch
  SegNet → top2 → exact_flip_distance; ≤120/chunk (#205 law); qa80_burn source = (N,H,W,3) uint8 npy.
- **QA24 endpoint archive** `/Volumes/VertigoDataTier/pact/ddm_782_qa24_endpoint_20260730/qa24_endpoint_archive.bin`
  (sha **e7640dee9c3cf41d…**); realized verdict d_seg 0.0052766 via `tools/pb1_receiver_realized_verdict.py`.
- **B checkpoint** `…/ddm_dw1_20260730/control/checkpoints/stage_seg_trunk_tau_final.npz` (ep440;
  ema::tokens_base (24,32,4) + ema::tokens_delta (600,24,32,4); cfg coder=smevr, D16/c4/L16,
  cell_mask=qa24_grid_keep_mask_50.npy keep 0.5).
- **pn1 S1/S2 charter** `.omx/research/ddm_pn1_pantheon_of_pantheons_completion_20260728.md` (§2/§3/§4).
- **gc9** `.omx/research/ddm_gc9_from_here_convocation_20260730.md` (row-8 hull; row-3 product law) +
  canonical eq `tac.canonical_equations.ddm_gc9_seg_rate_product_law_20260730:product_c`.
- **FEED-wr1gb** `sub015_DAG_*:25449` (wr1 Gate-B truncation REFUTED, instance scope).
- **cs1 harvest** `.omx/research/ddm_cs1_consolidation_harvest_20260728.md` + rc1 salvage note
  `/Volumes/VertigoDataTier/pact/ddm_fu1_rc1_salvage_20260730/salvage_note.md`.
- **runtime** `tac.optimization.ddm_tr1_runtime` (compile_archive_from_checkpoint cell-mask-aware;
  render_frame1_camera_uint8) + `experiments/train_tr1_partition_renderer_mlx` (token_stream_bytes_smevr).

## §1 (ITEM 1) QA80 EXACT FLIP-DISTANCE FIELD PASS — DONE, MEASURED n600
The scorer job (ran FIRST; one scorer job at a time). Burn frames were NOT materialized (blocker the
harness documents), so I materialized them via the receiver path: `render_frame1_camera_uint8` over the
QA24 endpoint archive (e7640dee, identity re-hashed) → 600×(874,1164,3) uint8 npy
(sha **937f96727267bf15…**, 1,831,204,928 B, render 94.5s). Then the real-SegNet field pass
(`--frame-source qa80_burn --field-kind exact_flip_distance --device cpu`), detached, rc=0.

- **Output** `/Volumes/VertigoDataTier/pact/ddm_zb1_qa80_field_20260730/`: 600 `pair-NNNNNN.npz`
  {`exact_flip_distance` f32 (384,512), `winner` u8, `runner` u8}; `field_pass_manifest.json`
  (schema ddm_b2b_segnet_field_pass.v1, authority local-CPU-torch-advisory, score_claim False).
- **CUSTODY: per-pair sha256 VERIFIED 600/600** against the manifest. + `burn_frames_manifest.json`
  (frames npy sha + archive lineage).
- **Consumer budget (QA83 class_field_photo margin-slack + ea1-N3 burn-3):** flip-distance
  q50 median **1.8181**, q05 median **0.4302**, q95 median **2.7296**; tightest pixels ~**1.9e-6**
  (zero photometric budget at the boundary — the pp1 band lemma correctly flags where amplitude→0).
- verdict_scope: MEASURED producer field; consumers OPEN.

## §2 (ITEM 2) gc9 row-4 S2 ν-AUDIT (rate-side, $0) on the NEW-BEST anchor B
Ran the pn1 §3/§4 RATE-SIDE half on B's EMA tokens (`s2_nu_audit_B.json`). SCOPE (honest): the
|g| gradient-sensitivity map (MLX Σ-margin backward) and the Δd_seg≤+2e-4 safety confirm per q are
NOT run — they need an authority-scorer slot (QA80 occupied it). So ν here is the RATE-knee candidate,
not the authority ν. **verdict_scope: rate-side-only.**

- Base SMEVR token stream (shipped r7 coder, cell-keep-50) = **255,907 B** = **1.111 b/q** over the full
  1,843,200 quanta (2.221 b/q over the 921,600 kept). NOT G4-130KB-feasible as-is.
- **delta-zero fraction (kept) = 0.3959** — 39.6% of kept temporal deltas are exactly zero → the coder
  pays ~nothing for them. This is the MEASURED gauge/fiber tb1 design #3 predicted, but PARTIAL: SGD/EMA/
  STE moved ~60% of kept deltas OFF zero, so the zero-init gauge is **NOT fully verified** (steep, not
  flat — pn1's falsifier direction "measured free rate").
- **null-snap bytes(q) curve (|delta|-ranked, SMEVR-repriced):** q50→193,193 B; **q70 (thr=2 levels,
  snap 81.5% of kept deltas)→117,068 B = 0.508 b/q — BELOW the G4 0.578 ceiling**; q90→66,766 B.
- **TYPED VERDICT (burn-3 precondition):** G4 130KB is RATE-REACHABLE — the null-snap curve crosses
  the G4 b/q ceiling at ~q70, landing G4-feasibility right INSIDE pn1's predicted measurable window
  **ν∈[0.55,0.75]**. The OPEN falsifier (still the burn-3 precondition, needs a scorer slot): is the
  required snap depth (~q70, threshold 2 levels) d_seg-SAFE (Δd_seg≤+2e-4)? Run the |g|-ranked null-snap
  + realized-d_seg confirm to convert ν_rate→ν_authority.

## §3 (ITEM 3) QA34 $0 KNEE-BOUNDARY PRICE READ — DONE
Read burn (bc1 ep9-399) + B window (dw1 control ep404-440) a1_gate telemetry.
- **Renderer stream FLAT 3,284 B** the whole burn (zero marginal price) → ALL rate is tokens; the
  renderer↔token divergence is PERMANENT (renderer at floor, nothing to rebalance).
- Knee is INTERNAL to the token stream: two `at_knee` quant-anneal RESETS (ep~49 −24KB; ep~199 −122KB
  token bytes; d_seg re-descends cheaper after each).
- Burn endpoint marginal token price = **478 B per 1e-4 d_seg** — still productive (burn ended BEFORE
  the token-price knee, consistent with "E2 NOT converged").
- **B continuation price RISING 444→1332 B/1e-4 WITH bytes growing (250.7→255.9KB)** + intermittent
  net-negative intervals → **B has ENTERED the token marginal-price knee**; plain continuation past
  ~ep430 is diminishing returns.
- CONSEQUENCE: burn-3 should reclaim S2 free-rate (§2) and/or switch to a correction-band lever, NOT
  plain-continue. verdict_scope: measured on these two runs' gate ledgers.

## §4 (ITEM 4) pn1 S1 Stage-A DRESS REHEARSAL — DONE, DEPLOY PARITY CLEAN
FIRST end-to-end run of the export chain on the CURRENT best state (B ep440):
`compile_archive_from_checkpoint` (cell-mask-aware) → deterministic stored ZIP + parse/re-emit verify →
numpy receiver `render_frame1_camera_uint8` (E1) → frozen CPU-torch SegNet argmax verdict.
Receipts `/Volumes/VertigoDataTier/pact/ddm_zb1_s1_dress_rehearsal_20260730/`.
- Compiled archive **sha 438bc022fcd835ab…**, 360,735 B (research packet; the SMEVR-coded counted-rate
  ledger is 259,407 B — dress rehearsal validates d_seg deploy parity, byte-close to shipped SMEVR is a
  separate step).
- **Realized d_seg through the full export chain = 0.005114475** vs dw1's B endpoint 0.005114661 →
  **DEPLOY PARITY 1.86e-7** (3 orders below the pn1 row-9 1e-4 concern band). The pn1 row-3 falsifier
  ("Stage-A d_seg disagrees beyond drift band ⇒ deploy-parity bug") did **NOT** fire — parity CLEAN.
- Wall: compile 1.15s + verdict 150.3s ≪ 1800s → ample 30-min headroom for the (deferred) Stage-B
  contest-CPU flight. verdict_scope: dress-rehearsal receipt, NOT a score claim; advisory [macOS-CPU].

## §5 (ITEM 5) gc9 row-8 (d_seg, rate) CONVEX-HULL CHART — DONE
`gc9_row8_hull.json` + `.png` (+ token-family inset). Consumes `product_c`. Rows: burn-1 (composed,
c=0.14764), QA24 (token-SMEVR, c=0.08815), knee-B (staged, c=0.11640), **B (token-SMEVR, c=0.08835)**,
pj1 warm_l2 photometric fit (archive.zip, c=**10.70** — the confound corner, dominated).
- **BYTE-BASIS CAVEAT (surfaced):** the gc9 anchors MIX bases (burn-1 composed vs QA24/B token-SMEVR);
  rate is only apples-to-apples within a basis → the **token-family hull** is the live-vehicle frontier.
- **HEADLINE:** token-family hull = **{QA24, B}** at iso-c **~0.088**. **B did NOT move the hull** — it
  slid ALONG the ~0.088 iso-c contour (traded +8,509 B for −1.6e-4 d_seg; neither dominates). pj1's
  photometric fit is dominated at c=10.70. **The burns must MOVE the hull (c below ~0.088), not fill it.**
  This corroborates §2 (free rate to reclaim) + §3 (B past the productive-descent knee). Consumer:
  costate SENSE + burn-3 waterfill. (B is a candidate new product_c anchor for a future .py landing —
  NOT appended here, out of this arm's scope.)

## §6 (ITEM 6) QA18 HONEST CLOSURE — DONE
Ledger row QA18 → **CLOSED-MOOTED**: consumer wr1 is dead (FEED-wr1gb refuted the Gate-B
sensitivity-weighted truncation QA18's zero-flip-cell action fed). The underlying nullity question is
SUPERSEDED-and-better-answered by this arm's §2 S2 audit (39.6% delta-null MEASURED on B, receiver-
realized — the vehicle-native "zero-flip cells"). **verdict_scope: formulation** (the post-hoc
zero-flip-cell→wr1-truncation formulation is mooted; the nullity FAMILY stays LIVE in S2/burn-3
encode-side null-snap).

## §7 (ITEM 7) cn1 #726 / cn2 #727 RE-GATE — DONE
Ledger row QE03 gate re-pointed from the dead rc1 branch (retired 2026-07-30; `clwt/rc1_*` had ZERO
commits ahead of main, sole content = an unrun survey script; salvaged certify-or-block at
`/Volumes/VertigoDataTier/pact/ddm_fu1_rc1_salvage_20260730/`) → **cs1 07-28 harvest disposition table**,
which subsumed rc1's recovery mission. cs1 row: the PDW1 fp32 module set is **superseded-by-main** (5
files already on main HEAD via 237b955ef7; 4 byte-identical; committing the stale draft would REGRESS;
NO strand owed). Net: cn1/cn2 recovered-signal is already-on-main.

## Wire-in / synthesis (results → system intelligence)
The three cheap reads CONVERGE on ONE burn-3 directive: **B is a hull-filler, not a hull-mover.**
§3 (B past the token-price knee, price rising + bytes growing) + §5 (B slid along iso-c 0.088, did not
move it) + §2 (39.6% delta-null + rate-reachable G4 at ν~0.70) say plain continuation is exhausted;
burn-3 must (a) reclaim the S2 free rate on the ENCODE side (authority ν confirm is the one owed scorer
step) AND (b) move the hull via an in-loop lever (correction band / QA80 pose-legibility-from-birth),
never post-hoc deletion (FEED-wr1gb, instance-refuted). QA80's flip-distance budget (§1) is the
per-pixel amplitude the burn-3 margin-slack photometric stream may spend at zero seg cost.

## Deliverable custody
- QA80 field: `/Volumes/VertigoDataTier/pact/ddm_zb1_qa80_field_20260730/` (600 npz + manifest, sha 600/600;
  burn_frames_f1_camera.npy sha 937f9672…; s2_nu_audit_B.json; gc9_row8_hull.json + .png).
- S1 dress rehearsal: `/Volumes/VertigoDataTier/pact/ddm_zb1_s1_dress_rehearsal_20260730/`
  (B_compiled_archive.bin sha 438bc022…; compile_receipt.json; realized_verdict/; dress_rehearsal_receipt.json).
- Scratch materializer/audit/hull .py: session scratchpad (one-off; not committed — rebuildable).

pointer 0.1910828242 [contest-CPU] UNMOVED  ·  [no-triality] [p0-ledger-ok]
