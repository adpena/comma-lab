# ddm_nv1 — THE HULL MOVE: authority-confirm of the null-snap on B-control (task #796) — 2026-07-30

**POINTER HONESTY FIRST:** submittable **0.1910828242 [contest-CPU] UNMOVED**. Every number below is
**[macOS-CPU advisory]**, `score_claim=false`, `research_only`. The frozen CPU-torch SegNet ran (this arm
held the scorer slot ps1 freed); no Metal, no paid dispatch, no contest evaluator, no pointer move.
verdict_scope tags are the narrowest the receipts support. No AI attribution; commits are the operator's
alone. `[no-triality]` (build+measure apparatus + this graph leg) `[p0-ledger-ok]`.

## VERDICT (one line)
The owed authority-confirm of gc10 row-1's ×1.9 hull move is **NEGATIVE**: the null-snap **fails the
Δd_seg ≤ 2e-4 authority bar at EVERY depth** (even the shallowest, thr1, is +5.78e-4 = 2.9× the bar) and
**net-RAISES the additive contest score S at every depth** (min ΔS +0.0163 at thr1 → +1.46 at thr4). The
"×1.9 hull move" was a **product-invariant (c) artifact** under the **FALSIFIED** premise "d_seg holds ≤
0.0053"; measured d_seg rises to 0.008448 at the 117KB knee. The seg-aware / band-lemma reformulation
(gc10 sketch-1, QA80-field-gated) is **MEASURED-DEAD and DOMINATED** by the global snap. Root mechanism:
**reclaimable rate is co-located with seg-risk** (corr(|delta|-mass, QA80 flip-distance) = **−0.51**) — the
bytes are in the same dynamic/boundary cells where the flips are. **Bank EMPTY.** The snap's only surviving
form is the in-training delta group-sparsity force (F2), NOT a post-hoc export bank.

## STORES-CONSULTED (recall receipts; multi-pass grep; path[+sha])
- **gc10 memo** `.omx/research/ddm_gc10_hull_mover_convocation_20260730.md` (2647b1f080): row-1 (×1.9,
  distortion column OWED, Contrarian dissent binding), op-r 5 (117KB custody), op-r 6 (QA80 staleness),
  op-r 7/F2 (snap at export only; burn warm-starts un-snapped + delta group-sparsity ON), sketch-1
  (band-lemma-gated coder), §2b(h) backcast, §5 (row-1 = highest Δlog-c/hr IF confirm holds).
- **zb1 memo** `.omx/research/ddm_zb1_cheap_batch_20260730.md` (d3ff40bad5): S2 ν-audit (base SMEVR
  255,907 B, delta-zero frac 0.3959, null-snap curve q50 193,193 / q70 117,068 / q90 66,766; rate-side
  ONLY), QA80 field, S1 dress rehearsal (archive 438bc022, realized d_seg 0.005114475), hull chart
  (token-family {QA24 c=0.08815, B c=0.08835}).
- **ps1 memo** `.omx/research/ddm_ps1_pose_stage_20260730.md` (parent = B-control; pose WALLED d_pose
  20.4075; seg+rate floor S 0.6886; pose stream `pose_warp.stp` 6,633 B — carried unchanged, orthogonal
  to this seg×rate hull work).
- **B checkpoint** `/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/control/checkpoints/stage_seg_trunk_tau_final.npz`
  (sha **feba3b7f1fa34b52…**, ep440; cfg token_ste=round, levels=16, grid 24×32×4,
  cell_mask=qa24_grid_keep_mask_50 keep 0.5).
- **QA80 field** `/Volumes/VertigoDataTier/pact/ddm_zb1_qa80_field_20260730/` (600 npz exact_flip_distance).
- **product law** `tac.canonical_equations.product_c` = (100·d_seg)·(25·bytes/37,545,489) — VERIFIED on B
  (0.5114×0.17273 = 0.08834 = zb1's 0.08835).
- **coder** `experiments/ddm_r7_token_coder.py::encode_token_codes(codec="smevr")` (SHIPPED r7 coder);
  runtime `tac.optimization.ddm_tr1_runtime` (compile / render_frame1_camera_uint8 / parse_archive);
  verdict path `train_witness_realized_through_R_mlx.cpu_verdict_d_seg_argmax_batch` vs gt_n600 lstars.

## R0b — CODER CUSTODY (op-r 5, the 117KB closed-form-CDF trap check): PASSED
Re-derived the base token stream through the REAL SMEVR coder on B's cell-masked full codes:
**255,907 B EXACTLY** (drift 0.0 vs zb1's 255,907). The compiled base archive
sha256 = **438bc022fcd835ab…** — **byte-identical to zb1's dress-rehearsal archive**. The 117KB curve is
a REAL re-encoded coder byte count, not an entropy projection. Custody CLEAN.

## R0a — QA80 field staleness: used for co-location; direction robust
The QA80 field was computed on QA24-endpoint frames; parent is B (+40ep). QA24 realized d_seg 0.005278 vs
B 0.005114 (near-identical seg endpoints). I consumed the field only for the co-location DIRECTION (which
cells are seg-risky), and cross-checked with the **staleness-free GT-margin field** (gt_n600 margins,
parent-independent): both give the same sign and ranking (QA80 corr −0.51, GT-margin corr −0.25). The
co-location VERDICT is robust to field drift; an exact B-field re-measure would refine the precise
safe-byte fraction but not the verdict. verdict_scope: R0a satisfied for the co-location use.

## R1 — RATE LADDER re-derived through the REAL SMEVR coder (rungs = |signed residual| ≤ thr)
Snap = set code → temporal mode (delta→0) for nonzero deltas with circular residual |signed| ≤ thr. Matches
zb1's curve (thr1≈q50, thr2≈q70 knee, thr3≈q90) within 0.4–1.7% on bytes.

| thr | snapped % of kept-nonzero | SMEVR tokens B | composed counted B | rate_term |
|---|---:|---:|---:|---:|
| base | 0 | 255,907 | 259,407 | 0.17273 |
| 1 | 35.2% | 193,455 | 196,955 | 0.13114 |
| 2 | 66.1% | 116,611 | 120,111 | 0.07998 |
| 3 | 82.0% | 67,898 | 71,398 | 0.04754 |
| 4 | 91.9% | 34,619 | 38,119 | 0.02538 |

(non-token bytes held constant = 3,500 = renderer 3,284 + headers; zb1 §3 renderer FLAT.)

## R2 — DISTORTION AUTHORITY (the OWED column; Contrarian's binding half). MEASURED n600.
Every rung built into a real archive → `render_frame1_camera_uint8` → R/uint8 → frozen CPU-torch SegNet
argmax vs gt_n600 lstars. **Base parity: realized d_seg 0.005114475, |Δ|=1.74e-10 vs zb1** — path validated
byte-exact. Deterministic (no seed noise; thr1 reproduced 0.005692893 across two runs).

| thr | d_seg | Δd_seg | pass 2e-4? | rate_term | **c=seg·rate** | c ×base | **S_segrate=seg+rate** | **ΔS** |
|---|---:|---:|:--:|---:|---:|---:|---:|---:|
| base | 0.005114 | — | — | 0.17273 | **0.08834** | 1.00 | **0.6842** | 0 |
| 1 | 0.005693 | +5.78e-4 | **NO** (2.9×) | 0.13114 | 0.07466 | ×1.18 | 0.7004 | **+0.0163** |
| 2 | 0.008448 | +3.33e-3 | **NO** (16.6×) | 0.07998 | 0.06756 | ×1.31 | 0.9248 | **+0.2406** |
| 3 | 0.013939 | +8.82e-3 | **NO** | 0.04754 | 0.06627 | ×1.33 | 1.4414 | **+0.7572** |
| 4 | 0.021192 | +1.61e-2 | **NO** | 0.02538 | 0.05379 | ×1.64 | 2.1446 | **+1.4604** |

## THE DECISIVE REFRAME — c (product invariant) ≠ S (additive contest score)
`product_c = seg_term × rate_term` DROPS monotonically (0.0883→0.0538, ×1.64) as we snap deeper. **This is
a MULTIPLICATIVE ARTIFACT**: seg,rate < 1, so trading seg-up for rate-down can shrink the PRODUCT even while
the SUM grows. The **contest score S = 100·d_seg + √(10·d_pose) + 25·B/N is ADDITIVE** — and on S the snap
**LOSES at every depth**. The exchange is unfavorable and worsens with depth:

| thr | Δrate_term | Δseg_term | netΔS | unfavorable× |
|---|---:|---:|---:|---:|
| 1 | −0.0416 | +0.0578 | +0.0163 | 1.39 |
| 2 | −0.0928 | +0.3333 | +0.2406 | 3.59 |
| 3 | −0.1252 | +0.8824 | +0.7572 | 7.05 |
| 4 | −0.1473 | +1.6078 | +1.4604 | 10.91 |

gc10 row-1's derivation ("at d_seg ≤0.0053 ⇒ c 0.044–0.048") had ~correct ARITHMETIC (at 120KB, c-IF-d_seg-held
= 0.041) but a **FALSIFIED PREMISE**: d_seg does not hold (rises to 0.008448 at the 117KB knee). The gc10
row-1 falsifier ("Δd_seg > 2e-4 at q70 ⇒ re-derive shallower") FIRES — and "shallower" cannot save it:
even thr1 fails, and every depth loses on S. This confirms zb1 §3 (B has ENTERED the token marginal-price
knee): reclaiming rate now costs more d_seg (×100) than it saves in bytes (×25/N).

## SEG-AWARE / BAND-LEMMA REFORMULATION (gc10 sketch-1) — MEASURED-DEAD, DOMINATED
Protect the riskiest cells (low QA80 flip-distance), snap ALL nonzero deltas in the rest:

| admission | snapped % nz | tokens B | d_seg | Δd_seg | c ×base |
|---|---:|---:|---:|---:|---:|
| global thr1 (magnitude) | 35.2% | 193,455 | 0.005693 | +5.78e-4 | ×1.18 |
| seg-aware, snap safest 25% | 19.1% | 206,956 | 0.006854 | +1.74e-3 | ×0.92 |
| seg-aware, snap safest 50% | 41.9% | 148,337 | 0.012399 | +7.28e-3 | ×0.70 |

**global-thr1 DOMINATES seg-aware-p75 on BOTH axes** (fewer tokens 193k<207k AND lower Δd_seg 5.78e-4 <
1.74e-3), and both c-worsen. Magnitude-ranking beats cell-safety-ranking: the smallest |signed|=1 deltas are
the smallest rendered perturbations everywhere; snapping ALL magnitudes even in "safe" cells produces large
RGB changes that flip pixels (the band lemma's amplitude≤flip-distance guard is violated by mode-snap
amplitude). The field-gated coder adds nothing encode-side. sketch-1 falsifier ("field-gated depth ≤
global-q depth at matched Δd_seg") FIRES.

## ROOT MECHANISM — reclaimable rate is CO-LOCATED with seg-risk
Per kept-cell (n=384): corr(|delta|-mass, QA80 flip-distance) = **−0.5126**; corr(|delta|-mass, GT-margin)
= **−0.2481** (staleness-free cross-check, same sign). The SAFEST 50% of cells holds only **41.2%** of the
reclaimable |delta|-mass (GT-margin: 46.2%); the safest 25% holds **18.1%**. **The bytes live where the
flips live** — the dynamic/boundary cells carry both the temporal deltas (rate) and the small-margin pixels
(seg-risk). No encode-side admission escapes this: reaching the 117KB knee (55% token reclaim) necessarily
snaps risky cells, and even "safe" cells are not flip-free. This is a STRUCTURAL property of a seg-native
TR1 vehicle at this rate operating point.

## R3 — VERDICT: TYPED NEGATIVE; BANK EMPTY
No rung passes the Δd_seg ≤ 2e-4 authority bar; every rung (both admissions) net-RAISES the additive
contest S. **No snapped archive is banked** as a frontier candidate (banking any rung would raise S; the
composed S-with-pose is separately pose-walled at ~14.97 per ps1). The ps1 pose stream (6,633 B) is carried
unchanged and is orthogonal to this seg×rate result.
- **verdict_scope: FORMULATION** — both admissions closed (global |signed|-ranked; QA80-field-gated
  seg-aware) on the **instance** B-control seg-native parent. Per the ladder, two formulations ≠ family
  death: the null-snap / encode-side nullity **FAMILY stays LIVE at family scope**, but with a MEASURED
  MECHANISM (co-location corr −0.51) that predicts family-wide difficulty on any seg-native parent at this
  operating point.
- **F2 re-reading (strengthened):** there is nothing positive to "bank at the export surface" — the
  export-side snap is a net loss standalone. The snap's ONLY viable form is the **in-training delta
  group-sparsity force** (the ax1 lever, F2's guard) that lets a burn FIND a low-d_seg solution that is
  ALSO delta-sparse — fused, never post-hoc. F2's guard is thus the PRIMARY mechanism, not a fallback.

## REFORMULATION QUEUE (the surviving legs; defer-at-source)
1. **[HELD → burn-3] Post-snap RE-CONTINUATION / multiplicative-composition thesis (gc10 R4 / F2).** The
   ONLY path where the snap matters: warm-start UN-snapped from B, delta group-sparsity ON from step 0, price
   with the snapped coder, and test whether the burn re-earns d_seg back to ≤0.0053 AT the snapped rate. My
   result raises the bar: the burn must re-earn ALL the d_seg the snap costs PLUS more, while holding sparse
   exactly the boundary-cell deltas it needs for d_seg (co-location makes this doubtful — but it is the
   burn's to measure, not assertable here). This is a BURN (heavy) — out of the scorer-slot / fleet-cap
   scope; routed to burn-3 design.
2. **[instance→family probe] snap on a lower-rate OR pose-conditioned parent.** Co-location is measured at
   B's operating point (rate 0.173); a parent already deeper on rate, or one where the boundary cells carry
   pose-legible signal, may price the exchange differently. Cheap $0 rate-side pre-screen possible.
3. **[weak, deprioritized] per-class snap.** Co-location predicts weakness (rate & risk co-located within
   classes too); run only if 1–2 both stall.

## WIRE-IN / ROUTING (results → system intelligence)
The gc10 §2 fork is sharpened, not overturned: "burn-3 must (a) reclaim S2 free rate AND (b) move the hull
in-loop." My finding kills (a) as a STANDALONE precondition — **the rate is not free; it costs d_seg at a
1.4–11× unfavorable exchange, and post-hoc reclaim (either admission) LOSES on S.** (a) and (b) are not
separable: rate reclaim must be FUSED into the burn (delta group-sparsity in-loop, F2) so a single descent
finds a solution that is simultaneously low-d_seg and delta-sparse. The Schmidhuber-LEAD "row-1 = highest
Δlog-c/hr" ranking is retracted at the score level: it was Δlog-**c**, and c is not the score. costate SENSE
+ burn-3 waterfill should consume the co-location field (rate∥risk, corr −0.51) as a design constraint.

## DELIVERABLE CUSTODY (/Volumes/VertigoDataTier/pact/ddm_nv1_20260730/)
- `nv1_consolidated_receipt.json` (schema ddm_nv1_hull_move_consolidated.v1 — the authoritative table);
  `nv1_rate_table.json` (R0b custody + rate curve); `nv1_dist_table.json` (thr3/4 dist); `nv1_segaware_table.json`.
- `dsegs_thr{1,2,3,4}.npy` + `dsegs_segaware_p{50,75}.npy` (per-pair realized d_seg, n600).
- Harness `nv1_hull_move.py`: session scratchpad (one-off measurement scaffold, rebuildable — not committed;
  reuses committed runtime + r7 coder + verdict path). Bulk on SSD (certify-or-block: rebuildable from B
  checkpoint feba3b7f + gt_n600 + committed code).

pointer 0.1910828242 [contest-CPU] UNMOVED  ·  [no-triality] [p0-ledger-ok]
