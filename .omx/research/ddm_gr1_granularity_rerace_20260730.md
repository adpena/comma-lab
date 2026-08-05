# ddm_gr1 — TOKEN-GRANULARITY re-race on the pfs1 D1 dr7t lattice (ledger QA07/QA08/QA11/QA24)

UTC 2026-07-30 · axis **[macOS-CPU advisory]** (bytes = real SMEVR coder; d_seg = realized
render + frozen CPU SegNet argmax vs GT) · actuation **NONE** · score_claim **false** ·
promotion_eligible **false**. NON-PROMOTABLE. No scorer promotion, paid dispatch, archive
promotion, live-run mutation, or pointer mutation ran.

## Pointer honesty (operating manual §7 — the END first)

**The exact frontier did NOT move. 0.1910828242 [contest-CPU submittable custody] is UNMOVED.**
Effective bar 0.172 (official displayed best). This unit is a MEASURED rate-lever re-race; it
produced no byte-closed exact contest row. Every number below is `[macOS-CPU advisory]`; none is
a contest score. Stated plainly per the means/ends firewall.

## The question (co9 fired granularity_race_duty ARMED→DUE)

co9 round-9 (`ddm_co9_organ_round9_20260730.md`) re-parented the costate band onto the REAL
in-band burn endpoint (tb1 0.00389011, the pfs1 D1 live base) and fired the co8-pre-registered
`granularity_race_duty` ARMED→DUE. The token alphabet/granularity is the model class the rate
now lives in — the lossless coder is SATURATED (QA08/xi1: ideal 3-way mixing ceiling is only
1,617 B below SMEVR's realized 555,836 B). So re-race the DESCRIPTION granularity: does a finer
grid and/or per-token sensitivity-ordered precision allocation (QA11 law: 35× sensitivity
spread, 27% exact-zero grads → "continuous log-bit dominates the rung ladder") buy a lower
(bytes, d_seg) operating point through the REAL decode?

## STORES CONSULTED (recall-first, multi-pass)

- `CLAUDE.md`/`AGENTS.md` (NO-FAKE; measurement-first; verdict-scope ladder; #404 axis ratios;
  ONE-n600-at-a-time; serializer/patch-file; generic-triple + constants-are-poison +
  meet-it-where-it-is), `docs/operating_manual_craft_handoff.md`.
- **THE deferral ledger** `ddm_deferral_queue_ledger_20260729.md`: QA06 (wr1 waterfill —
  cell-drop Knee-A/B, realized gate, ck1 pose re-solve), **QA07** (r7 contingent lossy rung —
  nested {L16,L8,L4,base}), **QA08** (context-mixing coder, CEILING-PRICED-CLOSED, "reopens only
  with the ≥48×64 granularity re-race"), **QA11** (ν=0.7, 27% exact-zero grads, continuous
  log-bit dominates ladder), QA12 (token-LOTTO), **QA24** (granularity re-race / ECVQ, BLOCKED,
  4h re-burn), QA39 (carried-ξ INTER, NEGATIVE — "at 24×32 a single ego step is sub-token-cell").
- r7 coder-race memo + receipt (`ddm_r7_token_coder_race_20260729.md`: SMEVR is the decisive
  lossless winner 557,238 B; lossless SATURATED; contingent lossy rung `REALIZED-GATE-OWED`),
  sb1/QA11 sensitivity receipt (`ddm_sb1_20260729/qa11/receipt.json`), wr1 harness + descent
  receipt (`ddm_wr1_reverse_waterfill.py`, `wr1_descent_receipt.json`), pfs1 D1 build/eval
  receipts (`ddm_pfs1_d1_{build,eval}_receipt`), the r7 SMEVR coder `ddm_r7_token_coder.py`,
  the sb1 audit harness `ddm_lv1_s2_nullspace_audit.py`.

## The vehicle (measured)

pfs1 D1 archive `624ffe57…`, 569,996 B; eval S=2.26 (SegNet 0.00389011, PoseNet 0.22144216,
rate 0.01518148). Members: `state/tokens.dr7t` = **557,253 B = 97.8% of the archive** (the token
field IS the rate). Token lattice = **[600 pairs × 24 rows × 32 cols × 4 channels], mod-16 (L16
int4), SMEVR-coded** (dr7t header: codec_id=3, levels=16, shape 600,24,32,4). Non-token members
(floor) = 12,743 B.

## Method (archive-faithful; measurement-first)

1. Decode `tokens.dr7t` → codes (600,24,32,4). `factor_mode_delta` → per-cell temporal mode
   `base` + mod-16 residual `delta` (signed [-8,7]). Coarsening acts on the residual per token:
   step 1 = keep L16, 2 = L8, 4 = L4, 0 = drop-to-base (residual→0). Re-encode through the REAL
   SMEVR coder → **measured** bytes (archive-faithful; the same coder pfs1 ships).
2. Realized d_seg: load the T3 endpoint ckpt (`stage_seg_trunk_tau_final.npz`), ONE seg-loss
   backward → per-TOKEN |g| (sb1/QA11 sensitivity map, 35s, frac_zero=0.2704 confirming QA11).
   Inject coarsened codes into the model, `render_frame` → torch-R camera uint8 → frozen CPU
   SegNet argmax vs GT `lstars` → mean d_seg. Chunked ≤120, n48 ranked subset first, n600 for
   the winner only (ONE-n600 discipline; qp1 sibling coordinated).
3. **Correctness gates (both PASS):** (a) ckpt quantized codes == archive dr7t codes for
   **99.997%** (1,843,144/1,843,200; 56 differ by ≤2 = pfs1 build quantization noise, cancels in
   RD comparisons); (b) baseline injection → realized d_seg n48 0.0038626 / n600 0.0038892 vs
   pfs1 D1 evaluate.py 0.00389011 (Δ ≤ 2.8e-5) — the fast MLX render path is validated against
   the real evaluator. Harness `experiments/ddm_gr1_granularity_rerace.py`; receipts on SSD
   `/Volumes/VertigoDataTier/pact/ddm_gr1_20260730/`.

Reference seg+rate (the JOINT axis this rung controls) = 100·d_seg + 25·|archive|/N =
100·0.00389011 + 0.379537 = **0.76855** (n600); n48 anchor 0.7658. Water break-even = **1.273
bytes saved / flip introduced** (25·ΔB/N = 100·Δd_seg).

## MEASURED RD curves (n48 realized; bytes real SMEVR)

### A. TOKEN-granular sensitivity-ordered (the QA11 predicted winner) — DOMINATED

| candidate | archive B | rate | d_seg n48 | **seg+rate** | B/flip | vs ref |
|---|---:|---:|---:|---:|---:|---:|
| REF (L16) | 569,996 | 0.3795 | 0.003863 | **0.7658** | — | 0 |
| tok_drop27 | 560,784 | 0.3734 | 0.00478 | 0.8514 | 0.085 | +0.086 |
| tok_drop35 | 536,416 | 0.3572 | 0.005114 | 0.8686 | 0.227 | +0.103 |
| tok_drop50 | 462,270 | 0.3078 | 0.006652 | 0.9730 | 0.327 | +0.207 |
| tok_drop65 | 358,496 | 0.2387 | 0.012015 | 1.4402 | 0.220 | +0.674 |
| tok_rung_a | 470,758 | 0.3135 | 0.005724 | 0.8858 | 0.452 | +0.120 |
| tok_rung_b | 413,473 | 0.2753 | 0.006450 | 0.9203 | 0.513 | +0.155 |

**EVERY token-granular candidate is WORSE than the current point** (best = tok_drop27 at +0.086).
B/flip everywhere 0.04–0.51, all **below water 1.273**. Dropping even the bottom-10% (all
inside the 27% exact-zero-|g| set) jumps d_seg 0.00386→0.00478 for ~4 KB — and drop10/20/27 give
the SAME d_seg, i.e. the exact-zero-GRADIENT tokens are NOT flip-free under a finite drop (S2's
caveat confirmed at the realized render). Nested-rung (QA07) is likewise dominated.

### B. CELL-granular (spatial column, all 600 pairs × 4 ch share one rung) — the frontier

| candidate | archive B | rate | d_seg n48 | **seg+rate** | B/flip | vs ref |
|---|---:|---:|---:|---:|---:|---:|
| cell_drop35 | 439,836 | 0.2929 | 0.003882 | 0.6811 | 57.2 | −0.085 |
| **cell_drop50** | **359,221** | 0.2392 | **0.003947** | **0.6339** | 21.2 | **−0.132** |
| cell_drop63 | 277,815 | 0.1850 | 0.005013 | 0.6863 | 2.15 | −0.080 |
| cell_drop75 | 197,791 | 0.1317 | 0.009493 | 1.0810 | 0.56 | +0.315 |
| cell_rung_a | 354,946 | 0.2363 | 0.004681 | 0.7044 | 2.23 | −0.061 |
| cell_rung_b | 272,288 | 0.1813 | 0.006889 | 0.8703 | 0.83 | +0.104 |

**cell_drop50 = 359,221 B @ realized d_seg 0.003947 (barely above baseline 0.003863) → seg+rate
0.6339, −0.132 vs the current point.** The seg+rate MINIMUM over all no-retrain coarsenings. The
knee is ~drop50; drop35 leaves rate on the table, drop63 starts paying d_seg faster than it saves
rate. **cell_rung (graded {L8,L4}) is DOMINATED by cell-DROP at matched bytes** (cell_rung_a
354,946@0.004681 vs cell_drop50 359,221@0.003947) — intermediate precision never pays.

## Verdicts (verdict-scope ladder)

1. **Token-granular sensitivity-ordered allocation is DOMINATED** on the realized JOINT (seg+rate)
   axis — every candidate worse than the current point, and STRICTLY dominated by cell-granular
   drop at every byte budget (e.g. wr1 274 KB@0.00554 vs token drop50 462 KB@0.00665: cell has
   BOTH lower bytes AND lower d_seg). Scope: **INSTANCE/FORMULATION** (this vehicle, SMEVR
   alphabet, in-band base 0.00389) — NOT a family kill.
2. **QA11's "continuous log-bit dominates the rung ladder" is OVERTURNED through the real
   coder+render** (it was a $0 gradient prediction). Two measured mechanisms: (a) first-order |g|
   is a poor proxy for the finite drop-to-base flip cost (zero-gradient tokens flip pixels when
   dropped); (b) SMEVR conditions on the per-cell temporal mode, so scattered token drops fight
   the coder and save shallow bytes — **the CELL (spatial column), not the token, is the efficient
   coding + coarsening unit.**
3. **QA07 nested-rung {L16,L8,L4,base} is DOMINATED by clean drop-to-base at BOTH granularities.**
   Scope: **INSTANCE** — this is the measured `{cell_rung_a,b,c}` / SMEVR case under the recorded
   `|g|`-sum ordering, not a family-level "no middle ground pays" law.
   For a low-sensitivity cell, drop-to-base is nearly free (max byte saving, ~0 flips), so keeping
   it at L8/L4 only wastes bytes; for a high-sensitivity cell you keep L16. No middle ground paid
   in this instance; a lower-convex-hull allocation over the real `{L16,L8,L4,base}` RD curves is
   owed before FORMULATION/FAMILY wording.
4. **Cell-granular drop-to-base IS the RD frontier** (confirms wr1's unit choice as correct). The
   |g|-sum-ordered knee cell_drop50 (359 KB @ 0.003947) dominates the current point by −0.132 on
   seg+rate — a genuine JOINT win on the two terms this rung controls.

## The pose caveat travels (co9 R1.1 / wr1 / ck1)

Cell-drop freezes the dropped cells' content; the co9 bidirectional pose law prices far-field
(sky/hood) drops on the POSE axis (wr1 Knee-A realized +0.185 S pose from sky/hood freezes). My
|g|-sum ordering is SEG-only (the backward was seg-loss) — pose-BLIND. So cell_drop50's seg+rate
−0.132 is REAL on the two terms measured, but its full JOINT value requires the pose re-solve on
the dropped base (ck1 proved recoverable, parity 0.98×). This is exactly the wr1→ck1→v4b path;
my re-race supplies a candidate BASE, and pose is the P3v2/ck1 arm's re-solve on it.

## n600 confirm of the winner (cell_drop50) — byte-closed

The ONE n600 slot (chunked 120, 162.9s, qp1 sibling coordinated) confirms cell_drop50:

| | archive B | realized d_seg | seg (100·d) | rate (25·B/N) | **seg+rate** |
|---|---:|---:|---:|---:|---:|
| REF (pfs1 D1) | 569,996 | 0.00389011 | 0.38901 | 0.37954 | **0.76855** |
| **cell_drop50** | **359,221** | **0.004310** | 0.43104 | 0.23919 | **0.67023** |
| Δ | −210,775 | +0.000420 | +0.04203 | **−0.14035** | **−0.09832** |

**CONFIRMED: cell_drop50 dominates the current operating point by −0.098 on the seg+rate JOINT
axis** — the rate term falls 0.140 for a d_seg cost of only +0.042. Byte-closed, roundtrip-verified
archive `a6398e441f4bc818…` (359,221 B; tokens `305a2be96a292967…`) on SSD
`ddm_gr1_20260730/gr1_cell_drop50_archive.zip`. The n600 d_seg (0.004310) is +9% over the n48
estimate (0.003947) — the n48 subset mildly under-priced the drop; the −0.098 dominance holds
solidly. For reference wr1's realized Knee-A (274,333 B @ 0.00554, seg+rate 0.7364, −0.032): the
|g|-sum-ordered, less-aggressive cell_drop50 knee is a **better seg+rate operating point than wr1's
flip-mass Knee-A** (−0.098 vs −0.032), at the cost of +85 KB. Both are the SAME cell-drop family;
this refines the knee, it does not open a new mechanism.

## The QA08/QA24 finer-grid (≥48×64) question — answered as far as no-retrain can

My no-retrain harness CANNOT synthesize a finer ≥48×64 grid (that needs the QA24 re-burn — token
VALUES don't exist at 48×64 without retraining). But the measured result **informs** it: the token
(finer-than-cell) direction is DOMINATED at the SMEVR alphabet, and cell-drop already makes ~50%
of the 768 cells droppable near-free. A finer 48×64 grid = 4× more cells, each ego-step still
sub-cell (QA39) — so finer is a coder GAMBLE (must prove the 4× cells code cheaply enough), not an
obvious win. Conversely the COARSER direction is validated: post-hoc cell-drop of the low-|g| half
is near-free, so a from-birth COARSER burn (QA24, solve-init at ~384 effective cells) is the
promising re-burn — it can only exceed the −0.132 post-hoc drop. **QA08's reopen condition (the
≥48×64 re-race) is NOT met by a no-retrain win; the live path is coarser-not-finer.**

## Triality

- **DAG:** `ddm_gr1_granularity_rerace_DAG_FEED_20260730.md` (FEED block).
- **DSL/state:** research-only lane `ddm_gr1_granularity_rerace`; no eval/dispatch authority.
- **equations:** the seg+rate exchange (water 1.273 B/flip), the SMEVR-cell-unit lemma (token
  dominated by cell), the rung-dominated-by-drop lemma. Feeds the costate SENSE laws
  (QA11 continuous-log-bit law re-scoped to INSTANCE-overturned).

## Honest boundaries

- Every number `[macOS-CPU advisory]`; d_seg realized through the real render+SegNet, bytes real
  SMEVR, but NOT a contest score. Pointer 0.1910828242 [contest-CPU] UNMOVED. This unit is MEANS.
- n48 is a subset ranking anchor; the winner cell_drop50 is confirmed at n600 (above). The token
  DOMINATED verdict rests on n48 with large margins + strict domination (robust to n48→n600 drift
  ≈0.7%).
- The QA07 nested-rung sentence is VS1-regraded to INSTANCE scope: only the measured
  `{cell_rung_a,b,c}` / SMEVR allocation under the recorded `|g|` ordering is closed.
- cell_drop50's ordering is seg-only / pose-blind; the composed decision cell (base + pose
  re-solve, and optionally a seg-solve arm) is v4b/v4c's, not this rung's.
