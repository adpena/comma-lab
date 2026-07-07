# T5 CRUCIBLE — SEAT S4 POSITION — RATE (Shannon charter) — 2026-07-07

Seat: S4 (RATE — `25·bytes/37,545,489`). Anti-anchoring honored: no other `position_S*.md` read.
Axis: everything here is `[macOS-CPU advisory] NON-PROMOTABLE` build/measurement; the pointer
(0.19110 contest-CPU) moves ONLY through a byte-closed `upstream/evaluate.py` n600 exact row.
All new numbers below were MEASURED inline this session on the REAL mod32cap ep650 EMA-BEST
checkpoint (`experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz`,
447.9 KB fp32 on disk) with real brotli/lzma/zip byte counts — never estimated. Probe script:
scratchpad `s4_rate_probe.py` (foreground, <1 GiB, <2 min; envelope honored).

---

## Position

### P1. The byte accounting — what the next run's archive ACTUALLY counts (MEASURED)

**The counted-payload law (verified from source):** `upstream/evaluate.py:63` counts ONLY
`archive.zip` `.stat().st_size`. The byte-close grammar (`tools/levelset_byte_close_and_eval.py::build_levelset_blob`)
packs `LVLS1 | manifest_json | base_int8_brotli | code_int8_brotli | pose_sidecar
[| lane_band] [| pose_carrier]` into a single zip member. FREE under rule 118 (regenerated in
inflate.py): curvelet/shearlet bank `B`, self-orient directional feats (fixed point of the
decoder's own argmax), lane rasterizer/compositor, derive-H + exp_se3 + warp, R itself.
COUNTED: all learned weights, the per-(pair,frame) `code` table, lane manifold coeffs, ξ payload,
manifest, zip container.

**Per-tensor breakdown, mod32cap ep650 EMA-BEST (int8 symmetric + brotli-11, the shipped grammar):**

| tensor | shape | numel | int8-brotli (solo) | order-0 floor | share of blob |
|---|---|---:|---:|---:|---:|
| `code` | (1200, 32) | 38,400 | 20,355 | 31,799 | 24.8% |
| `film.weight` | (768, 32) | 24,576 | 20,285 | 20,271 | 24.7% |
| `hidden.{0..3}.weight` | 4×(96,96) | 36,864 | 32,461 | 32,042 | 39.5% |
| `in_proj.weight` | (96, 96) | 9,216 | 7,224 | 7,293 | 8.8% |
| heads (`out_sdf`,`out_tex`) + biases | — | 2,016 | ~2,000 | ~1,700 | 2.4% |
| **counted stream totals** | | **111,103** | **base 61,838 + code 20,355 = 82,193** | base H0 61,303 / code H0 31,799 | |

**Full-archive reconstruction (MEASURED, this session):** blob with the real ~1.9 KB JSON
manifest = 84,114 B → zip DEFLATED (`0.bin`, the tool's current choice) = **83,406 B →
rate 0.05553**. (The DAG's 0.05499/82.6 KB was an earlier checkpoint state; same class.)

**Two structural facts the whole rate face hangs on (both MEASURED):**
1. **The base (weights) stream is AT its order-0 entropy floor.** brotli 61,838 vs H0 61,303
   (100.9%); lzma-9e = 62,488 (+1%). There is NO coder slack on weights. The ONLY base-rate
   levers are (a) fewer symbols (bit-depth ↓ = Arm E) or (b) lower symbol entropy
   (in-training shaping = WeightEntropyPenaltyMLX). Coder migration is dead.
2. **The `code` stream is 36% BELOW its order-0 floor** (brotli 20,355 vs H0 31,799): brotli's
   context modeling already exploits inter-row correlation. Corollary measured below: PR95-style
   temporal-delta makes it WORSE; a per-dim (column-major) layout makes it better.

### P2. bytes → rate-term table for the candidate configs

Constants: rate = 25·bytes/37,545,489; **λ_bytes = 25/37,545,489 = 6.659e-7 S/byte (EXACT-ANALYTIC
from the score law — this is the arbitration currency, not an empirical fit).** 1 KB = 6.8e-4 S.

| config | base+code blob | lane band | pose ξ | manifest+zip | archive.zip | rate term |
|---|---:|---:|---:|---:|---:|---:|
| **mod32/h96 (control, MEASURED)** | 82,193 | — | — | ~1,213 | **83,406** | **0.05553** |
| mod32 + grammar-rev v2 (P5, MEASURED parts) | 80,860 | — | — | ~760 | ~81,620 | 0.05434 |
| mod32 + band(LBND4) + ξ | 82,193 | 30,892 | 2,700 | ~1,300 | ~117,085 | 0.07796 |
| mod32 + band(LBND2-smoothed) + ξ | 82,193 | 24,149 | 2,700 | ~1,300 | ~110,342 | 0.07347 |
| mod48 + band(LBND4) + ξ (blob INFERRED) | ~102,400 | 30,892 | 2,700 | ~1,300 | ~137,292 | 0.09141 |
| mod48 + band + ξ + Arm-E compress (int6-class, Δd_seg-gated) | ~67,600 | 30,892 | 2,700 | ~1,300 | ~102,492 | 0.06824 |

Deltas that generated the table (sources in §Derivations):
- **mod32 → mod48 = +20.2 KB = +0.0135 rate** (INFERRED from measured shapes/ratios: only
  `code`(1200×mod) and `film.weight`(768×mod) scale with mod_dim; hidden/heads do not).
  Pays iff Δd_seg < −1.35e-4. Consistent with FEED-07a: capacity is SECONDARY — and now costed.
- **Lane band = +24.1–30.9 KB = +0.0161–0.0206 rate, NOT "~0 bytes".** LBND2 41,526 → LBND4
  30,892 (MEASURED n600, FEED-08h) → LBND2-on-smoothed-source 24,149 (MEASURED, FEED-wf).
  LBND4-on-smoothed is unmeasured (predicted ~18–22 KB, RECESS R3). **FEED-07a's "near-zero
  byte" phrasing is FALSE at n600 and must not be load-bearing in any seat's arithmetic** —
  the band's generator is free; its per-pair fitted coeffs are 24–31 KB counted.
- **Pose ξ (store-nothing, #257) = +2.7 KB** (brotli-raw n600 winner; delta_ar 3.2 KB;
  derive-H removed the 43.2 KB redundant per-pair H — VERIFIED-VIA-SOURCE `xi_pose_coder.py`).
- **Uniform re-quantization on the measured checkpoint (Δd_seg NOT measured — upper-bound
  curve for Arm E):** int6 → 54,215 B (rate 0.0361, −34%); int5 → 41,326 (0.0275, −50%);
  int4 → 28,958 (0.0193, −65%).

**Budget check vs the FEED-07f win condition** (PR95 distortion at our rate → sub-0.15 needs
archive ≤ ~105 KB): mod32+band+ξ (110–117 KB) slightly EXCEEDS it; mod48+band+ξ (137 KB) far
exceeds it. **Consequence: if the stack includes band + pose + any capacity, the Arm-E
compress-half (or the smoothed-band + weight-entropy combination) is NOT optional — it is the
component that pays the budget back under 105 KB.** All configs stay far under the 167 KB
merely-beat-0.19110 line.

### P3. Arm E — train-big-compress-small: the concrete compress-half

**Design (uses ONLY built tools; nothing invented):**
1. **Sensitivity:** `tools/apply_sensitivity_bitalloc_witness.py` (#336 adapter, COMPLETE,
   never fired at our checkpoint) — per-tensor direct measurement: qdq tensor t to int-probe_bits,
   others at int8, measure Δd_seg through the REAL R + frozen CPU-torch SegNet on GT pairs →
   calibrates the separable KKT model `D = Σ c_t·2^{-b_t}`.
2. **Allocation:** the UNCHANGED #157 `tac.frontier_exact_bitalloc.waterfill_bit_allocation` +
   `lam_for_target_mean_bits` (reverse water-fill; λ-bisection; closed-form b_t*). By convexity
   the waterfill dominates my measured uniform int-N curve at equal distortion — the uniform
   curve is the floor of what to expect, not the target.
3. **Entropy coding: KEEP brotli-11. Verdict from measurement, not reflex** — base stream is at
   its order-0 floor (P1.1); lzma +1%; range/arith coding pays only where a cheap transmitted
   PMF exists, which is already OURS where it wins (ξ delta_ar coder, LBND4 varint/rice residual
   stage). No coder migration for weights.
4. **Realization:** qdq each tensor at its allocated b_t ON TOP of the shipped int8 grammar
   (fewer distinct int8 symbols → smaller brotli; reader unchanged) — exactly what the #336
   adapter implements. Zero inflate change.
5. **Predicted bytes at Δd_seg≈0 (pre-registered, first-principles):** the KKT solve at
   mean-bits 6 must land ≤ the measured uniform-int6 54,215 B at strictly better Δd_seg
   (convexity). Predicted counted base+code ∈ **[52, 68] KB at Δd_seg ≤ +5e-5** (ΔS_rate
   −0.010 to −0.020 vs int8, at ΔS_seg ≤ +0.005). Sensitivity expectation: `code`/`out_sdf`
   pinned near 8 bits (argmax-critical), `film`/`hidden` drop to 4–6 bits.
6. **What $0 probe measures this TODAY:** RECESS R1 below (the exact CLI is in the tool's
   docstring, pointed at the mod32cap dir). It exceeds my 10-min inline envelope (per-tensor
   scorer forwards), so it is a RECESS item, not run here.

**In-training entropy shaping vs post-hoc compression — BOTH, they compose, different mechanisms:**
- `WeightEntropyPenaltyMLX` (now PORTED to the levelset trainer as a DSL Lever — VERIFIED-VIA-SOURCE
  `curriculum_dsl.py:2416` + trainer `--weight-entropy-penalty-lambda`) LOWERS THE SYMBOL ENTROPY
  ITSELF — the only mechanism that can move the base stream, since brotli already sits on the
  order-0 floor (P1.1). The torch −19.6% anchor is a MECHANISM-EXISTS proof only (borrowed-number
  firewall is already welded into the lever's docstring); λ* open in {5,15,30}, default 15.
- Post-hoc #336 REDUCES BITS at measured Δd_seg — orthogonal axis (fewer symbols vs cheaper symbols).
- The A/B that decides: λ∈{0,15} rider on the next run's arms; headline metric =
  `measured_symbol_entropy_bits_numpy` + real `quantize_levelset_blob` bytes at matched d_seg.
- `CodeSpectralEntropy` (DM1b) is a CAPACITY/rank lever on cov(code), byte-free at the flag level;
  its rate relevance is indirect (a lower-rank code table should brotli smaller — the code stream
  is the one place with structure left). No rate prediction offered; ride it as designed by the
  vehicle seat, count the bytes after.

### P4. Never-fired RATE-lever ledger resolution (my face)

| lever | verdict | grounds |
|---|---|---|
| `AnalyticLaneRenderBand` | BUILD-and-compose, with its REAL cost booked (+24.1–30.9 KB, +0.0161–0.0206 rate) — net-ΔS gate is the composed A/B (FEED-07d ceiling [0.02, 0.26] vs rate cost 0.021: net-positive at the mid/upper band, NOT guaranteed at the conservative edge) | LBND2/LBND4/smoothing all MEASURED n600 |
| `WeightEntropyPenaltyMLX` | BUILD-and-compose as λ∈{0,15} A/B rider (never-fired on witness; own n600 byte-closed A/B is the admission) | P3; base at H0 floor makes this the ONLY base-entropy mover |
| `CodeSpectralEntropy` | compose per vehicle-seat design; rate side = measure-after (code stream is where structure remains) | P1.2 |
| Arm-E #336 post-pass | BUILD (RECESS R1 first, $0) | P3.5-6 |
| Grammar rev v2 (P5) | BUILD (byte-close tool change only, no trainer surface) | MEASURED −2.8 KB |
| Hyperprior-class entropy models | DEFER-with-reason: twice-ruled-out at our payload scale (measured no-2D-locality + side-info inversion, FEED-08e/§20) | do not re-open |
| LBND3 ego-predictive lane coding | DEFER-with-reason: MEASURED NEGATIVE (1.04–1.34× LARGER than LBND2, FEED-wf) | correspondence-before-prediction law |

### P5. Free measured rate wins — grammar rev v2 (byte-close tool only, −2.8 KB ≈ −0.0019 rate)

All three MEASURED this session on the real blob; all decode-trivial; zero trainer surface:
1. **Column-major `code` stream** before brotli: 20,355 → 19,022 B (**−1,333 B, −6.5%**) —
   per-dim streams expose within-dim continuity brotli can model (inflate adds one transpose).
2. **Brotli the manifest** inside the blob (1,895 → ~650 B) **and then STORE (not deflate) the
   zip member with a 1-char name**: current grammar (plain-JSON manifest + DEFLATED `0.bin`)
   = 83,406; brotli-manifest + STORED `x` = **82,970 B (−436 B)**. Nuance the #79 audit missed
   for OUR grammar: with the plain-text manifest, DEFLATED currently BEATS STORED by 816 B
   (deflate recovers manifest text) — the right fix is brotli-manifest first, THEN the #79
   STORED/1-char floor applies (container overhead → its 100 B theoretical minimum).
3. Combined estimate with both: ~81.6 KB → rate 0.05434 (**−0.0012 S** for free).

### P6. RATE-side composition rules — the marginal-ΔS-per-byte ranking DECIDE consumes

Byte-FREE (compose freely, rate-side no objection): all losses/schedules/curricula/optimizer
choices (incl. Muon, focal, logit-adjust, StepNativeActivation, StiefelW, eikonal, length,
seeds-by-RNG, EventTriggeredCurriculum), the basis REBALANCE at fixed width (freq_along/across
are manifest scalars), self-orient (fixed-point, GT-free). Second-order caveat (MEASURED,
FEED-gn class): schedule shape moves blob bytes ±0.5 KB (tau tail INFLATED payload
73,077→73,553; l7 deflated −461 B) — noise at λ_bytes scale (±3e-4 S), never load-bearing.

Byte-ADDING (each must clear ΔS_distortion > 6.659e-7·B):

| lever | bytes B | required Δ(100·d_seg + pose) to pay | measured/predicted ΔS available | marginal S/byte |
|---|---:|---:|---|---:|
| Pose ξ + FiLM (pose-ON) | +2,700 | > 0.0018 | up to ~31 (pose term collapse) — GATED on #248 realized-d_pose | ~1e-2 (if it works: the single best buy in the stack) |
| Basis WIDTH growth (per +32 input feats) | +~2,400 | > 0.0016 | −48% anchor class | ~0.03–0.07 |
| Lane band | +24,149–30,892 | > 0.0161–0.0206 | ΔS_seg ∈ [0.02, 0.26] (FEED-07d) | 0.7e-6–8e-6; composed A/B decides |
| mod48 capacity | +20,200 | > 0.0135 (Δd_seg < −1.35e-4) | unknown (2-point) | SECONDARY by design |
| Arm-E compress | **−14k to −30k** | — (it PAYS; gate is Δd_seg≈0) | ΔS −0.010 to −0.020 | pure win if gate passes |
| Grammar rev v2 | **−2,800** | — | ΔS −0.0019 | free |

Costing rule for S1 (basis seat): **counted cost of basis enrichment ≈ 75 B per added input
feature** (Δin_feat × hidden(96) × measured brotli ratio ~0.78) — the −48%-class lever's rate
price is ~0.0016 per +32 feats, i.e. effectively free; rebalance-in-place is exactly free.

---

## Derivations + assumption tags (#363)

- Score law + counted surface: VERIFIED-VIA-SOURCE (`upstream/evaluate.py:63` quoted in
  `.omx/research/archive_packaging_byte_audit_20260610T224611Z.md` §1; blob grammar
  `tools/levelset_byte_close_and_eval.py:339-458`).
- λ_bytes = 25/37,545,489 = 6.659e-7 S/B: DERIVED-EXACT (∂S/∂bytes of the score law; matches
  `costate_controller_design_20260705.md:63` ANALYTIC row).
- mod32cap counted bytes 82,193 (base 61,838 + code 20,355), archive 83,406 DEFLATED /
  82,970 grammar-v2, manifest 1,895: VERIFIED-VIA-ANCHOR (this session's inline probe on
  `levelset_witness_ema_BEST.npz`; script in scratchpad; mirrors `quantize_levelset_blob`
  exactly — `lever_b_levelset_generator.py:956-983`).
- Order-0 floors (base 61,303 / code 31,799), lzma +1%, uniform int6/5/4 curves, col-major
  code −1,333 B, temporal-delta code +13,056 B (NEGATIVE), zip STORED/DEFLATED variants:
  VERIFIED-VIA-ANCHOR (same probe, real byte counts).
- Lane band LBND2 41,526 / LBND4 30,892 / smoothed 24,149 / order-0 floor 26,179 / LBND3
  negative: VERIFIED-VIA-ANCHOR (DAG FEED-08h + FEED-wf rows;
  `experiments/results/lane_band_res_coder_20260707/lane_band_res_coder_n600_measured.json`).
- ξ payload 2.7 KB (brotli-raw winner) / 3.2 KB (delta_ar) / derive-H −43.2 KB:
  VERIFIED-VIA-SOURCE (`src/tac/boundary_math/xi_pose_coder.py:1-56,242-251`) — the 2.7 KB is
  the module's own recorded n600 measurement, not re-measured here.
- mod48 blob +20.2 KB: INFERRED (measured mod32 shapes + per-stream brotli ratios 0.825 film /
  0.530 code held constant; ratios may shift with mod_dim — the 2-point run measures the truth).
- LBND4-on-smoothed ~18–22 KB: INFERRED (sub-additive composition of two measured wins; RECESS R3).
- Arm-E predicted band [52, 68] KB at Δd_seg ≤ +5e-5: DERIVED (KKT waterfill dominates the
  measured uniform curve at equal D — convexity of 2^{-b}; Dykstra-feasibility grounding =
  the separable-model intersection the #157 solve implements) — the Δd_seg side is
  ASSUMED-AWAITING-VERIFICATION until RECESS R1 (why unavoidable: per-tensor scorer forwards
  exceed the seat's 10-min inline envelope).
- FEED-07f headroom (≤105 KB for sub-0.15 at PR95 distortion; ≤167 KB to beat pointer):
  VERIFIED-VIA-ANCHOR (DAG FEED-07f) — with its own pose-term caveat (p component of PR95
  distortion split is DERIVED, and the witness's realized pose is OPEN/UNMEASURED).
- "~0.026 spare rate budget": OPERATOR-STATED, derivation OWED (dossier §5A) — NOT used as
  load-bearing anywhere in this position; all budget rows recomputed from the score law.

## PR95 cargo-cult audit (my face)

| element | verdict | grounds |
|---|---|---|
| brotli quality=11 (L32) | JUSTIFIED-KEPT | MEASURED at the order-0 floor on base; lzma +1%; not reflex — re-measured on OUR payload |
| temporal-delta latent coding (L25) | **DROP** | MEASURED NEGATIVE on our `code`: 20,355 → 33,411 B (+64%); the witness code table is not a smooth temporal trajectory at int8 granularity |
| per-tensor storage perms (L22) | ADOPT-BY-MEASUREMENT (not reflex) | col-major `code` −6.5% MEASURED; base perms untested and unneeded (base at H0 floor — permutation cannot beat it) |
| monolithic single-member zip (L20) + #79 floor | JUSTIFIED-KEPT + corrected | container floor 100 B stands; but for OUR grammar STORED only wins AFTER brotli-ing the manifest (P5.2) — blind #79 application would have COST 816 B |
| range/arithmetic coding for select tensors (L30) | SPLIT | DROP for weights (no coder slack, P1.1); KEEP where already ours + measured (ξ delta_ar, LBND4 residual stage) |
| fp16 per-tensor scales (L29) | N/A-at-our-scale | 17 tensors → JSON scales are ~0.5 KB inside a brotli'd manifest; not worth a binary grammar |
| mod-dim ladder as capacity reflex | COSTED, SECONDARY | +0.0135 rate per mod32→48 step; basis-match (free) is PRIOR — the rate table makes the FEED-07a discipline quantitative |
| hyperprior entropy model (Ballé lineage) | DROP (measured) | twice-ruled-out at hundreds-of-bytes payload scale (FEED-08e) |

## RECESS measurement proposals

- **R1 — #336 sensitivity bit-alloc on the mod32cap checkpoint (the Arm-E gate).**
  What: KKT reverse-waterfill with measured per-tensor sensitivity.
  Command: `.venv/bin/python tools/apply_sensitivity_bitalloc_witness.py --ckpt-dir
  experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z --npz-name
  levelset_witness_ema_BEST.npz --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz
  --probe-pairs 16 --eval-pairs 96 --mean-bits 6 5 --out <scratch>/bitalloc_witness_ep650.json`.
  Cost: ~30–90 min CPU foreground, <8 GiB (chunked per-tensor; resumable by tensor if needed).
  Predicted band: counted base+code ∈ [52, 68] KB at Δd_seg ≤ +5e-5 (grounding: waterfill
  dominates the MEASURED uniform-int6 54,215 B by convexity). Kill: Δd_seg > +2e-4 at
  mean-bits 6 (rate gain 0.019 < seg harm 0.02 → compress-half dominated at this scale).
  Note: 16/96-pair subsets are ADVISORY (labeled); the launch verdict needs the n600 byte-close.
- **R2 — WeightEntropyPenaltyMLX λ∈{0,15} rider** on whichever training arms the crucible fires
  (operator-GO surface, not $0): headline = `measured_symbol_entropy_bits_numpy` + real
  `quantize_levelset_blob` bytes at matched d_seg. Predicted: −10 to −20% base bytes (torch
  mechanism anchor; does NOT transfer numerically). Kill: <−3% at any d_seg cost.
- **R3 — LBND4-on-smoothed-source ($0, ~1 min, 1 GiB):** rerun the FEED-08h measurement script
  (`experiments/results/lane_band_res_coder_20260707/`) with the win15-median-smoothed source.
  Predicted: 18–22 KB. Kill: ≥24,149 B (no gain over smoothed LBND2 alone).
- **R4 — grammar rev v2 fold ($0 build):** col-major code + brotli-manifest + STORED/`x` into
  `levelset_byte_close_and_eval.py` (+ inflate transpose + manifest-decompress); acceptance =
  bit-identical dequantized params + full parity gate; expected −2.8 KB (MEASURED components).
- **R5 — mod48 2-point rate verification** rides the capacity arm's own run (no separate rate
  measurement needed): my +20.2 KB projection is pre-registered; if the realized blob differs
  by >10%, the brotli-ratio assumption is falsified and the table re-derives.

## Interfaces

- **To S1 (basis/vehicle):** rebalance-in-place = 0 counted bytes; width growth = ~75 B per
  added input feature (~0.0016 rate per +32 feats). Spend basis width freely — it is two orders
  of magnitude cheaper per ΔS than capacity.
- **To the pose face:** ξ carrier = +2.7 KB counted (grammar block exists; derive-H free);
  a dead pose SIDECAR is bytes the scorer never reads — refuse it. Realized-d_pose through the
  FiLM render (#248) is the gate on the stack's single best S/byte buy.
- **To the schedule/curriculum face:** rate-side no objection to anything byte-free; tau-tail
  payload inflation (±0.5 KB) is real but sub-λ_bytes-significance.
- **To the costate DECIDE layer:** consume P6's table as the marginal-ΔS/byte prior;
  λ_bytes = 6.659e-7 S/B exact; re-rank when R1/R3 land.
- **From the crucible synthesis I need:** (a) the composed lane-band A/B verdict (its net
  positivity at the conservative ceiling edge is NOT rate-guaranteed); (b) the pose-ON decision
  (it moves the budget by ~31 S — every rate number is noise beside it); (c) which capacity
  point (mod32 vs 48) so the Arm-E target checkpoint is fixed.
- **Honest apparatus note:** the never-fired LEDGER is not wired to real-run activations
  (grounding packet caveat) — my per-lever verdicts used run configs + source, not the ledger.
