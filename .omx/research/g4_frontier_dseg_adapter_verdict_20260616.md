# G4 frontier-adapter d_seg verdict — NO-GO (theory-grounded, advisory-measured)

**Lane:** `lane_pr110_frontier_dseg_adapter_20260616`
**Date:** 2026-06-16
**Authority:** `[contest-CPU advisory] NON-PROMOTABLE` — exact frozen upstream SegNet,
GT decoded via `upstream/frame_utils.yuv420_to_rgb` only, torch-CPU, NEVER MPS.
No paid/remote dispatch. `score_claim=false`, `promotable=false`.

## TL;DR

A score-aware ADDITIVE d_seg-flip adapter on the verified 0.19110 contest-CPU
frontier **does NOT pay — by a wide, theory-bounded margin.** The frontier's d_seg
is already so low (`0.00055978`, ~110 flip pixels/frame) that the *positional
entropy of the residual flips exceeds the entire seg term they represent.* The
break-even is **1.27 bytes per fixed flip**; the cheapest measured appearance
correction spends ~60–100 B/flip (~80× over budget), and even the Shannon
position-only floor costs 1.20× the maximum recoverable seg. The frontier's cheap
seg levers (per-pair frame-bias selector + DQS1 single-weight decoder
substitution) are **already exhausted in the packet**; the residual flips are not
independently fixable by local RGB substitution (≈50% spillover into new
off-target flips). **PIVOT off the frontier-adapter path.**

## SEARCH-FIRST provenance (cited)

- Frontier archive: `experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/archive.zip`
  (177,169 B, sha `b46897267ded…`, member `x` ZIP_STORED 177,069 B), pointer
  `.omx/state/canonical_frontier_pointer.json:5,21` → contest-CPU **0.19109982**.
- Inflate runtime: `…/submission_dir/inflate.py` (FP11 wrapper → CTXR container →
  `HNeRVDecoder` → bicubic 874×1164 → channel-bias → round → per-pair `FECa`
  selector; `inflate.py:630-713`). SegNet reads **only frame1** (`upstream/modules.py:107`
  `x[:, -1, ...]` → bilinear 512×384 → argmax over 5 classes).
- d_seg functional: `upstream/modules.py:111-113` `diff=(out1.argmax!=out2.argmax).float().mean()`.
- Verified components (`…/recoded_cpu_eval/report.txt`): d_seg **0.00055978**,
  d_pose **0.00002942**, rate **0.00471878** → 0.19110.
- Existing flip machinery reused: `tac.residual_basis.residual_flip_delta`
  (RFD1 sparse-RGB flip-delta codec, `experiments/run_residual_flip_delta_advisory_s.py`
  is its Cool-Chic-base runner), `tac.score_aware_loop.targets.load_frozen_distortion_net`,
  `tac.boundary_math.seg_core.decode_gt_frame1_pairs`, GT cache
  `experiments/results/capstone_gt_targets_cache/gt_targets_n600.pt` (`seg` = L*,
  (600,384,512) int64).
- Frontier already carries the cheap seg levers (verified by parsing member `x`):
  `selector_kind=compact` (600 per-pair `FECa` frame-bias modes) **and** a DQS1
  q-domain decoder substitution (`storage_index=26, delta=+1, pair_all_frames`,
  31 selected pairs). The near-free seg/pose levers are spent.

## What was built (NO-FAKE)

`experiments/probe_frontier_adapter_dseg_flip.py`:
1. Decodes the frontier's ACTUAL frame1 for N pairs, pixel-for-pixel as scored
   (imports the frontier `inflate.parse_member`/decoder/selector verbatim).
2. Runs the EXACT frozen SegNet on the decoded frame1 → argmax; compares to L*.
   **Validation:** measured frontier d_seg on N=8 = **0.00054042** (≈ the verified
   0.00055978 — first-8-pair vs 600-pair average; the decode+scorer chain is
   faithful). ~106 flips/frame.
3. Builds an additive RGB correction at flip pixels toward GT (L-inf capped,
   optional tube dilation), **applies it, and RE-SEGMENTS through the exact
   SegNet** — the flips are actually applied + actually re-scored, not estimated.
4. Range-codes the correction (RFD1) and a compact-brotli lower bound; reports
   real adapter bytes + advisory ΔS, projected to the full 600 pairs at the
   contest rate denom.

## Measured rows (advisory, N=8)

| config (gated) | d_seg before → after | fixed / off-target | accepted | adapter B@8 (proj@600) | S(proj600) Δ |
|---|---|---|---|---|---|
| L-inf 255 (full GT) | 0.000540 → 0.000626 | 291 / **428** | 0/8 (ungated worsens) | 2.78 MB | +1.86 |
| L-inf 255, per-pair gate | 0.000540 → 0.000540 | 0 / 0 | 0/8 | — | +0.0006 (header only) |
| **L-inf 8, per-pair gate** | 0.000540 → 0.000529 | 35 / 17 | 4/8 | 3,564 (**267 KB**) | **+0.177** |
| L-inf 16, per-pair gate | 0.000540 → 0.000512 | 114 / 70 | 7/8 | 7,168 (**538 KB**) | +0.355 |

Every operating point projects to a WORSE score. The best per-pixel d_seg gain
(~5% relative at L-inf 16) is buried by an adapter that is **1.5–3× the entire
frontier archive**. The per-pair acceptance gate (only keep pairs where the
correction net-reduces that pair's flips) confirms the spillover is intrinsic:
at full GT substitution, **0/8 pairs** net-improve.

## The theory-bounded NO-GO (the decisive number)

Each fixed flip reduces S by `100/(600·196608) = 8.48e-7`. Each adapter byte
costs `25/37,545,489 = 6.66e-7` in rate. Therefore an adapter may spend **at most
1.27 bytes per fixed flip** to break even. Sparse position+value coding of
camera-res corrections cannot reach that:

- **Shannon position-only floor** (just naming WHICH ~110 px/frame flip, no
  values): `≈ log2(C(196608,110))` ≈ 1155 bits/frame → **98.7 KB / 600 frames**
  → rate cost **0.0673**, which is **1.20× the entire frontier seg term
  (0.05598)**. So even an information-theoretically optimal adapter that fixed
  *100% of flips with zero off-target* would still raise S.
- Off-target reality makes it worse: ~50% of touched flips induce a new flip, so
  net seg gain per coded flip is roughly halved while position cost is unchanged.

This is a **property of the frontier's operating point**, not a codec/design
weakness: the residual d_seg is below the rate at which per-flip correction can
ever amortize. (The RFD1 histogram-header pathology — 2.75 MB from a max-gap-sized
gap-frequency table over the 3M camera-res index space — was diagnosed and bypassed
with a compact-brotli estimate; even the compact estimate is ~80× over the
break-even, so the codec is not the blocker.)

## Byte-close readiness

Not scaffolded — correctly so. The advisory measurement proves the additive
section would RAISE S at every operating point, so appending it to the archive +
wiring an inflate-side consumer would produce a strictly worse packet. Building
the byte-close would be premature per the means/ends firewall.

## Pivot recommendation

The frontier's seg term is rate-floored for *per-pixel/per-flip* correction. The
remaining seg headroom (0.05598) can only be attacked by mechanisms whose byte
cost is **sub-linear in flip count** — i.e. the levers the frontier ALREADY uses
(per-pair frame-bias selector, single-weight DQS1 decoder substitution that shifts
a whole tensor) pushed further, or a decoder-capacity reallocation toward the
192×256/384×512 band where flips live (`experiments/probe_dseg_sensitivity_map.py`
lineage) — NOT an additive pixel sidecar. The fastest exact-eval row is therefore
NOT this adapter; redirect G4 budget to a cheaper-per-flip seg mechanism or to the
pose/rate axes.

## Wire-in hooks (Catalog #125)

1. sensitivity-map: ACTIVE — measured per-flip seg-benefit (8.48e-7/flip) +
   break-even (1.27 B/flip) is a reusable bit-allocator prior for any seg sidecar.
2. Pareto: ACTIVE — the position-entropy floor (0.0673 ≥ 0.05598) is a hard
   constraint: no per-flip seg sidecar can be on the frontier Pareto set.
3. bit-allocator: ACTIVE — "do not allocate bytes to per-pixel seg correction
   below d_seg ≈ position-floor crossover" rule.
4. cathedral autopilot: N/A (advisory, non-promotable; no archive deployed).
5. continual-learning posterior: the break-even + floor are the durable signal.
6. probe-disambiguator: this probe IS the disambiguator (additive-adapter pays? NO).

mission_contribution: `frontier_protecting` (a measured NO-GO that prevents
spending an exact-eval row on a dominated path).
