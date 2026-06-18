# FP-shrink rate-lever $0 PTQ smoke — bc20 n600 small-basis basin (task #136)

- **Date:** 2026-06-18T03:38Z
- **Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. Local CPU ≈ exact contest-CPU
  to ~0.001% per G3, so these d_seg/d_pose are contest-CPU-faithful (advisory). This is a
  FEASIBILITY smoke, NOT a byte-closed dual-exact row. **Frontier pointer UNMOVED.**
- **Spend:** $0 — no GPU dispatch, no paid spend. CPU-only (8 threads); the live MPS
  `launch_bind_all_taper_ab.py` train run was NOT touched (MPS owns the GPU; scorer CPU-only
  to avoid the 2×/23× MPS SegNet/PoseNet corruption).
- **Basin:** `experiments/results/torch_vehicle_full_mps_basin_bc20_n600/best/best_archive.bin`
  (89,136 B; base_channels=20, latent_dim=28, 600 pairs; the G3-validated trustworthy basin).
- **Probe:** `experiments/probe_fp_shrink_ptq_bc20_n600.py` (full 600-pair authority eval per mode
  via `RealScorerContext.exact_eval`; bytes MEASURED by re-encoding through the shipped vendored
  `codec.build_archive` + real brotli — NOT the predecessor's nbits/8 estimate).
- **Reusable helper landed:** `tac.post_hoc_weight_shrink` (`intn_qdq` / `e2m1_qdq` /
  `requantize_decoder_state_dict`); 21 NO-FAKE tests in `src/tac/tests/test_post_hoc_weight_shrink.py`.
- **Pipeline validation (NO FAKE):** fp32/int8 parse-back exact_eval reproduced the basin's
  `best_meta.json` BIT-FOR-BIT (d_seg 0.002601, d_pose 0.000342, S 0.3779); parse-back rebuild
  bytes 89,136 = on-disk (codec round-trip exact).

## The decisive $0 question

The bc20 decoder ships as per-tensor symmetric INT8. Rate term = 25·89136/37545489 = **0.0594**
(decoder weights dominate the 89 KB). FP-shrink = re-quantize decoder weights to fewer bits to cut
bytes. The rate saving is MECHANICAL; the OPEN question is the **distortion-hold**: does d_seg/d_pose
hold under fp8/fp6/fp5/fp4 weight quantization (no QAT)?

## Ranked bit-width → S table (full 600-pair, MEASURED bytes, [contest-CPU advisory])

| mode | d_seg | d_pose | bytes | rate | **S** | ΔS vs int8 | d_seg %vs int8 | d_pose %vs int8 |
|------|------:|-------:|------:|-----:|------:|-----------:|---------------:|----------------:|
| fp32 (≡int8) | 0.002601 | 0.000342 | 89136 | 0.0594 | **0.3779** | baseline | 0% | 0% |
| **int8** (shipped) | 0.002601 | 0.000342 | 89136 | 0.0594 | **0.3779** | 0 | 0% | 0% |
| int7 | 0.002808 | 0.000373 | 78750 | 0.0524 | **0.3944** | **+0.0165** | +8.0% | +9.3% |
| int6 | 0.003306 | 0.000508 | 68400 | 0.0455 | **0.4474** | **+0.0696** | +27.1% | +48.6% |
| int5 | 0.005074 | 0.002683 | 57475 | 0.0383 | **0.7095** | **+0.3316** | +95.1% | +685% |
| int4 | 0.012883 | 0.110085 | 46590 | 0.0310 | **2.3686** | **+1.9907** | +395% | +32,118% |
| fp4_mixed (E2M1 interior + int8 heads) | 0.006082 | 0.003394 | 64168 | 0.0427 | **0.8351** | +0.4573 | +134% | +893% |
| fp4_all (E2M1) | 0.006854 | 0.006598 | 64133 | 0.0427 | **0.9849** | +0.6070 | +164% | +1,831% |

S-minimizing mode = **int8** (ΔS 0). **net-S winner among sub-8-bit grids = NONE.**

## VERDICT: PTQ-COLLAPSES (no naive sub-8-bit grid lowers S)

**No naive post-hoc bit-shrink yields a net score win.** Every sub-8-bit grid RAISES S, monotone
and accelerating (int7 +0.0165 → int4 +1.99). The cause is structural: the contest weights d_seg by
**100×**, so even int7's small relative d_seg spill (+8.0%, +0.000207 absolute → +0.0207 in the
seg-term) outweighs its rate saving (−0.0069). The d_pose spill compounds this (pose is far more
fragile to the coarse grid: int4 d_pose explodes 322×, the weights fall outside the int4 representable
range).

**The rate axis IS real (the −0.022/−0.029 promise holds on bytes):**
- int4: 46,590 B (−47.7%, **Δrate −0.0283**) — matches the high end of the prompt's −0.022/−0.029.
- int5: 57,475 B (−35.5%, Δrate −0.0211); int6: 68,400 B (−23.3%, Δrate −0.0138).
- FP4 (E2M1) is DOMINATED on BOTH axes here: it cuts fewer bytes (~−28%, 64 KB) than int4 (47%) AND
  spills d_seg far more than int6 (fp4_mixed S=0.835 vs int6 S=0.447). The 15 distinct E2M1 levels
  zigzag-encode less sparsely through the int8 codec store than int4's tighter range.

The byte win is mechanical and confirmed; the distortion-hold is the blocker. **The −0.022 to −0.029
rate lever exists but is NOT bankable by post-hoc PTQ — it requires FP4/low-bit QAT/LSQ** (train the
decoder to tolerate the coarse grid). This is the next gated step; it was NOT built in this $0 smoke.

## How this is NOT a duplicate (SEARCH-AND-FAMILIARIZE)

Predecessors `experiments/probe_fp4_dseg_hold_smoke.py` + `probe_fp4_pctl_retest.py` reached the same
NO-GO but on the **`from0_ab_v2_n96` basin** (76,592 B, 96-pair memorized operating point) with
**estimated** int-N bytes (nbits/8 scaling that "ignores brotli interplay"). This probe:
(1) ran on the **G3-validated trustworthy bc20 n600 basin** (89,136 B, local-CPU ≈ contest-CPU);
(2) used **MEASURED** archive bytes (real codec re-encode + brotli at every bit-width);
(3) ran the **full 600-pair** authority eval (not a 96-pair subset). Same paradigm verdict, now on the
trustworthy basin with real bytes — confirming the register's μ1/Φ2 "needs FP4-QAT + d_seg-hold check".

## Recursive-self-review correction (NO FAKE)

The probe's first auto-headline said "PTQ-HOLDS: int7 … a NO-QAT partial rate win" — this was
MISLEADING and was corrected before landing. int7 passes a ±10%-relative d_seg/d_pose *hold* gate but
its empirical **S goes UP (+0.0165)** — a relative-hold is NOT a score-win criterion under the 100×
d_seg weight. The verdict logic now uses the correct **net-S (ΔS < 0)** criterion; no bit-width
satisfies it. `reports/fp_shrink_ptq_bc20_n600.json::verdict_logic_corrected` records the fix.

## Next step (the unblock)

Sub-8-bit rate win → S below 0.3779 requires **FP4/int4 QAT or LSQ** on the bc20 basin: train (or
short-finetune) the decoder with fake-quant in the loop so the weights land on the coarse grid without
the d_seg/d_pose spill. Helpers exist (`src/tac/quantization.py` FakeQuantFP4 / LSQScale / Uint8STE;
`tac.post_hoc_weight_shrink` for the post-hoc grids). The Δrate budget at int4 is −0.0283; if QAT
holds d_seg/d_pose within ~+0.0003 d_seg (the seg-term break-even vs the rate saving), the net win is
real and clears toward sub-0.15 per the small-basis register's reachability matrix
(`.omx/research/small_basis_optimization_register_20260615.md` μ1/Φ2).

## 6-hook wire-in (Catalog #125)

1. sensitivity-map: N/A (advisory diagnostic; bit-width sensitivity recorded in JSON byte_axis).
2. Pareto: ACTIVE — the (bytes ↓, d_seg/d_pose ↑) tradeoff curve is a Pareto frontier; int8 is the
   S-optimal corner under post-hoc PTQ.
3. bit-allocator: ACTIVE — `tac.post_hoc_weight_shrink` is a reusable bit-allocator primitive (the
   per-tensor int-N / E2M1 grids); the QAT next-step consumes it.
4. cathedral autopilot dispatch: N/A (no archive-deployable artifact; PTQ collapses).
5. continual-learning posterior: this memo + JSON are the empirical anchor; updates the register's
   μ1/Φ2 "needs QAT" with the trustworthy-basin measured confirmation.
6. probe-disambiguator: ACTIVE — this probe IS the disambiguator for "is the −0.022 rate win bankable
   by post-hoc PTQ?" → NO; QAT required.

mission_contribution: frontier_breaking_enabler (rate-lever feasibility; routes the win to QAT).
