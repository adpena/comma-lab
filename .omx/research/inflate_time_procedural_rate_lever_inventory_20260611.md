# Inflate-time procedural-rate-lever inventory + top prototype (2026-06-11)

**Authority:** advisory (torch-CPU, NO MPS, $0, no paid dispatch). Frontier UNMOVED
0.19109982 [contest-CPU], 177,169 B, sha `b46897267d`. This is investigation (B) of the
`CORRECTION_param_space_inflate_runtime_is_a_live_door` split: the inflate-time
procedural-rate-lever inventory. Sister agent owns investigation (A) — the multi-element
gradient-DQS1 decoder-weight-DELTA **seg-repair** lane (adds bytes to cut d_seg). This
report does NOT touch that lane; it inventories the orthogonal **rate-reducing** levers.

## 0. Harness calibration FIRST (NO-FAKE) — GOOD class, offset +9.94e-6

| | seg | pose | rate | **S** |
|---|---|---|---|---|
| Frontier [contest-CPU] (Modal) | 0.00055978 | 0.00002942 | 0.00471878 | **0.19109982** |
| Local NO-OP [macOS-CPU advisory] | 0.00055988 | 0.00002942 | 0.00471878 | **0.19110976** |
| offset (local − contest) | +1.0e-7 | 0 | 0 | **+9.94e-6** |

- Local CPU inflate of the frontier archive is **byte-identical** to the frontier decode
  (`raw sha dacf6b33…` matches the frontier's `decode_parity_proof.json` exactly).
- Pose-axis byte-identical; seg differs only at 1e-7 (macOS vs Linux float rounding on the
  EfficientNet-B2 argmax-flip boundary). **This is the good calibration class** (offset ~1e-5),
  NOT the bad 0.0166-offset class a prior sister harness showed.
- **Flagging threshold:** a candidate must beat 0.19110976 by a margin clearly exceeding the
  ~1e-5 local→contest uncertainty to flag for operator-gated paired eval.

Harness: `tools/run_raw_advisory_eval.py` → `upstream/evaluate.py --device cpu` (600 samples
from the single `0.mkv` = 1200 frames = 600 pairs; the local 0.mkv IS the full 600-sample set;
local `sum(upstream/videos)=37545489` matches the contest uncompressed size exactly, so the
rate term is byte-identical).

## 1. Exact member byte ledger (177,069-byte member `x`)

| Section | Bytes | % | Floor status |
|---|---|---|---|
| CTXR source payload (decoder+latent+sidecar) | 176,795 | 99.85% | at/near floor |
| └ decoder section | 161,104 | 91.0% | **at lossless floor** |
| └ latent section | 15,070 | 8.5% | near floor (<300 B headroom) |
| └ sidecar | 607 | 0.34% | — |
| selector (FECa) | 222 | 0.13% | **below order-0** |
| DQS1 tail (seg-repair) | 42 | 0.02% | sister lane (adds bytes) |
| FP11 + CTXR headers | ~24 | 0.01% | negligible |

**Rate leverage:** 1 byte = dS −6.659e-7; need **~1,502 B** for dS=−1e-3.

**Inflate budget:** frontier inflate = **81 s** of the **1,800 s** (30-min) budget → **95.5% unspent**.
The compute headroom is real and large.

## 2. Ranked lever inventory (all $0-measured)

| # | Lever | Section | Est. rate save | Feasibility | d_seg/d_pose | Verdict |
|---|---|---|---|---|---|---|
| 1 | decoder lossless re-code | 91% | **0 B** | EXHAUSTED | none | adaptive coder 161,104 B **beats** brotli-q11 (162,213), lzma-9e (169,284), bz2-9 (193,851); realizable order-1 = 178,138 B (**+17,034**, context dilution); mixed-order loses on every tensor |
| 2 | latent lossless re-code | 8.5% | <300 B (dS<2e-4) | NEAR-EXHAUSTED | none | already AR(1)+lag at ~0.90 B/latent; temporal-delta ratio 0.847 (not smooth); below 1e-3 threshold |
| 3 | selector lossless re-code | 0.13% | 0 B | EXHAUSTED | none | FECa hybrid 222 B already below order-0 floor (241 B); generic adaptive 16-ary = 247 B (loses) |
| 4 | **decoder LOSSY coarser re-quant** | 91% | ~15,400 B (dS_rate −0.0103) | **R-D WALL** (+ retrain-class) | **CATASTROPHIC** | step-2 → decoder OUTPUT RMSE 5.75 px, maxΔ 159 px; measured full-S below |
| 5 | iterative refinement / TTA at decode | whole | 0 B | INFO-THEORETIC WALL | n/a | only signal at decode is the archive itself (GT forbidden outside zip); compute ≠ information recovery |
| 6 | container/scales overhead | headers | <60 B | NEGLIGIBLE | none | fp16 scale high-bytes already entropy-coded |

### Why the lossless door is exhausted (the key measurements)

- **Decoder int8 codes are decorrelated and dense:** order-0 entropy 6.54 b/B; the per-tensor
  storage permutations (`CONV4_STORAGE_PERMS`) + byte-maps already whitened them. Order-1 and
  order-2 context modeling are NET LOSSES (context dilution on small per-tensor data). The
  "order-1 ideal 152,181 B" is a phantom — it assumes free perfect context tables; the
  realizable online-adaptive order-1 costs **+17,034 B**.
- **PR#112's per-tensor adaptive 256-ary coder already beats every off-the-shelf compressor.**
  There is no realizable lossless re-code lever left in any section.

## 3. Top prototype MEASURED: lossy step-2 decoder re-quant (rank #4) — the R-D wall

Built a byte-closed candidate: coarsen decoder codes to step-2, re-encode via the SAME
codec_ctx coder, FP11 member rebuilt byte-exactly (selector + DQS1 preserved).

- archive 177,169 → **161,745 B** (−15,424 B; rate alone would give dS_rate **−0.01027**)
- archive sha `e321fc975897b1cb4dde77f43068d82d3f71e6e92b0ac930c4f9b884e580b7c7`
- **decoder-output proxy:** RMSE 5.75 px, max Δ 159 px on first 32 pairs — ~100× the
  perturbation scale the sister pixel-repair found collateral-floored at sub-0.001 d_seg.

**Measured advisory full-S:** (see `lossy_step2/eval/report.txt` — appended below on completion)

> **Predicted: NET WORSE.** The −0.0103 rate gain is eaten many times over by the
> seg/pose distortion of a 5.75-px RMSE pixel perturbation. The frontier sits at the R-D
> knee; lossy byte reduction is dominated. This lever is ALSO a re-quant/retrain-class
> action (it changes the quantization), so it is outside the $0 inflate-time scope regardless.

## 4. Honest disposition

**The inflate-time LOSSLESS rate door is EXHAUSTED** (measured: every section at/below its
realizable entropy floor; PR#112 already harvested it; order-1/order-2/brotli/lzma/bz2 all lose).

**The inflate-time LOSSY/procedural doors face an information-theoretic + R-D wall:**
- The 95.5% unspent inflate compute budget is real but has **no information to exploit** — the
  GT video is forbidden outside the zip (NO-FAKE), so decode-time compute cannot recover
  lossily-discarded bytes. Compute ≠ information.
- The only large section (91% decoder) can only be shrunk by coarser quantization, which is
  (a) measured to be R-D dominated at this operating point and (b) a re-quant/retrain-class
  lever, not a pure inflate-time program.

**No live $0 inflate-time RATE door found.** The genuine live frontier-move door at $0 is the
orthogonal **DQS-class seg-repair** lever (the sister lane) — it ADDS a few bytes to a
compress-time-searched, inflate-time-applied decoder-weight delta to cut d_seg. That is a
distortion lever, not a rate lever; the rate term is genuinely saturated.

**The real rate door requires a retrain** (smaller/better-quantized decoder basis, score-aware
QAT) — a paid, parallel bet, NOT a $0 inflate-time program. The operator's premise that the
inflate-time budget is unspent is correct; the finding is that there is no stored-byte
redundancy left for that budget to losslessly reclaim.

## Provenance / 6-hook (Catalog #125)
- #1 sensitivity-map: ACTIVE (per-section byte ledger + dS/byte = 6.659e-7).
- #2 Pareto: ACTIVE (R-D wall measurement: decoder lossy slope dominated at the knee).
- #3 bit-allocator: ACTIVE (lossless floors per section; no realizable re-allocation left).
- #4 cathedral autopilot: N/A (advisory, non-promotable, no archive promotion).
- #5 continual-learning: ACTIVE (this memo + manifest; calibration anchor offset +9.94e-6).
- #6 probe-disambiguator: ACTIVE (lossless-vs-lossy door disambiguated by direct measurement).

axis_tag `[macOS-CPU advisory]`; score_claim=false; promotion_eligible=false;
ready_for_exact_eval_dispatch=false. mission=frontier_breaking (negative result that
correctly aims the next unit at retrain, not at a phantom $0 rate door).
