# GENIUS BLIND-SPOT PROBE A — frontier-rate-cut vs small-basis train (anchoring/sunk-cost test)

- **Date:** 2026-06-17 (UTC 155535Z)
- **Axis:** `[macOS-CPU advisory]` byte-level forensics + closed-form score math. NO new score claim (no retrain, no eval dispatch; the only eval cited is the EXISTING 600-sample contest-CPU anchor already on the frontier pointer).
- **Spend:** $0. CPU only (live MPS train untouched).
- **Lane:** `lane_frontier_rate_cut_anchoring_probe_a_20260617` (research_only=true).
- **Verdict:** **The frontier-rate-cut is NOT a faster sub-0.15 path. A lossless byte re-code WALLS — the payload is already at its entropy floor. The sub-0.15 byte budget (34.8% cut) is UNREACHABLE without changing the decoded tensors (i.e., a retrain / smaller model). The small-basis train (or any d_seg/capacity attack) remains the real sub-0.15 path; the anchoring hypothesis is REJECTED on the merits, but not because we were anchored — because the rate axis is genuinely exhausted.**

## 1. Frontier decomposition (exact)

Frontier = `lane_pr110_payload_entropy_recode_20260610`, archive sha `b46897267d…`, **177,169 B**, contest-CPU **S = 0.19109982** (600 samples, call_id `fc-01KTRAYS68F3S0YWFT0CX35HDG`).

Archive = single ZIP member `x` (ZIP_STORED, method=0, 177,069 B payload + 100 B ZIP overhead). Member is self-compressed; ZIP adds nothing.

| Term | Formula | Value | Share |
|---|---|---:|---:|
| seg | 100·d_seg (d_seg=0.00055978) | 0.055978 | 29.3% |
| pose | √(10·d_pose) (d_pose=2.942e-5) | 0.017152 | 9.0% |
| **rate** | 25·177169/37,545,489 | **0.117970** | **61.7%** |
| **S** | | **0.19109982** | |

The prompt's "rate ~0.118, RATE-BOUND" framing is **correct**: rate is 61.7% of S and d_seg/d_pose are excellent. seg+pose floor = **0.073130** (so sub-0.15 IS byte-reachable in principle — at ≤115,444 B).

**Byte budgets (holding d_seg/d_pose fixed):**
- sub-0.15 → bytes ≤ **115,444 B** → cut **61,725 B = 34.8%**. (Prompt said <115.6KB / 35% — confirmed.)
- any-win → bytes < 177,169 B → already AT the floor, **0 B headroom**.

## 2. Payload grammar (single member `x`)

```
FP11 | u32 source_len | source_payload(CTXR) | u16 sel_len | selector(FECa) | DQS1 tail
CTXR | u8 ver | u24 dec_sec_len | u24 lat_sec_len | u24 sidecar_len | dec_sec | lat_sec | sidecar
```

| Section | Bytes | % of payload | Codec already applied |
|---|---:|---:|---|
| **dec_sec** (decoder weights) | 161,104 | **91.0%** | PR#112 `codec_ctx` per-tensor adaptive 256-ary constriction range coder |
| lat_sec (per-dim AR latents) | 15,070 | 8.5% | PR#112 per-dim AR coder |
| sidecar | 607 | 0.3% | canonical-Huffman length-ranked |
| selector (FECa) | 222 | 0.1% | fixed-Huffman K16 |
| dqs1 tail | 42 | 0.0% | verbatim |

**The rate term is 91% decoder-weight bytes.** This is a model-size problem, not a packing problem.

## 3. The lossless re-code attempt (the decisive measurement)

The frontier is ITSELF already a lossless entropy recode (recode_manifest.v1): it replaced PR#101's split-Brotli + raw-LZMA1 with PR#112's SOTA `codec_ctx` range coder, saving only **−1,326 B (−0.75%)**. I attempted to push further. Per-section order-0 entropy + best-of-{brotli-11, lzma-9-extreme, bz2-9, zlib-9} re-compression:

| Section | bytes | order-0 entropy | best recompress Δ |
|---|---:|---:|---:|
| dec_sec | 161,104 | **7.999 bits/byte** | brotli +5 B (all coders GROW it) |
| lat_sec | 15,070 | 7.989 bits/byte | brotli +4 B (all GROW) |
| sidecar | 607 | 7.710 | +4 B (all GROW) |
| selector | 222 | 7.055 | +4 B (all GROW) |
| dqs1 | 42 | 4.851 | +4 B (all GROW) |

**Every section is at ≈8.0 bits/byte (maximally coded). No general-purpose compressor can shrink ANY section — every attempt makes it larger.** The dec_sec is 161,104 B against an order-0 floor of 161,082 B (22 B slack), and the range coder's conditional model already beats naive order-0 by ~108 B across the payload. `context_partition_codec` (suggested for reuse) is a codec for the SegNet argmax partition L\*, a DIFFERENT source — it does not apply to decoder-weight bytes.

**Best achievable lossless re-code = 177,169 B = current frontier. 0 B further headroom.** The −1,326 B win was already banked.

## 4. Verdict & EV comparison

- **Can the frontier reach <115,444 B (sub-0.15) losslessly?** **NO.** That requires cutting 34.8% of bytes; the payload is at its entropy floor. A lossless transform recovers 0 of the 61,725 B needed.
- **Can it reach any win (<177,169 B) losslessly?** **NO.** Already at the lossless floor.
- **What is the lossless floor?** **≈177,169 B = the current frontier itself.**
- **Is the frontier-rate-cut a faster sub-0.15 path than the small-basis train?** **NO.** Sub-0.15 on the rate axis REQUIRES reducing the decoded decoder-weight bytes, which means a **lossy** change — a smaller/retrained model (or quantization to fewer bits per weight) that must hold d_seg/d_pose. That is NOT a byte transform; it is a (re)train + byte-close + paired CPU/CUDA eval — the same class of work as the small-basis train. There is no hours-long shortcut hiding in the existing frontier bytes.

**The anchoring hypothesis is honestly tested and rejected.** We were right to suspect under-pursuit, but the rate axis on the EXISTING frontier is genuinely exhausted (entropy floor reached by PR#112's recode). The real sub-0.15 levers are model-side: (a) a smaller decoder/basis that holds d_seg (the small-basis train, IF it byte-closes ≤115KB while holding d_seg≈0.00056 — note current base_ch20 small basis is d_seg-LIMITED at ~0.0026-0.0036, ~5-6× WORSE than the frontier's 0.00056, so it would need both a smaller decoder AND a major d_seg recovery); (b) lower-bit decoder quantization (FP11→fewer bits) holding d_seg/d_pose — a lossy rate lever NOT yet exhausted, but it is a retrain/QAT path, not a recode.

## 5. System-intelligence wire-in

- Sensitivity-map: rate marginal on the frontier is now anchored — decoder-weight bytes are at entropy floor; ∂(rate)/∂(lossless-recode) = 0. Bit-allocator should treat the frontier payload as incompressible and route all rate-lowering EV to lossy model-side levers (smaller basis / lower-bit quant).
- Probe-disambiguator: this memo IS the arbitration between "anchored, missed a fast byte cut" vs "rate axis genuinely exhausted" — math arbitrates: exhausted.
- Continual-learning: the −1,326 B PR#112 recode is the LAST lossless rate win on this frontier; future "recode the frontier" proposals are pre-answered NO by §3.
- Pareto: the rate constraint at this operating point is hard (entropy floor); the achievable frontier moves only by trading the d_seg/d_pose budget for fewer decoder bytes (lossy).
- N/A: cathedral-autopilot dispatch hook (advisory $0 probe, no dispatch).
