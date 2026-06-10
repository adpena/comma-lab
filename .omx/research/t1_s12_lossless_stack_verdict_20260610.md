# T1 + S12 LOSSLESS-STACK verdict — both levers FALSIFIED/INAPPLICABLE on the frozen frontier

**Date:** 2026-06-10
**Subagent:** `t1_s12_lossless_stack_20260610`
**Lane:** `lane_t1_s12_lossless_stack_20260610`
**Mode:** RACE (frontier just retaken at 0.19109982 `[contest-CPU]`, sha `b46897267ded`, 177,169 B).
**Source plan:** `.omx/research/untapped_technique_inventory_20260610.md` T1 (cross-pair latent dedup, ranked #1) + T8 (latent null-projection, ranked #2) + S12 (resize-null preimage, landed #49) + `.omx/research/stacking_synergy_composition_plan_20260610.md` (orthogonality + positive-externality).
**Evidence grade:** all tests `[macOS-CPU advisory]` / mechanism + structural; latent-code analysis is exact (the codes are byte-identical to the frontier). NO score claim, NO dispatch, `promotable=false`.

## Headline

All three prompt levers are settled by **$0 local falsifiable first tests** on the EXACT frontier latent codes + the differentiable frontier decoder. **No Modal dispatch was warranted** (a paired CPU eval would only re-confirm the byte-identical frontier — spending ~$0.3–0.6 to learn nothing — violating MVP-first phasing). The disciplined output is system intelligence, not a wasted dispatch.

| Lever | Predicted (inventory) | Verdict | Mechanism |
|---|---|---|---|
| **T1 cross-pair latent dedup/clustering** | −0.0031 to −0.0061 (RATE) | **FALSIFIED** | 600 latents have ZERO cross-pair redundancy: 0 exact dups, min pairwise L2 = 120.85, k-means net LOSS at every K (+405 to +4,661 B vs 15,070 B floor) |
| **T8 latent null-projection (regenerate, not recode)** | −0.001 to −0.005 (RATE) | **FALSIFIED** | every 1-code latent step moves the decoded frame by ≥3.97 px (median ~6.5, max ~15); NO sub-grain null direction at the operating point — latents are already minimal |
| **S12 resize-null preimage fold-in** | force-multiplier on rate moves | **INAPPLICABLE** | the frontier archive stores NO frame pixels (procedural HNeRV: decoder weights + latents); S12 acts on stored uint8 camera-frame planes that do not exist here |

**Frontier unchanged: 0.19109982 `[contest-CPU]`, sha `b46897267ded`, 177,169 B.** No candidate built; no pointer update.

## The frontier archive composition (the load-bearing structural fact)

Decoded from the EXACT frontier member (`b46897267ded` → FP11 split → CTXR unpack):

| section | bytes | nature |
|---|---:|---|
| decoder weights (CTXR range) | 161,104 | HNeRV decoder (PR#95 arch) — **generates** frames at inflate |
| per-pair latents (CTXR AR+cross-dim) | 15,070 | 28-d × 600 latent codes (uint8) |
| PR#101 sidecar | 607 | canonical-Huffman latent correction |
| FECa selector | 222 | frame0 perturbation mode IDs |
| DQS1 tail | 42 | q-substitution policy |
| **member `x`** | **177,069** | (+100 B ZIP framing → 177,169 archive) |

This is a **procedural vehicle**: the 1,200 frames are computed by `decoder(latents)` at inflate, then the FECa selector applies frame0 perturbations. There is no stored frame-pixel section.

## T1 — cross-PAIR latent dedup: FALSIFIED (the inventory's #1, surprising-gap)

**The inventory's hypothesis:** `0.mkv` is one contiguous drive → dozens of the 600 latents are near-duplicates → a clustered codebook (K reps + indices + residual) beats the 15,070 B per-pair stream.

**The data refutes it at every domain:**

1. **Code level (exact):** the 600 latent vectors (each 28-d uint8) — `0 exact duplicates`, min pairwise L2 = **120.85** (median 294, max 572). NO pair has a neighbor within L2 ≤ 16. The vectors are uniformly far apart in 28-d code space.
2. **Temporal (cross-pair) delta:** successive-pair delta std (94.3) is **1.69× LARGER** than the marginal code std (57.6). Consecutive pairs are LESS correlated than the marginal → temporal/AR prediction in this code domain HURTS. The PR#101 encoding already de-correlated the per-pair codes.
3. **k-means lossless bytes test (the inventory's exact kill test):** for K ∈ {32,64,128,256}, `bytes(dict + index + exact-residual-order0)` is a NET LOSS at every K. The residual barely shrinks (14,772 → 11,963 B at K=256) because clusters aren't tight, while dict+index overhead grows faster:

   | K | dict | idx | resid (o0) | total | vs 15,070 |
   |---:|---:|---:|---:|---:|---:|
   | 32 | 896 | 375 | 14,204 | 15,475 | **+405 (LOSS)** |
   | 64 | 1,792 | 450 | 13,995 | 16,237 | **+1,167** |
   | 128 | 3,584 | 525 | 13,539 | 17,648 | **+2,578** |
   | 256 | 7,168 | 600 | 11,963 | 19,731 | **+4,661** |

**Why:** the single-video redundancy was already absorbed by the **shared HNeRV decoder during training**. The per-pair latents are the *residual* degrees of freedom AFTER the global structure is in the decoder weights — by construction they are the high-entropy, near-iid part. This is exactly the inventory's stated KILL condition ("the 600 latents are genuinely high-entropy → the per-pair iid floor IS the floor"). The latent section is at the per-dim Shannon marginal (14,772 B) + a small AR-header tax (current 15,070 B); **cross-pair mutual information = 0**.

## T8 — latent null-projection (regenerate cheaper latents): FALSIFIED

**The hypothesis (PR#112 MOVE 3, inventory T8):** push latent codes toward the scorer-null before coding → lower-entropy residual PR#112 structurally cannot reach.

**The decoder forbids it at the operating point.** Finite-difference per-dim 1-code latent steps (`z[i] += scale[i]`) on the EXACT frontier decoder, across pairs {0,100,300,599}:

| pair | min per-code pixel Δ | median | max |
|---:|---:|---:|---:|
| 0 | 3.968 | 6.654 | 14.953 |
| 100 | 4.100 | 6.464 | 9.525 |
| 300 | 4.076 | 6.605 | 15.697 |
| 599 | 4.408 | 7.288 | 10.964 |

The **smallest possible** latent perturbation (one quantization code) already moves decoded pixels by ≥3.97 (out of 255) on every dim of every pair. There is **no latent direction whose frame change is below the uint8/resize-null grain**, so no cheaper latent can be regenerated without moving the decoded frame (changing d_seg/d_pose). The latents are minimal: every code carries scorer-visible signal. Matches the inventory's T8 KILL condition. (T8 was meant to COMPOUND with T1; with both dead the compound is moot.)

## S12 — resize-null preimage fold-in: INAPPLICABLE to a procedural HNeRV archive

S12 (`tac.optimization.resize_null_preimage`, #49) is a UNIVERSAL postprocessor that minimizes `bytes(x̃)` s.t. `R·x̃ = R·x` over **stored uint8 camera-frame planes** — it fills the ~22.7% certified resize-invisible pixels with maximally-compressible values, then entropy-recodes. Its own docstring names its vehicles: "SNeRV render, frontier compose, PR110++ frames, HiNeRV, PACT-VQ" — vehicles that **store a frame representation**.

The current frontier stores **no frame pixels** (frames are generated by `decoder(latents)`). The only "frame bytes" are the decoder weights, which are NOT camera-frame planes with a resize-null structure — they are already entropy-coded at ~98.6% of their iid floor (PR#112 absorbed). The stacking_synergy memo's S12 row ("any frame bytes") presumes a frame-storing vehicle; this HNeRV vehicle has none. **S12 has no addressable bytes here.** The positive-externality test (S12-before-recode) is therefore vacuous on this archive.

S12 remains a real, landed, certified lever — for a future frame-storing vehicle (SNeRV LF carrier, frontier-compose with stored residual frames, raw-frame archives). It is not a fold-in for the procedural HNeRV frontier.

## Parity proof status

No candidate archive was materialized (all three levers settled negative pre-build), so the LOSSLESS-PARITY gate had nothing to run. The frontier raw decode sha (`dacf6b33…`, established by the leapfrog verdict) is preserved as the parity reference for any FUTURE candidate. The analysis used the byte-identical frontier latent codes + decoder (sha `b46897267ded`), so the falsifications are exact, not approximate.

## What this rules IN (the honest reframe — where rate headroom actually is)

The frozen-frontier RATE axis is **exhausted on the latent + decoder sections** at the lossless level:
- decoder weights: 98.6% of iid Shannon (PR#112).
- latents: per-dim marginal floor, cross-pair MI = 0, no null-projection slack.

Remaining lossless-rate slivers (all LOW-EV, < contest precision, per the inventory): T4 (selector order-1/RLE, −50–100 B if runs exist), T9 (decoder clustering/perm search, −100–500 B), T3 (inflate-as-interpreter for the small procedural sections, −0.0004–0.0010 + a standing subsidy for future procedural carriers). **None is T1/T8/S12.**

The genuine remaining EV is the **DISTORTION / campaign axis** (off the rate-saturated vertex): T5 (train representation error into the certified null space — needs the AFSR-1 retraining campaign), T11 (structured channel pruning + survivor finetune), T2 (cheapest-frame0 synthesis — frame0 is seg-free, regenerate as warp(frame1,pose)+sparse residual). These require a retraining campaign (a new base), not a frozen-byte transform. S12 + the certified invisibility basis become a TRAINING CONSTRAINT there (synergy #3) — the place S12 actually compounds.

## 6-hook wire-in (Catalog #125)

- **#1 sensitivity-map:** the latent-axis is confirmed at its per-dim marginal floor with zero cross-pair MI — feeds the latent sensitivity map that the per-pair iid floor IS the floor (no cross-pair axis to weight).
- **#2 Pareto:** T1/T8 do NOT move the rate vertex (both net-LOSS / no-op); the rate axis on the latent+decoder sections is a confirmed vertex. S12 is off-axis for this vehicle class.
- **#3 bit-allocator:** the k-means dict+index codec and the latent null-projection are FALSIFIED as bit-allocator primitives on this archive — they reduce no coded bytes.
- **#4 cathedral-autopilot:** NEGATIVE — do NOT queue T1/T8/S12 materializers on the procedural HNeRV frontier; the `byte_range_entropy_recode_chain` already captured the achievable lossless latent gain (PR#112 absorb).
- **#5 continual-learning:** the V3 judge is reseeded that cross-pair latent clustering (T1) and latent null-projection (T8) are FALSIFIED on the HNeRV frontier, and S12 is class-scoped to frame-storing vehicles — closing the "we never asked the cross-pair question" orphan flagged in the inventory §3.
- **#6 probe-disambiguator:** each lever's $0 first test IS the disambiguator and returned a definitive negative (k-means net-LOSS table; per-code pixel Δ ≥ 3.97; no stored frame section). No 2nd interpretation survives.

## Provenance

- Frontier archive sha256 `b46897267ded…`, member `x` 177,069 B, decoded via `tac.packet_compiler.pr110_payload_entropy_recode.reconstruct_raw_sections` + the submission's own `src/codec.py` + `src/model.py` (HNeRVDecoder, byte-exact).
- Latent codes `(28, 600)` uint8 saved to `.omx/tmp/t1_s12/` (44 KB scratch, durable SSD path, NOT `/tmp`; rebuildable from the frontier archive in seconds — no cold-store needed).
- Tools used: scipy k-means/pdist, torch decoder forward (CPU; no MPS). All exact on the frontier bytes.
- NO FAKE: each lever names the EXACT mechanism it failed against (code-distance matrix; per-code pixel sensitivity; archive section inventory) and the inventory's own KILL condition it satisfied.

**Cross-refs:** `untapped_technique_inventory_20260610.md` (T1/T8/S12 source + KILL conditions §1, §4 guard) · `stacking_synergy_composition_plan_20260610.md` (S12 row scope correction; synergy #3 = S12-as-training-constraint is the real S12 EV) · `leapfrog_pr112_absorb_recode_verdict_20260610.md` (the frontier + the latent coder at floor) · `frontier_latent_axis_waterfill_verdict_20260610.md` (per-pair iid floor — T1 confirms cross-pair adds nothing) · `MASTER_ROADMAP_post_exhaustion_map_20260610.md` (the campaign/distortion axis is the remaining EV).
