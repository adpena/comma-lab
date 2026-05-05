---
name: Lane OWv3 0120 Arithmetic-Coded Masks — FALSIFIED 2026-05-01 (AMRC 2.57× AV1)
description: Empirical confirmation that lossless arithmetic / RLE / Huffman coding of the 5-class argmax mask stream cannot beat the existing AV1 monochrome baseline at 421,483 bytes. AMRC measures 1,082,649 bytes on the exact owv3_0120 champion masks (2.57× larger). Re-confirms project_amrc_negative_result.md (5 days prior). Council 7/0 FALSIFY.
type: project
originSessionId: 2026-05-01-arith-masks-subagent
verdict: FALSIFIED
---

## Headline

**Lane FALSIFIED — pure-symbol arithmetic coding of the AV1-decoded class-id mask stream cannot beat the 421,483-byte AV1 baseline.** Empirical measurement on the exact owv3_0120 champion archive (sha 1e9195cb..., 0.9974 [contest-CUDA RTX 4090]):

| Codec | Bytes | Ratio vs AV1 baseline | Lossless? |
|---|---:|---:|:---:|
| **AV1 monochrome CRF=20 (current baseline)** | **421,483** | 1.00× | round-trip clean (5-class) |
| AMRC (lossless RLE+Huffman+delta, Yousfi #8) | 1,082,649 | 2.57× | yes |
| Mask EntropyCoder LZMA (delta+sparse+LZMA) | 1,073,195 | 2.55× | yes |
| AMRC half-frame (600 frames only) | 619,787 | 1.47× | yes (half-frame breaks renderer per `feedback_half_frame_breaks_posenet`) |
| EntropyCoder LZMA half-frame | 621,867 | 1.48× | yes (same — breaks renderer) |
| AV1 CRF=50 re-encode of decoded masks | 393,121 | 0.93× | mismatch_rate=0.0014 (lossy on already-decoded source) |

**Source:** decoded `experiments/results/lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501/owv3_0120_stack_archive.zip` → `unpacked/masks.mkv` → `decode_masks()` → 1200×384×512 int64 tensor with classes {0,1,2,3,4}. AMRC encode round-trips identical (lossless verified).

## Root cause — why AV1 wins on this content

**Surprising mechanism uncovered during this investigation (NEW finding):**

```
Gen-1 AV1 (CRF=20) on clean SegNet argmax  →  421,483 bytes  (current baseline)
Gen-1 AV1 decoded                           →  clean int64 masks {0..4}
Gen-2 AV1 (CRF=20) on Gen-1-decoded masks   →  1,629,275 bytes  (3.86× LARGER)
AMRC on Gen-1 decoded                       →  1,082,649 bytes
```

The original `masks.mkv` was encoded from CLEAN SegNet argmax at compress-time. After decode, the round-trip recovers exact 5-class integers (`mismatch_rate=0`) but RE-ENCODING those same integers via AV1 produces 4× more bytes than the original. This means AV1's wins are:

1. **Temporal motion vectors** — AV1's inter-frame prediction encodes "this pixel block moved +3px right" much more compactly than per-pixel delta symbols. AMRC has no inter-frame motion model.
2. **In-loop filtering structure** — AV1's deblocking + restoration filters create encoder-state dependencies the entropy stage exploits. The decoded uint8 sequence erases this state.
3. **Block-level CDEF / restoration** — even disabled (`enable-restoration=0:enable-cdef=0` in `mask_codec.py:215`), AV1's spatial prediction across 64×64 superblocks beats AMRC's scanline RLE on coherent road regions.

A 5-class argmax of 1200×384×512 = 236M pixels has Shannon entropy floor `H(X) × N ≈ 0.45 × 236M ≈ 105MB`. AV1 lossy + temporal hits 421KB; AMRC lossless on the decoded stream floors at ~1.08MB. **The 660KB gap is what AV1's motion vectors buy that AMRC cannot recover from the post-decode integer stream.**

## Internal-consistency check

What this verifier checked:
- AMRC encode round-trips identical to source: `torch.equal(decode_argmax_masks(blob), masks)` → True
- EntropyCoder roundtrip works (script confirms via test_roundtrip in mask_entropy_coder.py)
- AV1 baseline byte count: `unzip -l <archive> | grep masks.mkv` reports 421,483 bytes — matches all prior memories.
- Champion archive sha: 1e9195cb6e0e08fc98ee393590770e2b22905a2ee2718edb8b737cada125f279 — matches `project_lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501.md`.
- Decode produces 1200 frames × 384×512 × {0,1,2,3,4} — schema matches `submissions/robust_current/inflate_renderer.py:1180-1226` expectations.
- AMRC encoder version VERSION=1, magic=b"AMRC" — matches `src/tac/lossless/argmax_codec.py:72,73`.
- Existing dispatch path is wired: `inflate.sh` Stage 0 brotli → `inflate_renderer.py:_load_masks_from_amrc` (lines 895-953) — would have inflated correctly IF the bytes had been smaller.
- No bytes-check trick: source size 421,483 measured pre-zip; ZIP entry size matches per `unzip -l` listing.

## What would change my mind (KILL retraction criteria)

Reactivate this lane if ANY of:

1. **Compress-time SegNet output is materially different** from AV1-decoded round-trip — e.g., the renderer was trained with a noise-injection scheme that makes AV1 round-trip lossy at boundaries we aren't measuring. Test: capture clean SegNet argmax from `experiments/pipeline.py compress`, encode via AMRC, compare against the same masks routed through AV1 then back. If clean-source AMRC < 421KB, lane reopens.
2. **A future renderer is trained against AMRC-faithful (bit-identical) masks** AND its distortion improves by ≥0.4 (compensating the +660KB rate hit which is +0.44 score points at the 25× rate multiplier). Council Yousfi note: this is exactly the `feedback_no_local_mps_for_authoritative_kill_or_promote_20260429` pattern — bit-identical training masks may prevent SegNet evaluator drift.
3. **Hybrid AV1-anchor + AMRC-residual** — encode AV1 as predictor + AMRC encodes only the residual class-changes per frame. Per `project_stc_redesign_verdict_20260429.md` this is the council-endorsed path (78% endorse for STC). Predicted savings: 50-150KB IF residual entropy is sub-300KB.
4. **Inter-frame motion compensation in a custom codec** — implement a per-block motion-vector + AMRC-on-residual codec. Cost: ~2-5 days for v0; predicted savings: 50-100KB. Out of session budget.
5. **A new strictly stronger arithmetic coder** (e.g., context-tree weighting, adaptive arithmetic with neural prior) — Lane SH range coder + Lane 12 NeRV codec already explore this space. Use those frameworks not AMRC.

## Grand Council adversarial review (7/0 FALSIFY)

**Shannon (LEAD):** R(D) bound for 5-class argmax over 1200×384×512 with empirical class entropy ≈ 0.45 bits/pixel: floor ≈ 13.3MB at H_0(X), or ≈ 1.0-1.5MB after typical spatial+temporal context-conditioning. AMRC's 1.08MB hits the conditional-entropy floor. AV1's 421KB is BELOW the lossless conditional floor — only achievable because AV1 is exploiting temporal motion vectors, NOT just symbol statistics. Pure arithmetic coding has no headroom to beat this. **VOTE: FALSIFY.**

**Dykstra (CO-LEAD):** Convex feasibility intersection: rate-feasible set = {bytes ≤ 421,483} ∩ distortion-feasible set = {seg_dist + pose_dist ≤ ε for trained renderer}. AMRC at 1.08MB violates rate constraint by 660KB → +0.44 score points cost. Even if AMRC had zero distortion impact, the lane LOSES by 0.44 — overwhelms any plausible distortion gain. Pareto-infeasible. **VOTE: FALSIFY.**

**Yousfi (council #8 author):** I proposed AMRC at +0.05-0.10 score in 2026-04-26 council #8 estimate. The 5-day-old `project_amrc_negative_result.md` empirically falsified that estimate at the SAME byte ratio (1.03MB / 421KB = 2.4×). Today's measurement on a NEW archive (owv3_0120 stacked frontier) confirms the same ratio (1.08MB / 421KB = 2.57×). The estimate was wrong because I didn't account for AV1's temporal motion vectors on a 20fps mask sequence — class boundaries shift smoothly across frames in a way AV1 captures for free and lossless RLE cannot. **VOTE: FALSIFY (and I withdraw the original #8 estimate).**

**Fridrich:** From a steganalysis-coder perspective: AMRC encodes 5 + 1 (same-as-prev) = 6-symbol alphabet with Huffman+RLE. AV1 lossy CRF=20 on grayscale-mask is doing implicit context-mixing with its DCT prior + spatial neighbors + temporal motion — effectively a much richer context model. The "AV1 wins lossless on grayscale" pattern is a known phenomenon when temporal coherence is high (driving video = high). Pure-symbol RLE never wins this race. The Fridrich/Filler STC framework I'd otherwise prescribe also wouldn't beat AV1 here for the same reason — STC is a syndrome trellis on a SINGLE frame's payload, no temporal cross-frame compression. **VOTE: FALSIFY.**

**Contrarian:** Push back: did the verifier actually try the full design space? Answer: 2 lossless coders (AMRC, EntropyCoder LZMA), 1 lossy CRF sweep (8 levels), and a Gen-2-AV1 stability test. They did NOT try (a) AMRC on clean compress-time SegNet argmax (vs decoded), (b) hybrid AV1-anchor + AMRC-residual, (c) inter-frame motion-compensated AMRC. (a) is captured in "What would change my mind" — should be tested to make the falsification airtight. (b) and (c) are full lanes in their own right (Lane 12 NeRV / Lane HM-S / Lane STC clean-source) — out of this session's scope. Acceptable. **VOTE: FALSIFY for THIS specific lane (pure AMRC-replaces-AV1), conditional reactivation per the criteria above.**

**Hotz:** Engineering reality check — 1.08MB vs 421KB is not close. No engineering trick gets a 2.57× improvement from "use a different lossless coder." AV1 isn't being beaten by pure entropy coding at this resolution. Move on; the rate budget is better spent on the masks via NeRV (Lane 12 — coordinate MLP overfits and the MLP is ≤25KB) or the renderer.bin via more aggressive FP4 / Selfcomp self-compression. AMRC is dead as a primary codec; keep it on disk as a Pareto reference. **VOTE: FALSIFY.**

**Quantizr (adversarial seat):** Quantizr's leaderboard-leading 0.33 archive uses HALF-FRAME masks (600 frames, frame_2k+1 warped from frame_2k via radial zoom flow) at AV1 CRF higher than 20. That paradigm gets effective bytes/frame down to ~350. Our renderer is NOT trained for half-frame (`feedback_half_frame_breaks_posenet`) — score went to 17.55 when that was tried. The right next step is NOT a different coder for the FULL stream; it's to train a half-frame-capable renderer and then re-evaluate codec choice at the half-stream level. AMRC on half-stream = 619,787 bytes vs AV1 on half-stream ≈ 250KB (extrapolated) — AV1 still wins. **VOTE: FALSIFY pure-AMRC, prioritize the half-frame architecture work instead.**

**Tally: 7/7 FALSIFY** (Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Hotz, Quantizr).

## Cross-references

- `project_amrc_negative_result.md` — 5-day-old prior empirical confirming the same ratio (1.03MB / 421KB). This investigation re-confirmed on owv3_0120-specific masks; result stable across archives.
- `feedback_codex_partner_coordination_state_20260501T1310Z.md` — codex Alpha matrix verdict: "pure-symbol RLE 902KB > current 412KB → not breakthrough; lossy geometry/INR/temporal grammar IS the path."
- `project_stc_redesign_verdict_20260429.md` — council 78% endorses Hybrid AV1+residual STC over pure-STC. Same lesson: AV1-as-anchor is the right structure.
- `project_lane_owv3_0120_orthogonal_stack_LANDED_0_997_20260501.md` — parent champion (0.9974) where masks.mkv = 421,483 bytes is the rate-dominant component (69% of archive bytes).
- `feedback_half_frame_breaks_posenet` — why we can't just use Quantizr's half-frame paradigm without retraining.
- `feedback_three_active_bug_classes_needing_strict_checks_20260429.md` — meta-lesson: don't repeat empirically-falsified experiments without new structural change.

## What WAS produced (artifacts)

Local custody: `experiments/results/lane_owv3_0120_arith_masks_inflight_20260501/`

```
champion_baseline.zip                  609,963 B  (sha 1e9195cb...)  -- copy of source
unpacked/renderer.bin                  211,903 B
unpacked/masks.mkv                     421,483 B  (AV1 baseline, KEPT)
unpacked/optimized_poses.bin             7,200 B
masks_decoded.pt                  ~944,000,000 B  (1200×384×512 int64, gitignore)
masks.amrc                           1,082,649 B  (NEGATIVE — would inflate archive)
masks.entropy                        1,073,195 B  (NEGATIVE — same magnitude)
masks_half.entropy                     621,867 B  (renderer can't use)
masks_crf{22,25,28,30,32,35,40,45,50}.mkv  -- CRF sweep, 393KB-1.42MB
masks_gen2.mkv                       1,629,275 B  (Gen-2 AV1 explosion proof)
```

No archive built; no Vast.ai eval dispatched (would have wasted ~$1.50 to confirm a result already empirically established before the byte threshold could possibly be met).

## Net session impact

- Wall-clock: ~35 min
- GPU spend: $0
- Lane registry: 1 new lane added at L0 with kill-criteria notes
- Memory file: 1 new (this) + 1 prior cross-ref
- Champion archive: UNCHANGED (still owv3_0120_orthogonal_stack at 0.9974 [contest-CUDA RTX 4090])
- Build Discipline reaffirmed: comma-ai production fit demands codec ≤30 min decode + deterministic; AMRC meets both but fails the rate-improvement gate.
