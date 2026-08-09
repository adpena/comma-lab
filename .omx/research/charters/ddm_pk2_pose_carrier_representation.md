# ddm_pk2 — PR130 pose-carrier REPRESENTATION attack (the 12.24% section nobody has re-represented)

**Owner:** codex arm · **Base:** PR130 CPR1 · **Axis:** rate (pose section) · scorer-gated
**Authority:** `[macOS-CPU advisory]` · `score_claim=false` · no CUDA here.

## THE TARGET (MEASURED by MAIN 2026-08-09, reproduce before trusting)

PR130 archive 191,052 B. `models_raw` = `[u32 sem_len][u32 pose_len][semantic 40,252][pose 23,054][hpac 20,179]`.
Pose section marginal (exact leave-one-out on the reproduced bytes): **23,384 B = 12.24% of archive
= 0.0155704 S**. Pose distortion contribution ≈ 0.014025 S ⇒ d_pose ≈ 1.967e-5 (MEASURE it; do not
inherit this figure).

Decoded structure (`carrier_codec.decode_compact_carrier`, constants from `inflate.py:34-35`):

| component | object | coded size | bits/value |
|---|---|---:|---:|
| basis | `int8[CARRIER_DIM=12, 3, H=24, W=32]` = 27,648 values, canonical-Huffman | 104,135 bits = **13,017 B** | 3.77 |
| coefficients | `int12[N=1200, 12]` = 14,400 values, Rice/Golomb per-dim, delta-zigzag | 79,076 bits = **9,885 B** | 6.59 |
| scales | `f32[12]` ×2 + 12-byte header | ~104 B | — |

## WHAT IS ALREADY CLOSED — DO NOT RE-RUN

The **coder axis on this section is measured shut.** Split-stream re-coding of the pose section
(brotli-q11 / lzma variants, MAIN 2026-08-09) came out at **23,058 B — 4 bytes WORSE than the
23,054 B already in the archive.** Payload byte-entropy is 7.9817 bits/byte (22,989 B floor). The
section is canonical-Huffman + zigzag + bit-packed and sits at its own byte entropy.

**Therefore: only a REPRESENTATION change can move this section.** A coder race here is a
re-measurement of a settled negative. If you find yourself racing brotli/lzma/zstd/ANS on these
bytes as they stand, stop — that is the closed cell.

(Note the distinction that governs the sister token section: swapping the ENTROPY CODER under the
SAME model won −2,120 B on tokens, because the range coder there sits +1.85% over its model's
cross-entropy. The pose section has no such slack: it is AT its entropy. Different fact, same file.)

## THE BREAK-EVEN (this is a LOSSY axis — measure both terms or the row is void)

Shrinking the carrier changes d_pose. Saving B bytes buys pose-contribution headroom
`25·B/37,545,489`:

| bytes saved | ΔS_rate | permitted pose contribution | permitted d_pose | ratio vs current |
|---:|---:|---:|---:|---:|
| 4,000 | −0.002663 | ≤ 0.016688 | ≤ 2.785e-5 | 1.42× |
| 8,000 | −0.005327 | ≤ 0.019352 | ≤ 3.745e-5 | 1.90× |
| 11,527 (half) | −0.007675 | ≤ 0.021700 | ≤ 4.708e-5 | **2.39×** |
| 16,000 | −0.010653 | ≤ 0.024678 | ≤ 6.090e-5 | 3.10× |

Report every arm as the JOINT `ΔS = 100·Δd_seg + Δ√(10·d_pose) + 25·Δbytes/37,545,489`. An arm that
reports bytes without measured d_pose is not a row. d_seg must also be checked: the carrier feeds the
render, so a carrier change can move seg — measure it, do not assume 0.

## OPTIMAL FORM

**Reference form of the family:** low-rank + smooth-trajectory coding of a learned basis/coefficient
factorization. Our own prior instance is `tac.torch_vehicle.pose_film.encode_pose_section_lowrank`
(#140, `.omx/research/lowrank_pose_section_codec_landed_20260617.md` +
`pose_lowrank_CORRECTED_fidelity_20260617.json`) — measured 2.7× at MSE ≤ d_pose target, but on OUR
3,088 B FiLM section, a DIFFERENT object. **That number does NOT transfer** (L18 ancestor rule);
reuse the CODE and the METHOD, re-measure the RESULT.

**Provenance pins (verify each before use; a pin that does not reproduce is a STOP):**
- target archive sha256 `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`,
  191,052 B, at `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`
- reproduction record commit `12031094d9` (`.omx/research/ddm_pr130_reproduce_20260809/PR130_REPRODUCED_HERE.md`)
- closed-coder-cell receipt commit `f0a7ebf750` (`ANS_REAL_TABLE_MEASUREMENT.md`; the pose split-stream
  +4 B row is in the same reproduce dir)
- codec under test: `carrier_codec.py` + `inflate.py` in the READ-ONLY intake clone at
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/` — hash them at arm
  start and record the hashes in the receipt; if they differ from what this charter describes
  (`MAGIC=b"CPR1"`, `HEADER="<4sII"`, `CARRIER_DIM=12`, `CARRIER_H,CARRIER_W=24,32`), STOP and report.
- our reusable prior implementation: `src/tac/torch_vehicle/pose_film.py` (repo HEAD at arm start —
  record the commit you read).

Declared reductions:
- SCOPE (legal): n≥120 stratified-random pair subset for the ladder sweep; the WINNER must be
  re-measured at full n600 before any composed row.
- MECHANISM: none permitted. If any arm reduces the mechanism below its reference form (e.g. rank-1
  only, no residual, no per-dim quantizer search), declare it TOY-BRACKET and it cannot produce a
  family verdict.

Provenance pins: intake clone `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/`
is **READ-ONLY** (never edit, never `git add` inside). Reproduced archive
`/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`,
sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`, 191,052 B.

## THE LADDER (each rung = a measured (Δbytes, Δd_pose, Δd_seg, ΔS) row)

**A — COEFFICIENTS (9,885 B, 6.59 bits/value).** 1,200×12 smooth ego-motion-driven trajectories.
1. Measure the actual temporal spectrum + per-dim autocorrelation. Is the current delta-zigzag+Rice
   leaving predictable structure on the table? (This is a REPRESENTATION question — a smoother
   predictor is a different model, not a different coder.)
2. Low-rank across the 12 dims (SVD / joint factorization) — the dims are activations of ONE
   ego-motion process, so cross-dim rank is the hypothesis.
3. Spline/knot or acceleration-matched parametrization (our `tac.lie` ξ-curve + ddm_am1
   acceleration-matching crosswalk are the in-house priors) with a residual stream.
4. Per-dim quantizer search (the Rice `k` is chosen per dim already — is it chosen OPTIMALLY, and is
   the SCALE per-dim optimal for the d_pose the render actually needs?).

**B — BASIS (13,017 B, 3.77 bits/int8 over 27,648 values).** 12 RGB images at 24×32.
5. Spatial transform: DCT/wavelet/separable on 24×32 planes before coding — 3.77 bits/value on
   natural-image-like low-res planes suggests spatial correlation is uncoded.
6. Cross-plane structure: 12 dims × 3 channels = 36 planes. Rank across planes; chroma-vs-luma split.
7. Precision: does the basis need int8? Measure the d_pose response to int7/int6/int5 per-plane with
   per-plane scales (the canonical low-bit fixes: per-channel scale, outlier handling — NOT naive PTQ).

**C — CAPACITY.** CARRIER_DIM=12 is a PR130 constant nobody has derived for our purposes. Measure
d_pose vs CARRIER_DIM at fixed byte budget. This may dominate everything above; it is also the arm
most likely to require re-training the carrier, which is OUT OF SCOPE here — measure the response
curve from the existing basis (drop/zero dims by measured contribution) and report whether a
retrained smaller carrier is worth a separate arm.

Race A and B independently first (they are separate byte pools), then compose the winners and
re-measure jointly — do not assume additivity, MEASURE the composed row.

## HARD RULES

- Everything through the REAL `carrier_codec` round-trip and the REAL `inflate.py` decode path. A
  byte count without a parse-back that reproduces the decoded arrays is not a measurement.
- **n≥120 STRATIFIED RANDOM, never a prefix.** Pose is the axis where prefix bias is WORST-KNOWN:
  measured 2.54–4.21× HARDER on prefixes (m96 / `ddm_na2`), so a prefix NO-GO here is exactly the
  false-negative shape. Seeded, recorded seed, stratified.
- No `/tmp` in any persisted evidence. Bulk artifacts → `/Volumes/VertigoDataTier/pact/ddm_pk2_20260809/`.
- Commits via `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`,
  tags `[no-triality] [p0-ledger-ok]`, and **NO attribution trailer of any kind**.
- `.py` files need 2 × `tools/review_tracker.py mark-file <f> --status reviewed`; never
  `REVIEW_GATE_OVERRIDE=1` with a `.py`.
- Label every number `[macOS-CPU advisory]`, `score_claim=false`. Nothing here is a score.

## DELIVERABLE

One ranked table of measured rows (Δbytes, Δd_pose, Δd_seg, ΔS, arm, scope), the composed best, and
an honest verdict at the right scope level. If the section is genuinely irreducible at this d_pose,
that is a REAL and useful finding — say so with the measured response curve that proves it, and name
what would reopen it. State plainly which rungs you did not run.
