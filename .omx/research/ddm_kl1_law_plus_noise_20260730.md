---
schema: ddm_kl1_law_plus_noise.v1
date_utc: 2026-07-30
arm: ddm_kl1 (Kolmogorov law-coder; operator-directed)
lane_id: "lane_ddm_kl1_law_plus_noise_20260730"
research_only: true
score_claim: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU advisory; pure lossless coding + structure measurement over already-solved fields; no scorer/evaluator/dispatch/pointer mutation]"
operator_verbatim: "Name everything we still store as a list of numbers that is really one law plus noise. Code the law."
tools:
  - experiments/ddm_kl1_pose_field_coder.py       # race predictor FORM x coder, per-field
  - experiments/ddm_kl1_selector_and_tail_law.py  # B2 selector-from-xi + B3 rank-1 tail
  - experiments/ddm_kl1_pose_field_receiver.py     # the REAL lossless codec + rule-118 receiver
data: "SSD ddm_kl1_20260730/{b1_d2_pstar_coder.json, b1_ck1_pbest_coder.json, b1_qa43_ptwo_coder.json, b2b3_selector_tail.json, b1_receiver_verify.json}"
fires: "ledger QA52 (xi-trajectory coding); + new QA55 (container recompression), QA52-b defer"
tokens: "[no-triality] [p0-ledger-ok] [magnitude-ok]"
---

# ddm_kl1 — law-plus-noise enumeration + cheap law-coders (Kolmogorov pass)

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED. This arm does NOT move the
pointer.** Every number below is `[macOS-CPU advisory]`, `score_claim=false`, lossless-verified
over the already-solved fields of an ADVISORY vehicle (the pfs1/ck1 composed candidate,
S≈1.35–2.26, far from the pointer). No PoseNet/SegNet was run here — this is pure coding +
structure measurement. The value delivered is (a) the enumeration table, (b) a REAL verified
lossless pose-field codec worth ~0.7–0.9 KB, (c) the **falsification of three hypothesized
"laws"** (the operator's exact instruction: "Include streams where the verdict is honestly
NO-LAW").

## §0 Headline (answer first)

**The three law hypotheses in the seed list are MEASURED-FALSIFIED. The pose fields are
solver-output value-lists whose ONLY exploitable structure is a narrow alphabet — captured
near-optimally by generic byte-plane entropy coding, NOT by any temporal / low-rank "law."**

| seed law | hypothesis | MEASURED verdict |
|---|---|---|
| (1) pose field = ONE trajectory | se(3) B-spline/AR, 7.2KB→1-2KB | **FALSIFIED (temporal).** std(diff)/std(value)=1.14 (dim0) … 1.40 (dims1-5) ⇒ **temporally WHITE**. Poly/delta predictors LOSE to byte-plane. The real law is distributional (narrow alphabet). Lossless floor **5,948 B** (byte-plane), NOT 1-2KB (that was the LOSSY rank-1 → d_pose 0.207, already in the d2 ladder). |
| (2) selector bits from ξ yaw threshold | threshold(yaw)+exceptions | **FALSIFIED.** best |dim| corr with two-better = −0.23; a yaw threshold mispredicts **98/112**. The selector is just a sparse winner-set → **colex ~8 B (tail) / ~47 B (full-600)**; no yaw-law needed. |
| (3) tail correction = one shared scalar + noise | rank-1 shared axis | **FALSIFIED (scale artifact).** raw SVD "rank-1 = 1.0" is dim0 (speed~73) swamping dims1-5 (~1.8); **standardized SVD = [0.71, 0.15, 0.10, 0.03]**. Lossless rank-1+residual = **1,519 B LOSES to byte-plane 1,212 B**. |

**Why (one sentence):** these are per-pair INDEPENDENT GN optimizations landing in local optima
under f16 quantization — they have a low-entropy alphabet but no temporal/low-rank structure a
predictor can exploit beyond trivial dim0-scale dominance; generic byte-plane brotli reaches the
alphabet floor.

## §1 THE LAW-PLUS-NOISE TABLE (LEG A) — every stored list of numbers, named

MEASURED on the live ck1 STANDIN composed archive (`ck1_composed_STANDIN_fullbase_pstar_archive.zip`,
273,659 B, **all members stored method=0 / uncompressed**) + the queued v4b/v4c streams.

| stream | bytes now | the law (measured evidence) | law-coded bytes | consumer | owner |
|---|---:|---|---:|---|---|
| **pose_warp.stp** (600×6 f16 field) | 6,844 | narrow alphabet → byte-plane; **NOT temporal, NOT rank-1** | **5,964** (codec member) | v4c pose | **ddm_kl1** |
| two-plane tail (112×6 f16) | 1,344 | narrow alphabet → byte-plane; rank-1 LOSES | 1,261 | v4b pose tail | **ddm_kl1** |
| two-plane selector (600 bit) | ~75 | sparse winner-set → colex; **yaw-law FALSIFIED** | ~47 | v4b selector | **ddm_kl1** |
| tokens.dr7t | 261,590 | **ALREADY at entropy** (brotli 261,595 = 100.0%; per-cell MODE base owns temporal law, carried-ξ INTER LOST = QA39 FIRED) | — (no slack) | renderer tokens | gr1 / #774 (hands off) |
| renderer.sec | 3,341 | **ALREADY coded** (brotli 100.1%) | — | renderer weights | (coded) |
| selector.sec (renderer CONFIG json) | 535 | JSON text, brotli-able | 277 | renderer config | container (MAIN) |
| manifest.json | 1,266 | JSON text, brotli-able | 609 | archive manifest | container (MAIN) |
| pose_stub.sec | 83 | tiny | 76 | pose stub | (trivial) |
| dash phase | — | arc-length periodic + ∫v dt | DESIGN (not separable) | lane stream | gate: needs lane-primitive grammar (rg-family) |
| QA44-B exposure curve | — | a·warp+b per-pair (~4 B) | pm1 FIRED (family-level) | pose photometric | pm1 (consume) |
| QA46 restore cells | — | sky-region law not cell list | HELD (secondary) | seg restore | qa46 |

**Negative rows are signal (per charter):** tokens/renderer are ALREADY law-coded (zero
recompression slack) — the rate mass (tokens = **95.6% of the archive**) is closed to generic
coding; any further token win must be structural (gr1's token-granularity domain), not a
list→law recompression this arm can do.

## §2 LEG B1 — pose-field codec (REAL, verified, LOSSLESS)

`experiments/ddm_kl1_pose_field_receiver.py` implements the codec + the **rule-118-free receiver
decoder** and PROVES bit-exact f16 round-trip on all three live fields (zero d_pose risk by
construction — the receiver reconstructs the identical f16 params):

| field | raw | codec member (byte-plane colmajor brotli + 16 B hdr) | lossless | vs shipped |
|---|---:|---:|:--:|---:|
| D2 p_star (P0, 600×6) | 7,200 | **5,964** | ✓ bit-exact | .stp 6,844 → **−880 B** (member) / **−687 B** (payload-to-payload vs the .stp's internal 6,635-B brotli sub-stream) |
| ck1 p_best_kneeA (600×6) | 7,200 | 6,022 | ✓ | — |
| two-plane p_two_star (112×6) | 1,344 | 1,261 | ✓ | (lzma1 residual is 1,212, 49 B better on this tail) |

**The law raced and LOST (constants-are-poison discipline — measured, not assumed):** per-column
poly{1,2,3,5}/delta{1,2} predictors + entropy-coded residual land **6,180–6,268 B**; the byte-plane
distributional code is **5,948**. The `.stp` internal codec is row-major brotli (its header length
field 0x19eb = 6,635 = exactly my `row_major_brotli` control) — byte-plane column-major is the
strictly-smaller lossless encoding of the identical field.

**#404 magnitude:** −880 B = **0.00059 S** (25·880/37,545,489). On the 2.08-wide composed advisory
gap that is **0.028%** — real, zero-risk, but small. Honest: this does not move the pointer.

## §3 LEG B2 — selector-from-ξ (yaw-law FALSIFIED; ship colex)

`experiments/ddm_kl1_selector_and_tail_law.py`. Best-of selector over the tail = 95/112 pick
two-plane. Correlation of |any dim| with "two-better" ≤ 0.23; the best yaw threshold mispredicts
**98/112**. Realized coding: raw explicit 14 B, brotli-bitmap 18 B, **colex-of-winner-set 8.2 B
(tail) / 46.7 B (full-600)**. **Verdict: no yaw-law; the selector is a small sparse set — colex-code
it (~47 B on the full 600).** Realized-selection check: because we list the winner set exactly
(colex is lossless), the derived selector = the stored optimal selector, so **zero d_pose price**
(no pair flipped to the worse branch).

## §4 LEG B3 — rank-1 tail (scale artifact; byte-plane wins)

Raw SVD energy of the 112×6 tail field = [1.0, 0, …] — but this is dim0 (speed ~73) swamping
dims1-5 (~1.8). **Standardized SVD = [0.714, 0.153, 0.102, 0.031, 0, 0]** ⇒ genuinely ~rank-4, and
the small dims (rotation/translation-y) are exactly what the pose correction NEEDS (matches the d2
ladder: rank-1 → d_pose 0.207 WORSE than 6dof 0.1595). Lossless rank-1 + residual = **1,519 B, LOSES
to byte-plane 1,212 B.** No shared-scalar law; ship byte-plane.

## §5 LEG B4 — dash phase (DESIGN row, gated)

Dash-phase = arc-length-periodic + ∫v dt advance is a genuine law, but **NOT separable in the v4a/v4b
grammar**: the lane/dash content lives inside `tokens.dr7t` (261 KB, already entropy-coded, gr1
domain), not as its own stream. **Gate:** phase-law coding applies only after a lane-primitive
grammar extension (rg-family) exposes dash phase as its own stream. Emitting as a DESIGN row per
charter; not forced.

## §6 QA52 FIRED + the dynamics-regularized half (QA52-b, DEFER)

**QA52 (ξ-trajectory coding) → FIRED.** The RATE half is measured-falsified at $0: the field is
temporally white, so a smoothness/B-spline/AR prior cannot compress it (poly/spline LOSE to
byte-plane) — the ~1-2KB target does not exist losslessly; the lossless floor is 5,948 B (byte-plane),
a −687…−880 B win over the shipped `.stp`. The se(3) B-spline (designed in `tac.lie`, never fired on
this vehicle) is the **wrong chart for this field's RATE**.

**QA52-b (dynamics-regularized GN re-solve) — DEFER (PoseNet-gated).** The remaining QA52 "prize"
(neighbor-transfer to hard tail pairs) is a d_pose-improvement mechanism, NOT a rate mechanism, and
the RATE prize is already falsified ($0). Whether a bicycle-model-smoothness-constrained re-solve can
*improve tail d_pose* (info transfer from easy neighbors — 24% of tail in runs ≥4) is UNTESTED and
requires bounded PoseNet re-solves through the real receiver. **Falsifier:** the whiteness measurement
predicts NEGATIVE (neighbors don't predict each other), so a smoothness prior likely pulls tail poses
off their per-pair optima → d_pose degrades. Priced as bounded PoseNet work; deferred to MAIN/pi2 (who
own the PoseNet Jacobian surface) rather than spend it here to confirm a strongly-predicted negative.

## §7 v4c grammar recommendation (what to ship)

1. **Pose field → byte-plane codec member** (`encode_pose_field`/`decode_pose_field`, rule-118-free
   receiver): 6,844 → 5,964, **−880 B lossless, zero d_pose risk**. Applies to the v4b mixed field too.
2. **Selector → colex of the sparse winner-set** (~47 B full-600), not an explicit bitmap.
3. **Two-plane tail → byte-plane** (1,344 → 1,261; lzma1 residual 1,212 if the container prefers).
4. **Container (MAIN decision):** the STANDIN stores all members method=0. Compressing the
   config members (`selector.sec` 535→277, `manifest.json` 1,266→609, `pose_stub` 83→76) is
   another ~−915 B lossless (IF the real builder isn't already deflating them; the pose-member win
   is intrinsic to the `.stp` reformat regardless). Total lossless container slack ≈ **1.8 KB ≈
   0.0012 S**, zero seg/pose risk.
5. **Do NOT** ship a B-spline/AR/rank-1 pose code (all measured-worse and/or lossy).

## §8 Confounds + discipline

- **`tac` import HIJACK noted, not load-bearing here:** the shared venv editable-install points to
  the eg1 codex worktree; this arm ran `PYTHONPATH=$PWD/src` but its results are PURE coding
  (numpy + brotli + lzma over JSONL field values) — no tac/receiver/scorer path touched, so the
  hijack cannot affect any number. (The receiver-decoder round-trip is proven by re-decoding the
  encoded member, self-contained.)
- **Lossless is PROVEN, not asserted:** every codec path asserts bit-exact reconstruction of the
  f16 field (NO-FAKE gate) before reporting a size.
- **Verdict scope:** the three law-falsifications are FAMILY-level for THESE solver-output fields
  (temporal/rank-1 structure absent by measurement); the exact byte totals are INSTANCE (this
  vehicle, these solves). The container-recompression totals are INSTANCE to the STANDIN build.
- **Advisory everywhere:** frozen advisory vehicle, macOS-CPU, non-promotable, `score_claim=false`,
  pointer 0.1910828242 UNMOVED.
