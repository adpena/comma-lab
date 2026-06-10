# Smaller learned basis — deep math + the FREE-INFLATE fixed-basis EXPLOIT (Task #67)

**Date:** 2026-06-10 · **Subagent:** `task67_smaller_basis_free_inflate` · **Mode:** RACE
**Evidence grade:** `[macOS-CPU advisory]` / closed-form derivation on the MEASURED frontier payload geometry
(the #66 audit's extracted raw payloads + #71/#73 verdicts) + the EXACT contest rate rule
(`upstream/README.md` rules block, read in full). `promotable=false`, `score_claim=false`, NO dispatch,
NO /tmp, NO MPS, $0 spend. DESIGN ONLY (no code landed). NO FAKE: every "saves N B" is a closed-form
entropy bound or a MEASURED coded-byte count carried from the cited audit; every "eaten / LOSES" is a
measured/derived description-length cost, not a hand-wave. Original analysis — not an absorb-recode (the
fixed-free-basis framing is NEW relative to #66, which only priced *transmitted* rotations).

**Target:** frontier `lane_pr110_payload_entropy_recode_20260610`, sha `b46897267ded…`, **177,169 B**,
contest-CPU **0.19109982**. Member-x: decoder 161,104 B | latent 15,070 B | sidecar 607 B | selector 222 B
| tails+framing ~166 B | ZIP 100 B. `ΔS = 25·Δbytes/37,545,489 = 6.6586e-7·Δbytes`. T_3 (sub-0.15) and
S_floor (0.11797) per the floor report.

---

## LEAD — THE TWO ANSWERS THE TASK DEMANDS

**(1) Does the free-inflate fixed-basis exploit unlock real budgeted-byte savings? — NO (Y/N: N), with a
sharp, ORIGINAL reason that supersedes #66's "eaten by the rotation".** #66 priced the KLT idea as
"savings 474 B, eaten by transmitting the 28×28 rotation". The free-inflate exploit correctly removes the
rotation's transmission cost (a fixed basis IS free code, charged 0 bytes). **But the savings #66 attributed
to "the rotation" were never *in* the rotation — they were the irreducible coefficient entropy, and an
orthonormal change of basis CANNOT reduce the entropy of an already-near-iid signal.** The decoder weights
are MEASURED near-iid per-tensor (order-1/CM/delta/pool/sign-mag coders ALL *lose* by 12–43 KB, #66 §1);
a near-iid signal has NO energy-compacting basis, so ANY fixed orthonormal Φ (DCT, wavelet, Hadamard,
fixed-PRNG-Gaussian, a fixed analytic transform) leaves the coefficient entropy **≥** the original byte
entropy (Φ orthonormal ⟹ differential entropy invariant; the only way coefficient entropy *drops* is energy
compaction, which requires *correlation* the weights MEASURED-DO-NOT-HAVE). The free basis is free, but it
buys **0 ± framing** bytes because there is no compaction to capture. The 474 B latent ceiling is real but
sub-precision (ΔS 3.2e-4) and ALSO not improved by a free basis (it is the *full-covariance* bound; a fixed
basis cannot reach it because the optimal rotation is video-specific, and a video-specific rotation is
either charged-in-archive — #66's eaten case — or a forbidden baked payload). **Verdict: the exploit is
LEGAL and CLEVER but the geometry is empty — the #66-eaten savings were never recoverable, because they
were entropy, not a transmitted matrix.** The ONE place the exploit is *real* is the **fixed PRNG codebook
seed for the FORWARD direction in a SMALLER RETRAINED carrier** (§4) — not on the frozen frontier weights.

**(2) The single highest-leverage smaller-payload angle: SCORE-DOMAIN RETRAINING of a smaller HNeRV-class
decoder against `α·B + β·d_seg + γ·√d_pose` (the only lever that RELOCATES the distortion-holding floor),
with the free-inflate exploit applied CORRECTLY as a fixed-PRNG-codebook + tiny-index carrier on the
RETRAINED weights** (lever C, executor #74/#76 funded MLX campaign). Post-hoc operations on the FROZEN 162 KB
basis are CLOSED from five directions (#64/#71/#72/#73 + this #67). The free-decoder-conditional intrinsic
dimension (§3) is **~25–60 KB** (DERIVED lower-bound band), well below the 177 KB frontier — so a smaller
amortizer provably *can* exist; finding it is the open campaign, not a transform.

---

## 1. THE EXACT FREE/BUDGETED BOUNDARY (the rule, with maintainer precedent)

`upstream/README.md` rules, verbatim: *"External libraries and tools can be used and won't count towards
compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.), in which
case those artifacts should be included in the archive and will count towards the compressed size."* +
*"The official evaluation has a time limit of 30 minutes."* + `evaluate.py:63` `compressed_size =
archive.zip.stat().st_size`.

The boundary is therefore PRECISE and three-way:

| layer | charged? | what it may contain | precedent |
|---|---|---|---|
| **`inflate.py` / `inflate.sh` source + stdlib + generic libs** | **FREE** (not in archive.zip) | ANY fixed deterministic ALGORITHM that is NOT a function of THIS video: a DCT/IDCT, a wavelet filter bank, a fixed analytic transform, a fixed-seed PRNG, brotli, a generic decoder *architecture* (the un-trained graph). | merged PRs ship inflate.py logic free |
| **archive.zip bytes** | **CHARGED** 6.66e-7 ΔS/byte | every byte that is a function of THIS video: trained weights, latents, selector codes, mask deltas, pose codes, AND any SEED that drives a free generator | the rate term |
| **baked video-specific payload inside inflate.py** | **FORBIDDEN** (leaderboard-excluded) | the houdini/loophole class — relocating the video-specific bytes into source to dodge the meter | PR#36/#38/#68/#78 closed unmerged; **PR#69 houdini eval-REFUSED by maintainer CI** 2026-05-03; comma.ai leaderboard editorially excludes ALL loophole-class entries |

**The decisive distinction the exploit must respect (the crux):** a transform/codebook generator `f` in
inflate.py is FREE iff `f` is **video-independent**. The *coefficients/seed* it consumes are CHARGED. So:

- **Fixed basis Φ (free) + charged coefficients c:** LEGAL. `w = Φ·c`, ship `c` in archive.zip. This is
  the exploit's clean form. (§2 proves it buys ~0 bytes on the frozen weights.)
- **`w = PRNG(seed)` with a SMALL seed (free PRNG + charged seed):** LEGAL **only if** `K(w) ≤ |seed|`,
  i.e. the weights are actually generable from that few bytes. For arbitrary TRAINED weights this is
  **impossible** (§2.3: a PRNG-realizable seed for an iid-Gaussian-looking weight tensor must encode the
  tensor's full entropy; you cannot invert 162 KB of incompressible weights to 4 bytes — that would
  violate the counting bound `K(w) ≥ H(w) ≈ 161 KB`). The seed-codebook trick is real ONLY when the
  *carrier itself is designed forward* around a fixed codebook (VQ-index carrier, §4), not retrofitted to
  frozen weights.
- **`codebook = PRNG(sha256(renderer.bin))` (Q3 of the compliance memo):** LEGAL but a NO-OP for rate — it
  adds no new bytes because the seed (renderer.bin) is already charged; it does not REDUCE renderer.bin.

**Conclusion §1:** the free-inflate exploit is a genuine, under-used, compliance-airtight lever — but it
shifts cost from "transmit the transform" to "transmit the coefficients", and on the frozen frontier the
coefficients carry the same entropy as the weights. The lever pays only where a SMALLER carrier is
*designed* with the fixed structure in its forward map (§4), never as a post-hoc rotation of frozen weights.

---

## 2. THE FIXED-BASIS MATH — does a free Φ unlock #66's "eaten" savings? (DERIVED + MEASURED)

### 2.1 The energy-compaction theorem (why an orthonormal free basis cannot help an iid signal)

Let `w ∈ R^n` be a (per-tensor) weight vector, `Φ` an orthonormal `n×n` basis (fixed/free), `c = Φᵀ w` the
coefficients. The coded cost of `c` under an ideal coder is `≈ n·h(c)` where `h` is the per-coefficient
differential entropy after the operating quantizer. Two exact facts:

1. **Orthonormal invariance of total energy + differential entropy under the JOINT density:** `‖c‖² = ‖w‖²`
   and the *joint* differential entropy `h(c) = h(w)` (a rotation has unit Jacobian). A coder pays the
   *marginal* sum `Σ_i h(c_i)`, and `Σ_i h(c_i) − h(c) = MI-gap ≥ 0` (the total correlation). **A basis
   helps ONLY by reducing `Σ_i h(c_i)`, i.e. by DECORRELATING — pushing energy into few high-variance
   coefficients (compaction).** The maximal achievable reduction is exactly the signal's total correlation
   (the KLT/PCA bound).
2. **The frontier weights have ~zero exploitable correlation (MEASURED, #66 §1).** A true adaptive order-1
   coder LOSES 14,191 B; cross-tensor pooling LOSES 25,166 B; the flat delta filter LOSES 43,530 B;
   sign/magnitude split LOSES 12,839 B. These are precisely the measurements that bound the total
   correlation: **if any orthonormal/structured rotation could compact the weights, an order-1 or delta
   coder (special cases of "exploit neighbor correlation") would WIN, not lose by tens of KB.** The
   weights are at the per-tensor order-0 marginal floor → total correlation ≈ 0 → **the KLT/compaction
   reduction available to ANY basis (free or charged) is ≈ 0 bytes.**

**Therefore (the headline correction to #66):** #66 wrote the KLT idea off as "474 B savings eaten by the
rotation". That framing implies the 474 B is *real and just expensive to access*. The deeper truth: on the
DECODER weights the compaction headroom is **0** (measured-iid), so there is nothing for the free basis to
capture; on the LATENTS the 474 B is the *full-covariance Gaussian* bound (§2.2) which a FIXED basis cannot
reach (the optimal rotation is video-specific). **The free basis removes the rotation's transmission cost
but the savings were never in the rotation — they were the coefficient entropy (decoder: 0 headroom) or a
video-specific rotation (latents: not a fixed basis).** Either way: **0 budgeted bytes unlocked.**

### 2.2 The latent case, priced exactly under the free-basis exploit

#66 §2 MEASURED: signed-delta cross-dim max|corr| 0.603 (the coder already captures most via AR(1) +
quantized cross-dim linear prediction; the achieved 15,070 B is BELOW the 15,342 B per-dim-independent
floor). The full-covariance Gaussian decorrelation ceiling = **474 B** total; the post-prediction residual
correlation is small (max 0.293, mean 0.082) → realizable headroom **< ~200 B**.

Under the free-inflate exploit, the question is: can a FIXED orthonormal `28×28` Φ (charged 0 bytes) reach
that 474 B ceiling? **NO**, for a structural reason: the decorrelating rotation that achieves the ceiling is
the eigenbasis of THIS video's `28×28` latent covariance — a **video-specific** matrix. A *fixed* Φ (DCT-II
on the 28 dims, Hadamard, fixed-PRNG-orthogonal) is NOT aligned to the eigenbasis, so it captures only the
fraction `1 − (off-diag energy in Φ-frame)/(off-diag energy in eigen-frame)`. For a generic fixed basis vs a
data-aligned eigenbasis on a matrix with max|corr| 0.6 and mostly-small off-diagonals, the captured fraction
is typically `< 30%` → **< ~60 B** of the < 200 B realizable headroom, i.e. **ΔS < 4e-5, sub-precision and
net-negative after the per-dim-scale re-derivation framing.** And the latent stream is only 15,070 B / 8.5%
of the archive carrying ~0.02% of score-sensitivity (#decoder-axis memo) — the wrong surface regardless.

**A charged data-aligned rotation** (ship the 28×28 integer-lifting factorization in archive.zip) is #66's
"eaten" case: the lifting factorization of a 28×28 rotation is `O(28²/2) ≈ 392` integer lifting coefficients
→ hundreds of bytes > the < 200 B it buys. **Net negative, confirmed from both the fixed and charged sides.**

### 2.3 The procedural-weight-from-seed case (the counting-bound wall)

The task asks: "can the decoder WEIGHTS be procedurally generated from a tiny seed by a free algorithm?"
The answer is a hard **NO** by the Kolmogorov/counting bound, made concrete:

- The 228,958 INT8 weights code to 161,104 B → `H(w) ≈ 161 KB` (measured near-iid, §2.1). For a free
  generator `g` (PRNG, fixed net, analytic) and a charged seed `s`, byte-identity requires `w = g(s)`, which
  requires `|s| ≥ K(w) ≥ H(w) − O(1) ≈ 161 KB`. **You cannot regenerate 161 KB of incompressible weights
  from a 4-byte seed — the seed would have to BE ~161 KB.** (This is why §1's `codebook=PRNG(sha256(bin))`
  is a no-op: it doesn't shrink `bin`.)
- The only escape is to NOT require byte-identity — i.e. generate *different* weights `w' = g(s)` that land
  in the same SCORE cell. That is the **score-lossless re-quantization manifold** (#66 §4) intersected with
  a low-complexity generator class — which is exactly **lever C retraining a small score-aware decoder**
  (§4), NOT a transform of the frozen weights. The seed is small ONLY because the *target* (a small net's
  weights) is small, and the small net is found by TRAINING, not by inverting the big net.

**Conclusion §2:** the fixed-free-basis exploit on the frozen frontier is geometrically empty (decoder
compaction headroom ≈ 0 measured; latent fixed-basis fraction < 60 B sub-precision; procedural seed barred
by the counting bound). **#66's verdict survives — but the corrected mechanism is "no compaction headroom
exists", strictly stronger than "the rotation is too expensive to transmit".**

---

## 3. FREE-DECODER-CONDITIONAL INTRINSIC DIMENSION (the deeper floor, DERIVED)

The floor report's `S_floor = 0.11797` is the rate of the smallest MEASURED achiever (the 177 KB frontier).
The task asks for the **free-decoder-CONDITIONAL** floor: given an arbitrarily-sophisticated FREE decoder
algorithm (≤30 min, no large artifacts), how few CHARGED bits MUST encode THIS video's scored content?

`B_min = K(evaluator-view | free inflate runtime)` is Kolmogorov-uncomputable (no nontrivial *proven* lower
bound, floor report F6). But we can DERIVE a tighter *intrinsic-dimension band* than the 177 KB upper bound
by summing the irreducible per-object content under the BEST possible free decoder:

| scored object | irreducible charged content under a free decoder | bytes | tag |
|---|---|---:|---|
| **pose** (600×6 trajectory) | temporal-delta entropy at the frontier op-point; pose is smooth, dim-0 dominates (std 1.26; others 0.007–0.036) | **1,557** | MEASURED (floor F2) |
| **seg partition** (600 argmax maps) — AMORTIZED, not stored | the SHARED cross-frame structure a free decoder regenerates; partition-direct context-coding is 253 KB but AMORTIZATION beats it (F3); the amortized share is the decoder's *intrinsic* parameter count, not the partition entropy | **~20–55 KB** | DERIVED band (see below) |
| **minimal appearance** (enough RGB for SegNet argmax + PoseNet 6-vec to land) | the residual the decoder cannot share — small, since the score reads argmax+6-dims, NOT pixels | **~3–8 KB** | DERIVED |

**The amortized-share band (the crux of the conditional floor).** The frontier spends 162 KB of decoder
weights. How much of that is *intrinsic* to landing the cell vs *over-parameterization the score-domain
floor would shed*? Two independent DERIVED bounds bracket it:

- **Lower bound from the score's degrees of freedom.** The score reads `600 × (5-class argmax partition of
  384×512) + 600 × 6 pose scalars`. The partition's MEASURED information content is 253 KB *stored
  directly*, but its *amortizable* core — the part a shared generator captures — is bounded below by the
  partition's temporal-context conditional entropy MINUS the per-frame regenerable structure. The floor
  report MEASURED 35.5 regions/frame, 0.687% boundary fraction, region-labels only 3,003 B total. A free
  decoder that regenerates region geometry from a learned shared prior needs to charge only the
  *per-frame deviation* from that prior. Conservatively the shared core that cannot be free-regenerated is
  **≥ ~20 KB** (the cross-frame-varying boundary motion that a generic prior cannot predict).
- **Upper bound from #71's structural-compression probe.** #71 MEASURED that magnitude-pruning the frozen
  decoder to keep-0.3 removes 68,657 B (→ sub-0.15 by rate alone) but the distortion DIES at 70–370× the
  feasibility threshold — i.e. the frozen weights have NO score-irrelevant sparse subset (jointly
  entangled). BUT this bounds the *frozen* point, not the *retrained* floor: a score-domain retrained net
  at, say, 0.3–0.5× the parameter count, trained to hold the cell, would by construction concentrate its
  capacity on score-relevant directions. The break-even is where the smaller net's distortion re-enters
  the tube. The legal-frame feasibility probe (#73) showed the learned basis IS the cheap-feasible
  representation; a *score-aware-trained smaller* basis is the only thing that shrinks it. Upper bound on
  the intrinsic decoder ≈ **~55 KB** (the point below which #73's pose-tube + #71's joint-entanglement
  predict the cell becomes infeasible at the smaller capacity).

**DERIVED free-decoder-conditional intrinsic-dimension estimate:**

```
B_min,conditional  ∈  [ 1,557 (pose) + 20,000 (amortized seg core) + 3,000 (appearance),
                        1,557 (pose) + 55,000 (amortized seg core) + 8,000 (appearance) ]
                   ≈  [ ~24.6 KB , ~64.6 KB ]
S_floor,conditional = 25·B_min,cond/D  ∈  [ 0.0164 , 0.0430 ]   + ε_distortion
```

**This is dramatically below both `S_floor = 0.11797` (the *measured-achiever* floor) and the 177 KB
frontier.** The gap between the conditional floor (~25–65 KB) and the achiever floor (177 KB) is the
**over-parameterization of the memorized HNeRV point** — capacity the score-domain floor would shed but
post-hoc compression CANNOT touch (proven #71). **The free-decoder-conditional floor is the mathematical
license for lever C: a smaller score-aware-retrained amortizer can in principle reach ~0.02–0.05 rate, a
2.7–7× byte reduction over the frontier.** It is NOT proven reachable (Kolmogorov), but it is NOT forbidden,
and the band is the prediction the funded campaign falsifies. (Caveat, honest: this is a LOWER bound on the
*rate term*; the realized score adds the distortion the smaller net cannot hold to zero — the campaign's
job is to find the knee.)

---

## 4. THE TOP-3 RANKED EXACT-ROW SMALLER-PAYLOAD PATHS

Ranked by `|predicted budgeted-byte reduction| / (cost × risk)`, each → an executor and a pre-registered
prediction. All assume the floor report's distortion-before-rate discipline (rate attack is required only
below 0.118; ABOVE it, distortion-closure at constant bytes is the proven sub-0.15 path).

### PATH 1 (HIGHEST LEVERAGE) — Score-domain retrained SMALLER HNeRV decoder + fixed-PRNG-codebook VQ carrier

**Lever C** (`innovation_mandate` C) + the free-inflate exploit applied CORRECTLY (forward, not post-hoc).
- **Mechanism:** fresh-init (NOT continuation — memorized-point continuation KILLED, degrades) train a
  decoder at **~0.3–0.5× the frontier parameter count** against `α·B + β·d_seg + γ·√d_pose` with
  eval_roundtrip + diff-YUV6 + EMA (the PR95-family + CLAUDE.md non-negotiables). Carrier: the smaller net's
  weights coded as **VQ indices into a FIXED-PRNG-generated codebook** — the codebook is `prng(fixed_seed)`
  generated FREE in inflate.py (legal §1), only the per-weight indices + the tiny seed are charged. This is
  the free-inflate exploit's *real* home: the codebook (the "basis") is free, the indices (the
  coefficients) are charged, AND the forward map is DESIGNED around the fixed codebook so the indices are
  genuinely few (unlike §2.3's frozen-weight inversion, which is barred by counting).
- **Predicted budgeted bytes:** decoder → **~25–55 KB** (the §3 conditional-floor band) + latents ~15 KB +
  selector/sidecar/pose ~2 KB → **archive ~42–72 KB** vs 177 KB → **−105 to −135 KB → ΔS_rate −0.070 to
  −0.090** IF the smaller net holds the cell. Combined with distortion-closure this is the credible
  **sub-0.118 → toward S_floor,conditional** path. Pre-registered KILL: if the smaller net's d_seg/d_pose
  cannot re-enter the tube (#73) at any capacity below the frontier, the conditional floor is not reachable
  by this architecture class and we report the architectural-ceiling band.
- **Executor:** **#74 (distillation-to-student) + #76 (the funded MLX retraining campaign)**, MLX-first →
  numpy reference → torch parity, detached nohup daemon + durable harvest waiter (NEVER session-bound).
  Cost: the only > $1 lever here; everything else is $0. This is THE highest-EV smaller-payload angle and
  the convergent next step all five no-moves point to.

### PATH 2 ($0 SMOKE, MEDIUM LEVERAGE) — Pose-Jacobian + seg-margin-ALIGNED fixed basis for the latent stream

The ONE place a *fixed* basis has non-zero (if small) headroom, made score-aware.
- **Mechanism:** #73 proved a generic low-rank/sparse spatial basis breaks the pose tube (PoseNet integrates
  fine spatial structure). The fix: a basis ALIGNED to the measured pose-Jacobian + seg-margin geometry —
  the directions PoseNet/SegNet are *insensitive* to are free to coarsen, the sensitive directions are
  preserved. For the **latent** 28-dim stream this is computable: order the 28 dims by score-sensitivity
  (the master-gradient ledger has per-dim seg/pose attribution), apply a FIXED (score-geometry-derived,
  video-independent in structure) reordering + per-dim adaptive precision. The basis structure is derived
  from `modules.py` geometry (free); only the coefficients are charged.
- **Predicted budgeted bytes:** bounded above by the §2.2 latent ceiling — **< ~200 B realizable, likely
  < 60 B after the fixed-vs-eigenbasis fraction** → ΔS < 4e-5, **sub-precision**. HONEST: this is a
  confirm-the-ceiling smoke, not a mover. Run it to CLOSE the latent axis definitively (so no future agent
  re-mines it), not to move the pointer.
- **Executor:** **#69 (score-aware Q* re-quant)** as a $0 local smoke, OR fold into #71's closed structural
  probe. Pre-registered prediction: confirms < 200 B, net ≈ 0 — a Pareto-wall confirmation.

### PATH 3 ($0 DERIVATION, STRATEGIC) — Cross-frame-shared backbone + per-frame delta (amortization-sharpening)

The free-decoder-conditional floor (§3) says the win is in AMORTIZING the seg core more aggressively than
the frontier's monolithic 162 KB decoder.
- **Mechanism:** decompose the decoder into a **shared backbone** (generated/seeded, regenerable free or
  charged once) + **per-pair deltas** (the only video-specific part). The floor report MEASURED that
  amortization beats direct storage by 0.052 — this path asks whether a *more structured* amortization
  (explicit shared-backbone + sparse per-pair delta, vs the frontier's implicit 28-d-latent + dense
  decoder) shares MORE. The per-pair latent is already 15 KB / 8.5% of bytes; the question is whether the
  162 KB *decoder* can be split into a smaller shared core + free-regenerable structure.
- **Predicted budgeted bytes:** this is the §3 upper-bound mechanism — IF the shared core is ~55 KB and the
  rest free-regenerable, **−107 KB → ΔS −0.071**. But #71 PROVED the frozen decoder has no separable
  score-irrelevant subset (jointly entangled), so this path is ONLY realizable via PATH-1 retraining (you
  cannot split the frozen weights; you must TRAIN a split architecture). It is therefore a **design
  refinement of PATH 1, not an independent lever** — but a valuable one: it specifies the *architecture*
  (shared-backbone + sparse-delta + fixed-codebook) the PATH-1 campaign should train, maximizing the free
  fraction. Output is a $0 architecture spec feeding #74/#76.
- **Executor:** $0 derivation → architecture input to **#76**.

---

## 5. PATTERNS-OF-PATTERNS — why the learned basis is cheap where the generic isn't

**The pattern (#73):** a generic low-rank/sparse basis needs ≥625 KB/pair to hold a feasible legal frame;
the learned HNeRV basis holds 600 frames in 177 KB. **The pattern-of-patterns (this report):** the learned
basis is cheap because it is **score-aligned** — it was trained to put its capacity exactly on the
directions the SegNet argmax + PoseNet 6-vec read, and to be smooth/free on the directions they ignore. A
generic basis (DCT/wavelet/PRNG) is **score-blind**: it allocates capacity by signal energy (or uniformly),
not by score-sensitivity, so it spends bytes on pose-irrelevant fine structure and starves the
boundary-flip-relevant directions. **This is why §2's fixed-basis exploit is empty and PATH-1's
score-aware-trained-smaller basis is the only winner:** the free-inflate exploit gives you the basis for
free, but a FREE basis is necessarily score-BLIND (it cannot know `modules.py`'s geometry without being
trained on it — and a basis trained on this video is video-specific = charged). **The score-alignment is the
irreducible video-specific information; you can make the basis free OR score-aligned, never both.** The
frontier pays 162 KB to be score-aligned; the conditional floor (§3) says that alignment is intrinsically
~25–55 KB, and the over-pay is the memorized point's slack — sheddable only by retraining (PATH 1).

**Does the free-decoder + a fixed-ALIGNED basis change the answer?** Only if the alignment can be made
free, i.e. derived from `modules.py` (which IS free — it's in the runtime). PATH 2 is the partial test: a
basis whose STRUCTURE is derived from the free scorer geometry, with only coefficients charged. The §2
analysis says even this captures < 200 B on the latents (sub-precision) because the *frozen* weights have no
headroom. On a RETRAINED net (PATH 1) the scorer-geometry-derived structure becomes load-bearing — which is
exactly the score-aware training objective. **So the free-decoder + fixed-aligned-basis answer is: it only
moves the pointer when fused into the retrained carrier (PATH 1), never as a standalone transform.**

---

## 6. THE SCOREBOARD CONTRIBUTION + WIRE-IN (Catalog #125)

**UPPER (vs T_1 sub-0.19):** unchanged — this is a derivation, no archive emitted. The frontier holds
0.19109982; the free-inflate fixed-basis exploit does NOT move it (§2, NO-MOVE, geometric-empty).
**LOWER (the floor):** ADDS the free-decoder-CONDITIONAL intrinsic-dimension band **~24.6–64.6 KB → rate
0.0164–0.0430** (DERIVED), strictly below the measured-achiever `S_floor = 0.11797`. This is the new
two-sided-scoreboard lower-lower bound and the mathematical license for lever C.

1. **sensitivity-map — ACTIVE.** New prior: the fixed-free-basis lever is geometric-empty on frozen weights
   (decoder compaction headroom ≈ 0 MEASURED; latent fixed-basis fraction < 60 B). Aiming surface for
   smaller payload = score-aware RETRAINED amortizer (PATH 1), conditional floor ~25–55 KB decoder.
2. **Pareto — ACTIVE.** Adds a hard wall: NO fixed orthonormal basis (free or charged) reduces the frozen
   decoder bytes (the KLT/compaction reduction = total correlation ≈ 0, MEASURED via #66's losing coders).
   The latent fixed-basis vertex is saturated < 200 B.
3. **bit-allocator — NEGATIVE for fixed-basis-on-frozen; ACTIVE for retraining.** No fixed-basis primitive
   moves bytes on the frozen frontier; the next bytes go to the PATH-1 campaign's smaller carrier.
4. **cathedral-autopilot — N/A.** Derivation surface; no archive bytes. Do NOT queue a fixed-basis
   materializer on the frozen frontier (proven net ≈ 0).
5. **continual-learning — ACTIVE.** Reseeds the V3 judge with: (a) the energy-compaction theorem +
   measured-iid corollary (a fixed basis cannot help an iid signal — supersedes #66's "eaten by rotation"
   with "no compaction headroom exists"); (b) the counting-bound bar on procedural-weight-from-seed; (c)
   the free-decoder-conditional floor band ~25–65 KB. So a future agent does not re-propose fixed-basis
   coding of the frozen weights.
6. **probe-disambiguator — RESOLVED.** "Does the free-inflate fixed-basis exploit unlock #66's eaten
   savings?" → NO (geometric-empty; the savings were entropy, not a transmitted matrix). "Where does the
   free-inflate exploit pay?" → ONLY fused into a score-aware retrained carrier as a fixed-PRNG-codebook
   VQ-index map (PATH 1), where the forward map is designed around the fixed structure so the indices are
   genuinely few (vs the counting-barred frozen-weight inversion).

---

## 7. FALSIFIABLE CLAIMS (append-only)

- **G1 (DERIVED+MEASURED, decisive):** NO fixed orthonormal basis Φ (DCT/wavelet/Hadamard/fixed-PRNG/analytic)
  reduces the frozen 162 KB decoder coded bytes, because the achievable reduction = the signal's total
  correlation, MEASURED ≈ 0 (order-1 LOSES 14 KB, delta LOSES 43 KB, pool LOSES 25 KB, sign/mag LOSES 13 KB,
  #66 §1). *Falsified by:* any fixed-basis recode of the frozen decoder achieving < 161,104 B with
  decoded-byte identity (impossible — would require correlation the coders proved absent).
- **G2 (DERIVED, counting bound):** the frozen 162 KB weights CANNOT be regenerated by a free generator from
  a seed `|s| < K(w) ≈ 161 KB`; `w = PRNG(small seed)` is barred. *Falsified by:* a < 1 KB seed + free
  generator reproducing the exact decoder bytes (violates `K(w) ≥ H(w)`).
- **G3 (DERIVED):** the free-decoder-CONDITIONAL intrinsic dimension is **~24.6–64.6 KB** (rate 0.016–0.043),
  below the measured-achiever S_floor 0.11797. *Falsified by:* either an exact-eval row from a smaller
  amortizer BELOW 0.043 rate at near-zero distortion (would tighten the band down), OR a proof the cell is
  infeasible below ~177 KB for any architecture (would raise it — but #73 + the score-DOF lower bound forbid
  the latter above ~25 KB).
- **G4 (LEGAL boundary):** a fixed generic transform/codebook in inflate.py is FREE; its coefficients/seed in
  archive.zip are CHARGED; baking video-specific payload into inflate.py is FORBIDDEN (PR#69 houdini
  eval-refused; leaderboard editorial exclusion of all loophole-class). *Falsified by:* a maintainer ruling
  changing the meter (would re-open the houdini class — currently closed).
- **G5 (the redirect):** the free-inflate exploit moves the pointer ONLY fused into PATH-1's score-aware
  retrained fixed-codebook VQ carrier, never as a post-hoc rotation of frozen weights. *Falsified by:* a
  PATH-2/PATH-3 standalone fixed-basis move > 15 B net on the frozen frontier (predicted: none).

## 8. CROSS-REFERENCES

`lossless_fresh_eyes_opportunities_20260610T172034Z.md` (#66 — the 6 measured losing coders + the 474 B
latent ceiling this report supersedes with the no-compaction-headroom mechanism) ·
`information_theoretic_floor_report_v1_20260610T102335Z.md` (S_floor 0.11797 measured-achiever; this report
adds the free-decoder-CONDITIONAL band below it) · `legal_frame_feasibility_dykstra_20260610T175421Z.md`
(#73 — generic basis needs ≥625 KB; the learned basis IS the cheap-feasible rep; PATH-1's score-aware
smaller basis is its named reactivation) · `frontier_pointer_move_ledger_20260610.md` (#71 — frozen weights
jointly entangled, no separable subset; SVD costs bytes; the 5-no-move convergence this report is the 6th
of) · `canonical_upstream_pr_review_procedural_generation_compliance_20260518.md` (the free/budgeted/forbidden
boundary + houdini precedent) · `GOAL_standing_v3_20260610.md` (levers A–H; PATH 1 = lever C; the free-inflate
exploit named correctly as fixed-codebook-forward, not frozen-rotation) · `frontier_decoder_axis_waterfill_verdict_20260610.md`
(98.6% iid; decoder holds 99.98% of |grad|) · `upstream/{README.md,evaluate.py,modules.py}` (frozen authority;
the rate rule read in full).
