---
title: "VCM / task-aware compression — the primitive theory layer (the actual math), bridged to architectures and to OUR exact problem + the task-RD floor verdict vs 0.118"
authority: "[research/advisory] — pointer UNMOVED 0.19110; $0; NO exact eval run"
score_claim: false
promotable: false
date: 2026-06-19
provenance:
  - "CLAUDE.md GOAL (S = 100·d_seg + sqrt(10·d_pose) + 25·B/B0); .omx/research/eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md (the exact scorer chain); SESSION_SYNTHESIS_SoT_20260617_20260618.md (campaign: measured S_floor 0.118; rate binding 62%; the pincer)"
  - "WEB (titles/authors/years/arXiv below). Theory results are CITED; the BRIDGE to our problem is MY derivation (flagged where it is an analogy vs a theorem)."
cross_refs:
  - .omx/research/eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md
  - .omx/state/canonical_frontier_pointer.json (0.19110 contest-CPU)
key_finding: >
  Our problem is EXACTLY an indirect / remote rate-distortion (task-oriented source coding) problem.
  The canonical theory (Wolf-Ziv reduction; task-oriented RD hierarchy; deterministic-IB; RDC) gives a
  rigorous, ACTIONABLE prescription we have NEVER built: the rate-optimal representation is a compressed
  SUFFICIENT STATISTIC for (SegNet-argmax, PoseNet-6dim) — i.e. code the POSTERIOR / TASK-SPACE, not RGB
  pixels — and our per-video INR ("decoder weights as the code") is the DIB / COIN paradigm, which is the
  right paradigm but is currently anchored on a NON-MINIMAL (full-RGB) sufficient statistic.
floor_verdict: >
  The theory does NOT lower 0.118 and it does NOT obviously raise it; it RE-EXPLAINS it. The binding 62%
  rate is the H(Z) of the deterministic decoder's sufficient statistic. Two opposing forces: (a) the
  surrogate-distortion reduction says we are paying to reconstruct MORE than the task needs (an over-count
  argument that the floor could be LOWER); (b) the indirect MMSE-floor + the texture-wall pincer say there
  is an IRREDUCIBLE D_min (the task variable is NOT perfectly identifiable from a byte-cheap rep) that
  bounds how low rate can go at fixed d_seg (a floor argument that 0.118's "d_seg->0 byte-cheaply"
  assumption is OPTIMISTIC — consistent with the campaign's pincer). Net: 0.118 is a LOWER bound that is
  likely NOT TIGHT on the achievable side; the achievable task-RD floor is plausibly HIGHER than 0.118 but
  LOWER than 0.191 IF a task-space (not pixel-space) representation is built. That gap is the prize.
---

# The primitive theory layer of coding-for-machines, bridged to our exact problem

Operator: explore the foundations of task-aware compression / coding-for-machines DOWN TO THE MATH, then
bridge to architectures and to OUR problem; every result must name the implication for lowering S or the
floor it implies. `[research/advisory]`, $0, pointer UNMOVED 0.19110. This is honest: I flag every place a
theorem is being used as an ANALOGY rather than a proof, and I flag what is unverifiable without an exact row.

---

## 0. Our problem stated in information-theoretic primitives (the object the theory acts on)

Let `X` = the source video (one clip, 600 frame-pairs, camera-res RGB). Let the **task variable** be
`Y = (Y_seg, Y_pose)` where:
- `Y_seg = argmax SegNet(s(X)) ∈ {0..4}^(384·512)` — a DETERMINISTIC function of X (frozen net), per pixel.
- `Y_pose = PoseNet(s(X))[:6] ∈ R^6` — a DETERMINISTIC, smooth function of X (frozen net), per pair.

We emit an `archive.zip` of `B` bytes that, via `inflate.sh`, deterministically produces a reconstruction
`X̂` (raw uint8 camera-res frames). The score reads ONLY the task variable of X̂ against the task variable
of X:

  `S = 100·d_seg + sqrt(10·d_pose) + 25·B/B0`,  `B0 = 37,545,489`,
  `d_seg = mean[ Y_seg(X̂) ≠ Y_seg(X) ]`  (Hamming / 0-1 distortion on the seg label field),
  `d_pose = MSE( Y_pose(X̂), Y_pose(X) )`  (squared-error distortion on the pose vector).

**This is the central observation that the whole memo turns on:** the distortion is measured on `Y = f(X)`,
NOT on `X`. We do not need `X̂ ≈ X`; we need `f(X̂) ≈ f(X)`. That is the literal definition of an
**indirect / remote / task-oriented source coding problem**, and there is a mature theory for exactly it.
Our objective in one sentence:

> **the minimum number of archive bytes B such that a FROZEN SegNet's argmax and a FROZEN PoseNet's 6-dim
> output are preserved within tolerance — over ONE video.**

The "over ONE video" + "decoder weights are the code" facts pin us to a SPECIFIC corner of the theory
(per-instance / deterministic-encoder coding), covered in §6.

---

## 1. PRIMITIVE 1 — Shannon rate-distortion R(D) (the parent)

**The function** (Shannon 1959; Cover-Thomas Ch.10):
  `R(D) = min_{ p(x̂|x) : E[d(X,X̂)] ≤ D } I(X; X̂)`.
It is the infimum of rates (bits/source-symbol) achievable with average distortion ≤ D. It is **convex,
non-increasing** in D. Its operational meaning for us: the rate term `25·B/B0` is an (un-normalized) proxy
for `R`, but with TWO twists that the rest of the memo develops: (i) our D is on `f(X)` not `X` (indirect,
§4); (ii) our "rate" is the description length of a SINGLE realization's deterministic decoder, i.e. a
H(Z)-style codelength, not an I(X;Z) ensemble rate (DIB/Kolmogorov, §6).

Quadratic-Gaussian closed form (the only fully closed case, used for the floor calc in §8):
  `R(σ², D) = ½ log₂(σ²/D)` for `0 < D ≤ σ²`, else 0.   [ScienceDirect RD overview; Cover-Thomas]
**Implication for S:** R(D) is convex-decreasing, so the LAST bits of distortion reduction are the most
expensive — exactly why the campaign sees d_seg get exponentially hard near the floor. The score's
`sqrt(10·d_pose)` term ADDS its own convexity (∂S/∂d_pose = 5/sqrt(10·d_pose) = 85.8 at the operating
point — the √-fragility already in the roundtrip memo): pose distortion is doubly-penalized as it shrinks.

---

## 2. PRIMITIVE 2 — Information Bottleneck (the task-relevant compression Lagrangian)

**The Lagrangian** (Tishby-Pereira-Bialek 1999; Tishby-Zaslavsky 2015):
  `L_IB[p(z|x)] = I(X;Z) − β·I(Z;Y)`,  minimized over the stochastic encoder `p(z|x)`,
where `Z` = the compressed representation and `Y` = the task variable, under the Markov chain `Y — X — Z`.
`I(X;Z)` is the rate (compression); `I(Z;Y)` is the task-relevance (predictive sufficiency). β is the
inverse temperature / RD slope: small β → compress hard (toss task-irrelevant info); large β → keep all
task info. The fixed-point ("self-consistent") equations
  `p(z|x) ∝ p(z)·exp(−β·D_KL[ p(y|x) ‖ p(y|z) ])`
say the optimal code groups inputs whose **task-posteriors `p(y|x)` are close in KL** — i.e. Z only needs
to resolve X up to its effect on Y. [emergentmind IB topic; arXiv 1904.03743 IB+DL survey]

**The exact mapping to us:** `Y = (Y_seg, Y_pose)`. IB says: the optimal `Z` (our archive payload) should
**preserve only what changes the SegNet argmax and the PoseNet 6-vector, and discard everything else.**
This is the formal statement of "code only what the scorer reads." We have known this informally
(the ker(D) null-space lever, the margin-polytope lever in the roundtrip memo); IB is its parent law.

**THE DETERMINISTIC-Y SUBTLETY (critical for us, and a trap).** Our `Y = f(X)` is a DETERMINISTIC function
of X. When Y is deterministic in X, `I(Z;Y)` is piecewise-linear in the IB plane and the standard IB
Lagrangian is **non-strictly-concave → it CANNOT trace the interior of the RD curve; gradient methods snap
to corner solutions** (you get either "keep everything" or "keep nothing", not the useful middle).
[Rodriguez-Galvez et al., "The Convex Information Bottleneck Lagrangian", arXiv 1911.11000 / Entropy 2020].
The fix is the **convex IB Lagrangian**: replace `−β·I(Z;Y)` with `−β·u(I(Z;Y))` for a convex `u(·)`
(e.g. an exponential/log shaping), which restores strict concavity and lets you select ANY point on the
curve by choosing β. **Implication for our training loss:** a naive "minimize rate − β·task-fidelity"
objective on a deterministic scorer will degenerate; the score-aware loss must SHAPE the task term
(margin/CE surrogates, the convex-u trick) to land on an interior operating point. This is the theory
behind why "margin-hinge throughout" (campaign's canonical seg lever) works where a raw flip-count loss
(zero-gradient a.e.) does not — it is a convexified, gradient-bearing surrogate for the piecewise-linear
deterministic-IB task term. (BRIDGE/analogy: margin-hinge is not literally the convex-u of 1911.11000,
but it is the same fix — make the deterministic task term strictly-curved and gradient-bearing.)

---

## 3. PRIMITIVE 3 — Deterministic IB (the codelength version — this is OUR rate term)

**The DIB** (Strouse-Schwab 2017, "The Deterministic Information Bottleneck", UAI/Neural Comp.):
  `max_f I(Y; f(X))  s.t.  H(f(X)) ≤ R`,
i.e. the encoder is a DETERMINISTIC map `f` (hard clustering), and the cost is the **entropy / codelength
`H(Z)` of the representation, NOT the channel rate `I(X;Z)`.** Strouse-Schwab argue H(Z) "better captures
compression" (it is the literal number of bits to STORE Z), and the optimal encoder is a hard partition.

**This is the most exact primitive for our problem, and it is under-appreciated in our prior memos.** Our
archive is a DETERMINISTIC artifact (decoder weights + latents); `B` (the bytes) is precisely `H(Z)` =
the codelength of a single deterministic representation, NOT an ensemble `I(X;Z)`. So the object we are
minimizing is the DIB objective, not the IB objective: **min H(Z) s.t. Z determines an X̂ with
I(Y; Y(X̂)) high enough (d_seg, d_pose small).** Two consequences:
- The "rate" we pay is a one-shot DESCRIPTION LENGTH (Kolmogorov/MDL flavor), which is why entropy-coding
  the payload to its Shannon floor (the campaign measured the frontier IS at 8.0 bits/byte) exhausts the
  LOSSLESS rate lever — DIB says the remaining rate must come from a SMALLER deterministic representation
  (fewer/cheaper task-sufficient bits), i.e. a LOSSY model-side change. The campaign reached this verdict
  empirically ("sub-0.15 rate REQUIRES a LOSSY model-side change"); DIB is the reason.
- Hard-partition optimality says the task-sufficient representation can be DISCRETE/quantized without loss
  of task-relevance — supportive of aggressive quantization (QAT) of the decoder, PROVIDED the partition
  it induces still separates the task classes (the int5 d_seg-wall in the campaign = the partition started
  merging task-distinct inputs; per-channel/LSQ = finer partition = the re-test #147).

---

## 4. PRIMITIVE 4 — Indirect / remote rate-distortion (THE problem class we are in) + the Wolf-Ziv reduction

This is the heart of the bridge. We never observe `Y` to be coded directly in a vacuum; we have `X`, and
the score grades `f(X̂)`. That is the **indirect (a.k.a. remote) source coding** setting.

**The indirect RD function** [Wolf-Ziv 1970; Witsenhausen; modern: Kipnis-Rini-Goldsmith; Shao-Zhang-... "An
Indirect Rate-Distortion Characterization for Semantic Sources", arXiv 2201.12477; "Model-Aware
Rate–Distortion Limits for Task-Oriented Source Coding", arXiv 2602.12866]:
  `R_X(D_Y) = min_{ p(ŷ|x) : E[d(Y,Ŷ)] ≤ D_Y } I(X; Ŷ)`,  under `Y — X — Ŷ`.
We minimize information about X needed to hit a TASK distortion `D_Y` on `Y`, NOT on X.

**THE WOLF-ZIV REDUCTION THEOREM (the single most useful result in this memo).** The indirect problem
reduces to a STANDARD rate-distortion problem with a **surrogate distortion** obtained by taking the
conditional expectation of the task distortion over the hidden variable given the observation:
  `d̃(x, ŷ) = E[ d(Y, ŷ) | X = x ] = Σ_y p(y|x) · d(y, ŷ)`,
and then `R_X(D_Y) = min_{p(ŷ|x): E[d̃(X,Ŷ)]≤D_Y} I(X;Ŷ)` is an ordinary R(D) with distortion `d̃`.
[Confirmed in arXiv 2507.17432 Thm 1 / eq.13 form `d′(x,ŝ)=Σ_s p(s|x) d_s(s,ŝ)`; "Rate-Distortion Risk in
Estimation from Compressed Data" arXiv 1602.02201; "estimate-and-compress" literature 1707.00420.]

**What the reduction PRESCRIBES (and what we have never built):** the optimal indirect scheme, in the
canonical separation, is **(1) form the posterior / Bayes-optimal estimate of the task variable given the
observation, THEN (2) standard-quantize that estimate.** The "estimate-and-compress" / "estimate-then-
compress" structure: code a SUFFICIENT STATISTIC of X for Y, not X itself. The task-oriented hierarchy
(arXiv 2602.12866) makes the bytes-ordering explicit:
  `R_Y(D_Y)  ≤  R_X(D_Y)  ≤  R_Ỹ(D_Y)  ≤  R_Ỹ^{E&C}(D_Y)`
  ( code-Y-directly  ≤  indirect-optimal-from-X  ≤  code-a-point-estimate  ≤  naive estimate-then-compress ).
And the paper's punchline: **the Bayesian sufficient statistic for task reconstruction is the conditional
posterior `V = p(·|X)`, NOT a hard point estimate** — coding soft task-posteriors beats coding hard labels.

**The DIRECT implication for our score and our representation choice (the highest-value idea in this memo):**
Our current vehicles (PR95/HNeRV/the bc20 INR) sit at or WORSE than `R_Ỹ^{E&C}`: they reconstruct full RGB
`X̂` (a point estimate of the entire image) and let the frozen scorer re-derive `Y`. That is the most
EXPENSIVE corner of the hierarchy — we pay to reconstruct all the texture, lighting, sky, etc. that the
SegNet/PoseNet argmax/pose are invariant to. The theory says a representation closer to `R_X(D_Y)` —
coding only a task-sufficient statistic — is provably ≤ our current rate at the same task distortion. **We
have an entire dominated rung of the hierarchy to climb.** The catch (the honest counter, §3 + §7): we are
FORCED by the harness to emit raw camera-res RGB frames (`TensorVideoDataset` requires uint8 frames) — we
cannot literally ship `p(y|x)`. So the win is not "ship the posterior"; it is "build a decoder whose CODE
(weights+latents) is a compressed task-sufficient statistic and whose forward map paints the cheapest RGB
that lands every pixel in the right SegNet argmax cell and the right PoseNet pose." That is precisely the
margin-polytope / texture-wall geometry the roundtrip memo found — and §4 is its information-theoretic
PARENT: the polytope IS the task-equivalence class; coding only enough to pick the right class is `R_X(D_Y)`.

---

## 5. PRIMITIVE 5 — Rate-Distortion-Perception, and the Rate-Distortion-Classification function

**RDP** (Blau-Michaeli 2019, "Rethinking Lossy Compression", ICML; Blau-Michaeli 2018 perception-distortion):
  `R(D,P) = min_{p(x̂|x)} I(X;X̂)  s.t.  E[d(X,X̂)] ≤ D  AND  d_p( p_X , p_X̂ ) ≤ P`,
where `d_p` is a divergence between the SOURCE distribution and the RECONSTRUCTION distribution (KL/JS/
Wasserstein). The theorem: tightening the perception constraint (P→0, "realism") ELEVATES the RD curve —
you must spend more rate or accept more distortion. The sharp quantitative result for perfect realism:
**`D^R(0) = 2·D^R(∞)`** — without common randomness, perfect realism DOUBLES the MSE at a fixed rate, and
the optimal decoder is POSTERIOR SAMPLING `P_{X̂|Y} = P_{X|Y}` [Theis-Wagner; Wagner; Chen-Yu-... role of
common randomness, arXiv 2202.04147, 2404.01111; "On the RDP function" 2204.06049].

**The decisive implication for our score — a TAX WE DO NOT OWE (and must not accidentally pay).** Our score
imposes **no perception/realism constraint.** Nothing checks that `X̂` looks like a real driving video;
only `f(X̂)` is graded. Therefore the RDP theory tells us: **do NOT optimize for realism / FID / pixel-MSE
/ "looks like the video".** Any byte spent on perceptual fidelity (the `P` constraint) is pure waste on our
score, and the 2× distortion-doubling penalty for realism is a penalty we should REFUSE. The campaign's
"half-res store induces d_seg 0.00554" and "per-pixel RGB seg-correction sidecar fails" negatives are
consistent: pixel-faithful moves don't help; task-faithful moves do. **The correct objective is
Rate-Distortion-CLASSIFICATION, not Rate-Distortion-Perception.**

**RDC** (the right framework for the seg term) [Liu et al. "Rate-Distortion-Classification approach", Sig.
Proc. 2023; "A Theory of Universal Rate-Distortion-Classification Representations", arXiv 2504.09932;
2504.13191/2504.09025]:
  `R(D,C) = min_{p(x̂|x)} I(X;X̂)  s.t.  E[d(X,X̂)] ≤ D  AND  P[ class(X̂) ≠ class(X) ] ≤ C`,
convex in (D,C). The "universal representation" line of work asks: can ONE representation be simultaneously
near-optimal for distortion AND classification? Result (qualitative, from the abstracts; full theorem PDF
did not extract): there EXISTS a universal representation but with a quantifiable penalty — preserving
classification costs strictly less rate than preserving reconstruction (classification = a coarse
deterministic function = few bits), and the joint optimum is NOT the reconstruction optimum.

**Implication for our score:** our seg term is the `C` constraint (an argmax-preservation / 0-1 loss); our
problem is `R(D, C)` where `D` is whatever pixel fidelity the scorer's CONV FEATURES need (the texture wall
— the seg net is not a pure label function, it reads local texture) and `C` is the argmax-flip rate. RDC
says preserving the classification (argmax) needs FAR fewer bits than preserving the image, AND there is a
universal-representation penalty when you must ALSO serve pose. This explains the campaign's seg↔rate
coupling: you cannot make `C` arbitrarily small without spending `D`-rate, because the frozen SegNet's
argmax is not a clean low-dim function — it depends on texture evidence (the measured texture wall).

---

## 6. PRIMITIVE 6 — Per-instance / Kolmogorov coding (why our archive IS the DIB/COIN corner)

Two facts pin us to a specific corner that changes which theorems apply:
1. **ONE video.** There is no ensemble; the "rate" is the description length of a single realization. The
   relevant limit is algorithmic (Kolmogorov complexity `K(Z)` / MDL `min |model| + |data|model|`), of
   which Shannon `R(D)` is the ensemble shadow. DIB's `H(Z)` (§3) is the operational stand-in.
2. **Decoder-weights-as-code (overfitting).** Our INR/HNeRV decoder is the **COIN paradigm** [Dupont et al.
   "COIN: Compression with Implicit Neural Representations", 2021; "INR for Image Compression" 2112.04267;
   RECOMBINER 2309.17182]: overfit a small net to ONE signal, ship the quantized weights. Trades (free,
   compress-time) encoder optimization for tiny decoder bytes — exactly our regime (unlimited compress
   compute, 30-min inflate budget, bytes-graded).

**The amortization angle (a design lever the theory surfaces).** Amortized AE codecs (Ballé) pay an
"amortization gap" — a static shared prior cannot match per-input statistics [arXiv 2406.13059; 2006.04240].
Per-instance INR/COIN has NO amortization gap (it overfits this video) but pays in encoder time and in the
absolute size of the weight code. **The frontier and our vehicles are already on the right (per-instance)
side of this tradeoff.** The open question the theory raises: are the decoder WEIGHTS themselves coded as a
task-sufficient statistic, or as a generic function approximator? The DIB answer (§3) + the indirect-RD
answer (§4) agree: **the weights should encode the task-equivalence partition, not a generic image model.**

---

## 7. THE BRIDGE — primitive → architecture → our exact problem (one table)

| primitive (the math) | architectural manifestation | what it says about OUR S | have we built it? |
|---|---|---|---|
| Shannon R(D), convex | entropy coder at the payload's Shannon floor | last bits cost most; lossless recode is exhausted (8.0 b/byte) | YES (frontier at floor) |
| IB Lagrangian `I(X;Z)−βI(Z;Y)` | score-aware training loss, β = RD slope | "code only what the scorer reads" is the parent law of every null-space/margin lever | PARTIALLY (score-aware loss yes; explicit β-sweep no) |
| convex-IB fix (deterministic Y) | margin/CE surrogate w/ curvature, NOT raw flip-count | why margin-hinge works & flip-count loss degenerates | YES (margin-hinge throughout) |
| DIB `min H(Z) s.t. I(Y;Z)` | the archive = a deterministic codelength; quantized weights | rate lever must be a SMALLER task-sufficient rep (lossy), not a recode; QAT-OK if partition preserved | EMERGING (QAT re-test #147) |
| indirect-RD + Wolf-Ziv surrogate `d̃=E[d(Y,·)|X]` | code a SUFFICIENT STATISTIC for (seg-argmax, pose), not RGB | **we sit at the DOMINATED `R_Ỹ^{E&C}` rung — code task-space, climb to `R_X(D_Y)`** | **NO — the prize** |
| RDP (realism tax `D(0)=2D(∞)`) | FID/realism/pixel-MSE objectives | a tax we DON'T owe — refuse perceptual fidelity; the score is task-only | (avoid; partly internalized) |
| RDC `R(D,C)` + universal-rep penalty | joint seg(C)+recon(D) representation | preserving argmax << preserving image; seg+pose joint pays a universal-rep penalty | implicit (the seg↔rate coupling IS this) |
| per-instance / COIN | overfit decoder weights as the code | right paradigm; no amortization gap; weights should encode the task partition | YES (paradigm) / NO (task-partition weights) |

The single sentence the bridge produces: **we are solving an indirect rate-distortion problem with a DIB
codelength and an RDC (not RDP) distortion, in the COIN per-instance corner — and our representation is
anchored on the WRONG (full-RGB, `R_Ỹ^{E&C}`) rung of the hierarchy; the theory's prescription is to build
a decoder whose code is a compressed task-sufficient statistic.**

---

## 8. THE TASK-RD FLOOR ANALYSIS (vs the measured S_floor = 0.118)

The CLAUDE.md / campaign floor `S_floor = 0.11797` was derived assuming `d_seg → 0` is achievable
byte-cheaply (rate-dominated floor: rate 0.0594 + pose 0.0585 with d_seg negligible). The pincer
(SESSION_SYNTHESIS) then MEASURED that d_seg→0 is NOT byte-cheap (flat reps survival-wall ~0.0067;
continuous reps capacity-wall ~params^-0.71). What does indirect-RD theory say about the floor?

**(A) The over-count argument (floor could be LOWER than the achievable we see — headroom).** Wolf-Ziv §4:
our vehicles pay to reconstruct full RGB `X̂` and let the scorer re-derive Y. The achievable
`R_Ỹ^{E&C}(D_Y)` (our rung) is `≥ R_X(D_Y)` (indirect-optimal). The gap `R_Ỹ^{E&C} − R_X` is bytes we are
spending on task-IRRELEVANT reconstruction (texture/lighting the argmax+pose ignore). This is a strict
inequality whenever the scorer is invariant to some image variation (it is — huge ker(D), huge
margin-polytope interior). **So the achievable task-RD floor is STRICTLY BELOW our 0.191 frontier**, and
plausibly below 0.118 ON THE RATE AXIS if the representation were task-space — BUT only down to the point
where the seg-`D` (texture) and pose-`D` constraints bind.

**(B) The irreducible-D_min argument (floor is HIGHER than 0.118's optimistic d_seg→0).** Remote/indirect
RD has an IRREDUCIBLE distortion floor: even at infinite rate, `R_X(D_Y)` is finite only for
`D_Y ≥ D_min`, where `D_min` is the Bayes error of estimating `Y` from `X` (here ~0 since Y=f(X) is
deterministic in the FULL X — but NOT 0 from a byte-cheap, roundtrip-degraded X̂). The Gaussian intuition:
remote `R(D) = ½ log( σ_eff² / (D − D_min) )` blows up as `D → D_min⁺` [remote-Gaussian RD, PMC7514694 /
arXiv 1805.06515; quadratic form `½log(σ²/D)`]. **Translated to us:** as we push `d_seg → 0` with a
byte-cheap rep, the texture-wall is the `D_min` of a degraded observation — the rate to hold a small
`d_seg` DIVERGES (the campaign's exponential/stretched-exp d_seg-vs-epochs/params curves ARE this blow-up).
So the **0.118 floor's assumption "d_seg→0 at the 0.0594 rate" is information-theoretically OPTIMISTIC**:
holding frontier-grade d_seg AT well-below-frontier bytes is exactly the `D → D_min⁺` regime where rate
explodes. This is the theory's confirmation of the campaign's "capacity↔rate tension" crux.

**(C) THE FLOOR VERDICT (raw, for the parent).** The theory neither cleanly lowers nor cleanly raises
0.118 — it RE-FRAMES it as a LOWER BOUND that is **not tight on the achievable side**, with two opposing
corrections:
- 0.118 IGNORES the over-count headroom (B): a TASK-SPACE representation could beat it on rate at fixed
  task distortion → the true ACHIEVABLE-with-the-right-representation floor is plausibly **below 0.191 and
  the rate term could go below 0.0594**, IF (and only if) we climb off the `R_Ỹ^{E&C}` rung.
- 0.118 IGNORES the D_min blow-up (B'): holding small d_seg byte-cheaply is the divergent regime, so the
  *jointly* achievable (small d_seg AND small rate) point that 0.118 imagines may be INFEASIBLE with any
  representation → the realistic floor on the FULL S is plausibly **above 0.118**.
- Net honest estimate: **the achievable task-RD floor S* most likely lies in (0.118, 0.191), strictly
  inside both bounds, and is REACHABLE only by a task-space (indirect-RD) representation we have not built.**
  0.118 should be relabeled "rate-only lower bound assuming free d_seg" (loose); 0.191 is "best achievable
  with the dominated full-RGB rung." The PRIZE is the gap, and the theory says the lever to capture it is
  REPRESENTATION (climb the hierarchy), not more epochs on the current rep.
**Unverifiable-without-an-exact-row caveat:** every number here is an information-theoretic bound/analogy;
the Gaussian closed forms are an ANALOGY (our source is not Gaussian, our distortion is argmax-0-1 + MSE),
and `R_X(D_Y) < R_Ỹ^{E&C}(D_Y)` is a THEOREM but its MAGNITUDE for our specific (SegNet,PoseNet) is
unmeasured. The §9 probes are designed to MEASURE the gap cheaply before any build.

---

## 9. RANKED $0 PROBES THE THEORY IMPLIES (measurement-first; each names the theorem it tests)

Ranked by `|EV toward a lower exact S| / cost`. All $0, all on the existing frontier-decoded frames; none
conflicts with the running NCA gate. Each tests a SPECIFIC theory claim so the result is decisive.

1. **[HIGHEST] P-SUFF — measure the indirect-RD hierarchy gap `R_Ỹ^{E&C} − R_X` (the over-count, §4/§8A).**
   The theory's biggest claim is that we sit on a dominated rung. TEST it cheaply: on the frontier decoded
   frames, measure how much of the reconstructed RGB the scorer is INVARIANT to. Concretely — (a) take the
   frontier X̂; (b) replace each per-pixel RGB with the CLASS-CONDITIONAL MEAN texture-patch (or a
   low-rank/PCA task-sufficient projection that keeps the SegNet conv features inside the argmax polytope
   AND the YUV6 pose carrier) → a "task-space-only" reconstruction; (c) measure d_seg/d_pose through the
   REAL roundtrip + estimate the byte cost of coding ONLY that task-sufficient projection. If task-fidelity
   holds at materially fewer bits → the dominated-rung gap is REAL and large → spec a task-space decoder.
   This is the decisive go/no-go for the §4 prize. (Sister to the roundtrip memo's P1 margin-polytope but
   AIMED at the rate axis / sufficient-statistic, not just the boundary residual.)

2. **[HIGH] P-βSWEEP — convex-IB operating-point sweep (§2 deterministic-IB fix).** Confirm our training is
   on the interior, not a corner. Sweep the task-term weight/shaping (the β / convex-u knob) on a short
   bc20 run; plot realized (rate, d_seg, d_pose). If the curve is a clean interior trade (not a corner
   snap) → the convex surrogate is working and tells us the LOCAL RD slope `dB/dd_seg` = the exact
   exchange-rate between bytes and seg-flips at our operating point (directly feeds the bit-allocator /
   the #147 QAT allocation). $0 (reuses checkpoints).

3. **[HIGH] P-RDP-REFUND — audit + remove any realism tax (§5).** Grep every active loss/objective for a
   pixel-MSE / perceptual / FID / "reconstruct the image" term that is NOT routed through the scorer. The
   RDP theorem says these cost up to 2× distortion for ZERO score benefit. Each one found is a free refund
   (reweight toward the task term). $0 code audit; immediate.

4. **[MED] P-RDC-UNIV — measure the seg↔pose universal-representation penalty (§5 RDC).** On the frontier,
   measure d_seg and d_pose when the decoder capacity is allocated (i) jointly vs (ii) split-by-head
   (already partly done via --split-by-head). RDC predicts a quantifiable joint penalty; if splitting the
   representation (separate task-sufficient codes for seg vs pose) reduces total bytes at fixed (d_seg,
   d_pose), that is the universal-rep penalty made concrete → a representation-factoring lever.

5. **[MED] P-DIB-QAT — DIB partition-preservation test for QAT (§3, feeds #147).** DIB says quantization is
   free IFF the induced hard partition still separates task classes. Before the int5 per-channel/LSQ
   re-test, MEASURE the per-channel quant-error budget that keeps each pixel inside its SegNet argmax
   polytope (the margin field from the roundtrip memo's P1 gives this directly) → allocate quant bits by
   margin. Turns #147 from a blind retry into a margin-allocated QAT. $0 (margin field + weight stats).

6. **[LOW/FRAMING] P-FLOOR-RELABEL — register the corrected floor semantics.** Relabel 0.118 in the
   canonical equations registry as "rate-only LOWER bound (assumes free d_seg; NOT tight)" and add the
   §8C verdict (achievable S* in (0.118,0.191), representation-gated). Prevents future agents from citing
   0.118 as a TARGET; it is a loose bound. Pure hygiene, $0, but it stops a recurring signal-loss.

---

## 10. ONE-PARAGRAPH SYNTHESIS (the ends, stated plainly)

Our contest is, in the precise language of information theory, an **indirect (remote) rate-distortion
problem**: minimize a deterministic codelength (the DIB `H(Z)` of the archive) such that a frozen SegNet's
argmax (an RDC classification constraint) and a frozen PoseNet's 6-vector (an MSE distortion) are
preserved over one video, in the per-instance COIN corner. The Wolf-Ziv reduction says the rate-optimal
representation codes a **sufficient statistic for the task** (the surrogate distortion `E[d(Y,·)|X]`),
which provably needs **strictly fewer bytes** than reconstructing full RGB and letting the scorer re-derive
Y — and that is exactly what every one of our vehicles (PR95/HNeRV/bc20) currently does, placing us on the
DOMINATED top rung of the task-RD hierarchy. The Rate-Distortion-PERCEPTION theory adds a sharp negative:
the score imposes NO realism constraint, so any byte spent on pixel/perceptual fidelity is a tax we don't
owe (and realism would cost up to 2× distortion for nothing). The floor verdict: 0.118 is a LOOSE
rate-only lower bound (it ignores both the over-count headroom AND the d_seg→D_min rate blow-up); the
achievable task-RD floor S* most likely lies strictly inside (0.118, 0.191) and is reachable ONLY by
building a task-space (indirect-RD / sufficient-statistic) representation — NOT by more epochs on the
current full-RGB rep. **The single highest-EV move the theory surfaces is the one $0 probe we never ran:
P-SUFF — measure how many of the frontier's reconstructed RGB bits the frozen scorer is actually invariant
to (the `R_Ỹ^{E&C} − R_X` gap). If that gap is large, the prize (a lower exact S via a task-sufficient
decoder) is real and the campaign should pivot the representation, not grind the epochs.**

---

## Sources (web; titles / authors-where-known / year / arXiv-or-venue)
- Shannon RD / quadratic-Gaussian closed form: ScienceDirect "Rate Distortion Function" overview; iphome.hhi.de RD theory notes; Cover-Thomas Ch.10 (standard).
- Information Bottleneck: Tishby-Pereira-Bialek 1999; Tishby-Zaslavsky 2015; survey arXiv:1904.03743; emergentmind IB topic.
- Convex IB Lagrangian (deterministic-Y fix): Rodriguez-Galvez et al., "The Convex Information Bottleneck Lagrangian", arXiv:1911.11000 / Entropy 22(1):98 (2020).
- Deterministic IB: Strouse & Schwab, "The Deterministic Information Bottleneck", arXiv:1604.00268 / UAI 2016 / Neural Computation 29(6):1611 (2017).
- Indirect/remote RD + reduction: Wolf & Ziv 1970 (foundational); Kipnis-Rini-Goldsmith "indirect RD of a binary i.i.d. source" arXiv:1505.04875; Shao et al. "An Indirect Rate-Distortion Characterization for Semantic Sources" arXiv:2201.12477; "Non-Asymptotic Achievable RD Region for Indirect Wyner-Ziv" arXiv:2507.17432 (Thm 1 surrogate distortion); "Rate-Distortion Risk in Estimation from Compressed Data" arXiv:1602.02201; "Compress-and-Estimate ... Vector Gaussian" arXiv:1707.00420.
- Task-oriented source coding hierarchy + posterior sufficient statistic: "Model-Aware Rate–Distortion Limits for Task-Oriented Source Coding" arXiv:2602.12866.
- Rate-Distortion-Perception: Blau & Michaeli, "Rethinking Lossy Compression: The RDP Tradeoff", ICML 2019 (arXiv:1901.07821); "On the RDP Function" arXiv:2204.06049; "The RDP Tradeoff: The Role of Common Randomness" (Theis-Wagner / Wagner) arXiv:2202.04147; "...Private Randomness" arXiv:2404.01111.
- Rate-Distortion-Classification: Liu et al., "A Rate-Distortion-Classification approach for lossy image compression", Signal Processing 2023; "A Theory of Universal Rate-Distortion-Classification Representations" arXiv:2504.09932; arXiv:2504.13191 / 2504.09025.
- Remote Gaussian RD closed form / MMSE floor: "Asymptotic RD Analysis of Symmetric Remote Gaussian Source Coding" PMC7514694; "Remote Source Coding under Gaussian Noise" arXiv:1805.06515.
- Per-instance / COIN: Dupont et al., "COIN: Compression with Implicit Neural Representations" (2021, openreview yekxhcsVi4); "Implicit Neural Representations for Image Compression" arXiv:2112.04267; RECOMBINER arXiv:2309.17182; amortization gap arXiv:2406.13059 / 2006.04240.
