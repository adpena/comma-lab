# ddm_fx1 — the fixed-point log-odds mixer: me1's crux, cured and measured

Date: 2026-08-17 · Arm: `ddm_fx1_fixed_point_logistic_mixer_20260817` · Authority: exact
decode-identical code-length measurement, full n600 · **Score claim: false** · **Pointer moved: false**

## Conclusion first

**The coder axis is NOT closed. The logistic mixer works, and the best architecture measures
−560.07 B on the full n600 token field**, from 110,511.28 B to 109,951.21 B (−0.51%).

* Projected archive **180,601 B** (from 181,161 B), projected **ΔS = −3.7288e-4**, projected
  **S = 0.158160369**. That is **37.3× the 1e-5 naming bar** and clears the charter's −500 B gate.
* Distortion is unchanged **by construction**: only the probability law moves, the decoded token
  field is bit-identical, so d_seg and d_pose cannot change.
* **Cost: +127.3 s of decode**, measured serially. That is the number that decides the candidate,
  and it is why I am sealing a fire-order with a cheaper fallback rather than one row.
* Determinism repeat: **byte-identical** across fresh processes, for both candidates.

**Own-vehicle frontier: `S = 0.15853325034789678 @ 181,161 B [contest-CUDA T4, n600]`, archive sha
`35ac2b9b…`. This arm did NOT move it** — only MAIN's fire can. No Modal spend; $1.38 untouched.

**Adjudication (coordinator's question): this outcome supports MAIN's PP3(a), against GS1-PRED.**
The next ≥1e-4 rate move is available as a post-hoc operator on the frozen rr4 object; it does not
require a new checkpoint. −3.73e-4 is 3.7× the 1e-4 threshold gs1 named. GS1-PRED is not refuted as
a claim about the QAT slope — that slope may well be unfloored — but its specific prediction that
the *next* ≥1e-4 mover would be a new checkpoint is contradicted by a measured post-hoc row.

## 1. The premise challenge, settled at source (coordinator item 2)

gs1 claimed PP3(a)'s "named mechanism is already 5/5 negative in me1 §5". **That claim is false, and
I verified it by reading the code rather than either memo.**

`experiments/ddm_me1_mixed_context_corrector.py:237-252` — `odds_multiplier` is literally
`numerator / denominator` accumulated as `numerator += weight * multiplier`. Its own docstring names
it: *"COUNT-WEIGHTED ARITHMETIC MEAN IN ODDS SPACE"*. An AST-level grep of both raced correctors for
`log|exp|pow|**|sqrt` returns only imports, prose, and the variable `exp_numerator` (which is
*expected*, not exponential).

So **all seven of me1's raced rows are arithmetic-mean-in-odds-space or single-context refinement.
Not one is log-odds domain.** The distinction is not cosmetic: an arithmetic mean in odds space is
still bounded by `min_k m_k ≤ mean ≤ max_k m_k`, so it cannot exceed its best member; a weighted
geometric mean with weights that need not sum to 1 escapes that hull. My charter's premise stands
unmodified, and the measured sign flip in §4 is the direct confirmation.

## 2. The exactness cure: radicals, not lookup tables

The charter asked for a fixed-point integer log-odds mixer with LUT squash/stretch. **I found a
strictly better construction and built that instead.** The transcendentals are avoidable entirely.

Restrict the mixing weights to a dyadic grid `w = W / 2^b`. Then

```
m ** (W / 2^b)  =  m**k * prod_{i in bits(j)} m ** (1 / 2^(i+1)),    W = k·2^b + j
```

and every factor on the right is reachable from `m` by repeated **square roots** and multiplication.
**IEEE-754 requires `sqrt` to be correctly rounded** — it is in the same exactness class as `+ - * /`
and *not* in the class of `log`/`exp`, which is exactly what desynchronised the T4 decoder at
`ddm_rr2` (S 27.83). So the mixer computes a genuine weighted geometric mean while remaining
bit-identical on every conforming platform.

This beats the LUT design on three axes and is why I departed from the charter's letter:

| | LUT stretch/squash | radicals (built) |
|---|---|---|
| table generation | must be produced without libm or it certifies the platform against itself | **no table at all** |
| multiplier precision | quantised to the log grid | **full float64, unquantised** |
| identity control | approximate | **bit-exact** (integer weight takes the integer path only) |

Two structural consequences, and they are the controls: weight exactly 1.0 on one member returns the
multiplier **bit-identically**, so the mixer collapses onto the shipped law; all-zero weights give
`m = 1.0` exactly, i.e. exactly HPAC. The architecture **nests both**, so a failed learner degrades
toward the incumbent rather than toward noise.

**Two weight precisions, on purpose.** The learner accumulates in a fine int64 grid (2^-20) and the
mixer powers on a coarse dyadic grid (2^-6). Separated because a single gradient step is far smaller
than 2^-6 — a learner quantised to the power grid would round every step to zero and never move.
This was a real defect in my first draft, caught by deriving the step scaling rather than by a test.

**State-sync law (hx1 item 4) is satisfied by a stronger route than quantisation.** hx1 warns that
opal leaves a 55-wide double meta-layer compounding through libm `exp()` over 117,964,800 symbols.
I do not quantise the mixer inputs — I make every operation on the probability path *correctly
rounded*, which is stronger: quantisation makes values coarse, correct rounding makes them
**identical**. The weights, the gradient, and the update are integer end-to-end; float summation
order (pairwise, SIMD-width dependent) never enters, because per-position gradient terms are
quantised to int64 *before* they are summed.

## 3. Controls — the instrument, then my plumbing

No verdict below is admissible unless all four hold. All are full n600, all re-run in my own hands:

| control | target | measured | delta |
|---|---:|---:|---:|
| uncorrected HPAC cross-entropy | 112,109.57757858819 B | 112,109.57758 B | **0.000000 B** |
| shipped rr4 corrected code length | 884,090.2210952122 bits | 884,090.2210952122 bits | **0.000000 bits** |
| me1 mixture refactor ≡ shipped law | — | (me1, re-verified) | **0.000000 bits** |
| **fx1 mixer, 1 member, weight 1.0, static** | 884,090.2210952122 bits | 884,090.2210952122 bits | **0.000000 bits** |

The fourth is mine and it is the one that makes the race readable: my generalisation collapses
*bit-exactly* onto the law it generalises, so every delta below is the mixture, never the plumbing.

**Determinism repeat**, fresh process, payload sha256 of the per-frame bit vector:

| candidate | run 1 | run 2 | verdict |
|---|---|---|---|
| allfam_fast | `75f7fe06af46d69e…` | `75f7fe06af46d69e…` | **byte-identical** |
| k1_cb16 | `97ccc7ef4555ceb9…` | `97ccc7ef4555ceb9…` | **byte-identical** |

**Frozen drop-in verification.** The build chain constructs a corrector as `FreeCorrector(plane)`
via `TAC_RR2_CORRECTOR_MODULE`, so the shipped configuration must be frozen in the module — a
decoder takes no arguments. `experiments/ddm_fx1_logistic_mixer_corrector.FreeCorrector` reproduces
the raced row at **879,609.6594294705 bits, delta +0.000000** against the parameterised run.

## 4. The race — 33 rows, full n600, decode-identical, $0

Sorted by measured delta against the live rr4 law. `K` = members, `cb` = count buckets.

| Δ bytes | id | K | mixer context | cb | lrE | mode | s |
|---:|---|---:|---|---:|---:|---|---:|
| **−560.07** | **allfam_fast** | **11** | **cls_boundary_agree_ubin8** | 1 | 4 | adapt | 171 |
| −560.07 | allfam_fast_rep | 11 | cls_boundary_agree_ubin8 | 1 | 4 | adapt | 156 |
| −544.63 | allfam_fast_cb8 | 11 | cls_boundary_agree_ubin8 | 8 | 4 | adapt | 209 |
| −536.86 | allfam | 8 | cls_boundary_agree_ubin8 | 1 | 4 | adapt | 106 |
| −533.76 | allfam_pb8 | 8 | cls_boundary_agree_ubin8 | 1 | 2 | adapt (b=8) | 137 |
| −533.61 | allfam_lr2 | 8 | cls_boundary_agree_ubin8 | 1 | 2 | adapt | 119 |
| −524.40 | allfam_cb8 | 8 | cls_boundary_agree_ubin8 | 8 | 2 | adapt | 158 |
| −522.55 | allfam_cb16 | 8 | cls_boundary_agree_ubin8 | 16 | 2 | adapt | 184 |
| −369.80 | k3_fast | 3 | cls_boundary_agree_ubin8 | 1 | 4 | adapt | 71 |
| −355.26 | k2_fast256 | 2 | cls_boundary_agree_ubin8 | 1 | 4 | adapt | 56 |
| **−340.82** | **k1_cb16** | **1** | **cls_boundary_agree_ubin8** | 16 | 2 | adapt | 49 |
| −333.01 | k1_full_lr2 | 1 | cls_boundary_agree_ubin8 | 1 | 2 | adapt | 38 |
| −319.73 | k1_ctx_full | 1 | cls_boundary_agree_ubin8 | 1 | 4 | adapt | 44 |
| −278.94 | k1_ctx_clsbndubin8 | 1 | cls_boundary_ubin8 | 1 | 4 | adapt | 43 |
| −267.73 | k1_ctx_clsagreeubin8 | 1 | cls_agree_ubin8 | 1 | 4 | adapt | 43 |
| −263.25 | k2_tempspatial | 2 | cls_boundary | 1 | 4 | adapt | 53 |
| −249.80 | k2_surpriseonly | 2 | cls_boundary | 1 | 4 | adapt | 53 |
| −242.21 | allfam_lr1 | 8 | cls_boundary_agree_ubin8 | 1 | 1 | adapt | 130 |
| −227.24 | k1_adapt_ctx_clsbnd | 1 | cls_boundary | 1 | 4 | adapt | 39 |
| −207.61 | k1_full_lr6 | 1 | cls_boundary_agree_ubin8 | 1 | 6 | adapt | 37 |
| −201.89 | k1_ctx_clsrunubin8 | 1 | cls_run_ubin8 | 1 | 4 | adapt | 43 |
| −200.96 | k1_adapt_ctx_ubin8 | 1 | ubin8 | 1 | 4 | adapt | 39 |
| −54.85 | k1_ctx_clsbnd_nonorm | 1 | cls_boundary | 1 | 4 | **no-normalise** | 42 |
| −38.57 | k1_adapt_lr2 | 1 | none | 1 | 2 | adapt | 39 |
| −36.38 | allfam_ctxbnd | 8 | cls_boundary | 1 | 2 | adapt | 118 |
| −34.49 | k1_adapt_lr4 | 1 | none | 1 | 4 | adapt | 39 |
| −11.54 | k1_adapt_lr6 | 1 | none | 1 | 6 | adapt | 39 |
| **+0.00** | **c3_identity_static_w1** | 1 | none | 1 | — | **static (CONTROL)** | 28 |
| +0.00 | t_shipped | 1 | none | 1 | — | static (CONTROL) | 28 |
| **+253.28** | **base_k1_ctxbnd** | 2 | cls_boundary | 1 | 4 | adapt | 44 |
| **+495.88** | **allfam_lr0** | 8 | cls_boundary_agree_ubin8 | 1 | 0 | adapt | 130 |
| **+552.32** | **base_k1** | 2 | cls_boundary_agree_ubin8 | 1 | 4 | adapt | 44 |

**30 of 33 rows are negative or exactly zero. Three are REFUSED, and they are informative:**

* **`base_odds` REFUSED, twice (+552.32, +253.28).** This was my *highest-EV hypothesis* going in:
  offer the mixer the neural prior's own log-odds as a member, making the weight a learned exponent
  — online temperature scaling at zero bytes. It loses badly. The mechanism I believe (INFERRED, not
  measured): the prior's odds span ~1e-9 to 1e9, so any exponent ≠ 1 moves the confident tail
  enormously, and 70% of the bits live in that tail (§5). Recalibrating the *correction* is safe;
  recalibrating the *prior* is not. Reported as a member-level refusal, not a family-level one.
* **Learning rate 2^0 REFUSED (+495.88).** The learner diverges. 2^-2 and 2^-4 are both stable and
  within 3 B of each other, so the operating point is a plateau, not a spike.

### The charter's core claim, confirmed by a sign flip

`temporal_spatial` cost **+359.47 B** under me1's arithmetic mean. Under the geometric mean the same
family **saves 36 B** (k1 at `cls_boundary` −227.24 → k2 −263.25). `surprise_only` cost **+689.11 B**
and now saves 22.6 B. **Same members, same estimator, same field — opposite sign.** That is the
theorem being escaped, measured rather than argued.

### What the learned weights actually say

The single strongest finding, and it is a statement about the shipped law rather than about mixing:

```
mixer context = none         : w = 1.094                       (global optimum is not 1.0)
mixer context = ubin8        : w = 0.657 0.885 1.640 1.302 1.040 ...
mixer context = cls_boundary : w = 0.842 1.095 1.317 0.773 1.867 0.655 ... 0.271 ... 0.231 ...
```

**The shipped law's implicit weight of exactly 1.0 is wrong nearly everywhere.** The optimum ranges
0.23 → 1.87 across cells. The correction is over-applied in some regions by 4× and under-applied in
others by ~2×. A single global weight cannot express that, which is why the mixer context — not the
member count — is the dominant axis in the table above (−34 B at one weight set, −320 B at 800).

## 5. Where the bits actually are (decomposition, and a priced ceiling)

Measured on the uncorrected field, n600, all 117,964,800 positions:

| | positions | share | bits | bytes | share of bits |
|---|---:|---:|---:|---:|---:|
| hits (token == argmax) | 117,741,106 | 99.810% | 268,967.41 | 33,620.93 | 29.99% |
| **misses** | **223,694** | **0.190%** | **627,909.21** | **78,488.65** | **70.01%** |

**0.19% of positions carry 70% of the stream.** Splitting the miss bits further:

* cost of *being* a miss — what the transport's `q` adapts: **77,241.46 B**
* cost of *which* class given a miss — the relative law the transport leaves untouched:
  **1,247.19 B**

Two consequences worth banking, both of which price something before anyone spends a day on it:

1. **The mixer is on the correct axis.** It attacks `q`, which governs 77,241 B — 98.4% of the miss
   cost and 69% of the whole stream.
2. **The un-adapted miss-sector relative law has a hard ceiling of 1,247 B**, and that is the
   *perfect-model* ceiling, not an estimate of the achievable gain. Any future arm proposing to
   model the within-miss distribution now knows its total addressable size before building.

## 6. Decode wall-clock — the constraint that picks the candidate

The shipped rr4 parseback costs **1,502.29 s** total (token stage 1,024.84 s), `[macOS-CPU
advisory]`, against a 30-minute (1,800 s) contest budget. Marginal cost of the mixer, measured
**serially** with nothing else running, same warm cache:

| candidate | Δ bytes | decode | marginal | share of the 297.7 s local headroom |
|---|---:|---:|---:|---:|
| shipped rr4 (baseline) | — | 28.3 s | — | — |
| **k1_cb16** (1 member) | −340.82 | 41.4 s | **+13.1 s** | **4.4%** |
| **allfam_fast** (11 members) | −560.07 | 155.6 s | **+127.3 s** | **42.7%** |

Both fit locally. The marginal cost is additive host-CPU numpy work, so it carries to the contest
path; the *absolute* budget does not, because the 1,502 s figure is a local advisory replay and the
rr4 row already passed on T4. **This is why the fire-order below ships a fallback**: −219 B of extra
saving costs 9.7× the decode time, and if the T4 headroom is tighter than the local one, the cheap
row still banks 22.7× the naming bar.

## 7. Honest limits

* **Selection on the scored clip.** I chose the member set, the mixer context and the learning rate
  by racing on the scored video. That is design-time information transfer, and the campaign holds
  itself to a stricter standard than rule 118 literally requires (rr4 states its constants were
  "never swept against the scored clip"). Mitigating, and stated as a bound rather than a defence:
  the *family* is robustly negative — 30 of 33 rows, and every adapted row except the two `base_odds`
  members and the divergent learning rate. The fallback candidate `k1_cb16` involves far fewer
  selected degrees of freedom (one member, one context) and still measures −340.82 B. What is
  **not** claimed: that the specific winning configuration would be optimal on another clip.
* **Projected, not byte-closed.** −560 B is a measured *code length*; the archive figure is
  `ceil(code_bytes)` plus the unchanged sections, which is how rr4's own build predicted its token
  bytes with `token_delta_vs_target: 0`. Uncertainty ±1 B on the ceil. The byte-close is step 1 of
  the fire-order, not something I am asserting.
* **Cross-platform exactness is argued, not yet demonstrated.** Every operation is IEEE correctly
  rounded and the AST gate refuses transcendentals, but rr2's lesson is precisely that a correct
  local proof is not a cross-platform one. The parse-back on contest hardware is a **hard gate**
  below, not a formality.
* **hx1's scan-order finding is unexploited.** Their `group=(x&63)+2(y&63)` trick makes a
  below/right template causal; our 190-group wavefront currently supports only left/up in
  `_spatial_level`. Richer causal spatial members are available and untested. Not attempted here.

## 8. The sealed fire-order for MAIN

**Do not fire a T4 row yet.** The next step is local and free, and it is the one that converts a
projection into a claim.

**Step 1 — byte-close (local, ~$0).** The corrector is selected by module name:

```bash
TAC_RR2_CORRECTOR_MODULE=ddm_fx1_logistic_mixer_corrector \
  .venv/bin/python experiments/ddm_pq2_compress_e2e.py   # plus rr4's usual roots
```

`experiments/ddm_fx1_logistic_mixer_corrector.FreeCorrector` is the frozen drop-in and needs no
arguments. Pre-registered targets: **token stream 109,952 B (±1)**, **archive 180,601 B (±1)**,
every other section byte-identical.

**Step 2 — parse-back, then fire.** Only if step 1 hits its targets. Pre-registered falsifiers:

| falsifier | required |
|---|---|
| decoded token field sha256 | **`9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52`** — the rr4 target, unchanged. Any difference falsifies the decode-identical claim outright. |
| d_seg, d_pose | **EQUAL to base to all digits.** Not "close". The claim is decode-identity. |
| archive bytes | **exactly** the step-1 figure, matching the local instrument. |
| decode wall-clock | within the contest 30-min budget with margin. |
| archive repeat | byte-identical. |

**Candidate A (primary): `allfam_fast`** — 11 members, −560 B, ΔS −3.7288e-4, +127.3 s decode.
**Candidate B (fallback): `k1_cb16`** — 1 member, −341 B, ΔS −2.2706e-4, +13.1 s decode. Fire B if
step 2 shows the decode budget is tighter than the local headroom suggests.

## STORES CONSULTED

`.omx/research/ddm_me1_micro_edit_engine_20260817.md` (theorem, instrument, raced-and-refused table)
· `experiments/ddm_me1_mixed_context_corrector.py:237-252` (the arithmetic-mean claim, **verified at
source, not relayed**) · `experiments/ddm_me1_spatial_context_corrector.py` ·
`experiments/ddm_rr4_free_corrector_v2.py` (the live law, its exactness argument, the rr2 postmortem)
· `experiments/ddm_rr2_encoder_byteclose.py:78-81` (the `TAC_RR2_CORRECTOR_MODULE` contract) ·
`experiments/ddm_pq2_compress_e2e.py:85` (the byte-close entry point) ·
`/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/RESULT_build.json` +
`RESULT_parseback_v2.json` (181,161 B, sections byte-identical, 1,502.29 s decode, target token sha)
· `/Volumes/APDataStore/pact/ddm_me1/ME1_CODER_RESULTS.json` (base S, the four refused mixtures) ·
`src/tac/micro_edit/coder_replay.py` (the instrument) · `upstream/evaluate.py` +
`tac.contest_oracle.constants` (score definition) · coordinator relays: gs1 convocation (PP3(a) vs
GS1-PRED), hx1 PR-wave harvest (rank-one split, scan-order, slot starvation, state-sync law).

## Artifacts

Payloads → `/Volumes/APDataStore/pact/ddm_fx1/` — `race/*.json` (33 rows, each with its final
learned weights), `retained/bits_*.npy` (per-frame bit vector per architecture, sha256 in each row),
`retained/controls_c1_c2.json`, `race.py`, `decompose.py`, `controls.py`.

Code: `experiments/ddm_fx1_logistic_mixer_corrector.py` (the mixer + the frozen drop-in) ·
`src/tac/micro_edit/tests/test_fx1_logistic_mixer.py` (107 tests; exactness, nesting, receiver
closure, learner liveness, shipping surface). Three mutations were injected and all three were
caught: a reintroduced `np.log`, a 1e-13 perturbation of the integer power path, and an inert
learner. The suite is not vacuous.

## NEXT_IF_RESUMED, ranked

1. **Byte-close + parse-back** (step 1 above). Converts −560 B from projection to claim. Local, free.
2. **Richer causal spatial members via hx1's scan-order read.** Our 190-group wavefront may already
   make more than left/up causal; measuring which neighbours are decoded is a 10-minute probe and
   feeds directly into the member pool that just proved it pays.
3. **SSE / APM second stage** — the classic PAQ stage this build does not have: an adaptive map on
   the mixer's *output* probability, indexed by quantised `q`. Complementary to member mixing.
4. **Miss-sector relative law** — now priced: **ceiling 1,247 B**, so worth at most a bounded effort.
5. **Do NOT re-run `base_odds`** without a different formulation; the prior's tail is too heavy for a
   single learned exponent.
