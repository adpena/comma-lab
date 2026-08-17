# ddm_hx1 — PR-wave harvest (129–138): the biggest transfer is a ZERO-BYTE re-solve, not an entropy coder

**Date:** 2026-08-17
**Arm:** ddm_hx1 (Opus) + 3 intake sub-arms. Charter
`.omx/research/charters/ddm_hx1_pr_wave_harvest_20260817.md`.
**Operator binding 2026-08-17, verbatim:** *"We can harvest all signal from those other PRs as
well and use their tricks and techniques and incorporate them into our work."* Off-the-shelf
grant (08-10) applies; honesty-half unchanged.
**Axis:** every number is a **RECEIPT** (read from their source / PR body / eval-bot comment) or
**DERIVED** (my arithmetic, marked). **Nothing here is measured on our vehicle.**
`score_claim: false`, `promotable: false`. **No launches, no Modal, no scorer runs. Spend $0.00.**
**This memo lands ZERO code.** It is an intake.

---

## ANSWER — the three transfers, ranked, and the one meta-finding

**1. PR133's byte-frozen coefficient re-solve is the highest-value thing in the entire wave, and
it is the only one confirmed by the maintainer's eval bot.** It moved the leaderboard
**0.172141 → 0.165780 (ΔS −0.006361, [contest-CUDA T4])**, and the author published his own
effort-matched control proving that **89.5 % of that (−0.0057) came from re-solving
already-transmitted integer coefficients against the exact forward PoseNet — at ZERO added
bytes.** Not a codec change. Not a retrain. A greedy coordinate search over codes that were
already in the archive. Consumer: our pose/`dxi` carrier and the pz4 lineage.

**2. PR138 opal's rank-one maximal-projector split** — the mechanism me1 stage-7 leg (iii) is
designing from the PR body alone. It is one scalar: model the binary "is the argmax wrong"
event, re-project keeping the complement's relative law exact. Worth −4,684 B on their vehicle
(**self-claimed, no eval-bot**). Full implementable detail in §2.

**3. The meta-finding, and it is the strongest cross-PR pattern in the wave:**
> **Every entry that moved its number closed the loop against the exact object that ships —
> with uint8 rounding inside the search loop. Every entry that did not, did not move.**

PR133 re-solves against the exact forward PoseNet and *real packaged bytes*. PR129 (the only
other eval-bot-confirmed row, 0.190503 CPU) puts an STE on the **container's own fp16 pack
grid**, so the train/pack gap is bit-zero. PR134 measured a **20× penalty** for rounding to
uint8 *after* the search instead of inside it. PR132 skipped the accept gate entirely and has no
measured outcome at all. This is our own optimal-form/realization law, independently re-derived
by four separate authors.

**We are not behind here.** Our own frontier is **181,161 B @ S 0.15853325034789678
[contest-CUDA T4 n600]** — smaller and lower than opal's *self-claimed* 182,040 B / 0.1591495384
and well under PR133's confirmed 0.165780. The value of this intake is mechanisms, not position.

---

## RELAY HEADER — for me1 stage-7 leg (iii), read these five first

1. **The mechanism is a rank-one projector split, and it is one scalar.** Take the base 5-class
   row, `maximal = argmax`, define the binary event `outcome = (symbol != maximal)`. Model only
   that. Re-project: `p[maximal] = 1−q`, `p[s≠maximal] = q · row[s]/wrong_mass`. The complement's
   *relative* law is preserved exactly; only the mass split moves. Receipt
   `rc64_backend.c:293-294`, `:138-147`, `:402`. **A 1-D correction to a 5-D law** — the
   minimum-parameter calibration that keeps the base model's learned shape, and it turns a 5-ary
   adaptive-context problem into a binary one, which is where all the cheap machinery lives.

2. **"55 causal context families" are NOT a taxonomy — they are 55 Cartesian index expressions**
   over ~12 primitive features, and the whole catalogue is 20 lines (`opal_model_impl.c:172-203`;
   production ids `rc64_backend.c:182-244`). Vocabulary: `conf` (101 bins of the base
   probability), `group` (190), `cell` (48), `prev`/`lag12`/`lag124`, `local_word` /
   `cross_word` / `temporal_word` / `causal_word` (256 each), their popcounts, `maximal_class`
   (5). 44 frame + 11 symbol-synchronous = 55. **Do not go looking for depth that is not there.**

3. **The scan order is engineered to make a non-causal template causal.** `group_of(y,x) =
   (x&63) + 2·(y&63)` (`opal_model_impl.c:72`); coding visits all 190 groups in order; a
   neighbour is legal iff its group is strictly earlier (`causal_at`, `:86-95`). Because `y` has
   weight 2 and `x` weight 1, an offset like `(dy=+1, dx=−4)` lands in an *earlier* group — so
   the template `causal_dx={-1,2,-1,-2,-2,-4,0,1}`, `causal_dy={0,-1,-1,0,-1,1,-1,-1}` (`:306-307`)
   legally reads pixels **below and to the right**, which a raster scan forbids. Cleverest thing
   in the submission, and free: choose the scan order to maximise the causal neighbourhood, then
   the template follows.

4. **The update law is a per-slot ridge-regularised online Newton step; the mixer is the same
   rule one level up.** Not counts, not CABAC shifts:
   ```c
   weight_j = clamp( -gradient[id_j] / (curvature[id_j] + 2.5), ±4.0 )      // rc64_backend.c:249-251
   q        = sigmoid( b + 0.22 · Σ_j multiplier_j · weight_j )             // :252-254
   gradient[id_j]  += (float)(q - outcome);                                 // :269  logistic grad
   curvature[id_j] += (float)(q · (1-q));                                   // :270  Bernoulli Fisher
   multiplier_j     = clamp(multiplier_j - g_j/(meta_curv_j + 0.75), ±8.0); // :275-277
   ```
   `curvature` is **exactly the Bernoulli Fisher information**. Each slot stores (Σ gradient,
   Σ Fisher); the weight is the Fisher-normalised Newton step. **This replaces the hand-tuned
   logistic-mixing learning rate of a classical CM coder with a self-scaling second-order step.**

5. ⚠ **Their encoder/decoder sync is structurally correct but numerically unguarded** — §4. They
   quantise *stored* state to float32 but let a 55-wide **double** meta-layer compound over
   117,964,800 symbols through `exp()`. **Build ours integer / fixed-point in the probability
   path from byte zero.** This is exactly the hazard our rr2 staging incident lived.

---

## STORES CONSULTED

- Existing intake, consumed not re-mined: `pr130_eureka_intake_acquisition_20260806.md`,
  `pr130_lift_wave_element_audit_20260806.md`, `ddm_eh1_20260806/`,
  `ddm_fd135_fractal_decomposition_20260810.md`, `ddm_pi135_pr135_intake_20260810.md`,
  `ddm_pi136_leaderboard_breadth_intake_20260810.md`,
  `pr135_pr133_direct_intake_facts_20260810.md`.
- Sibling arm reused for custody: `ddm_pq2_packet_polish_20260817.md` (PR137/138 archives +
  PR138 source already retained). Not re-downloaded.
- Our own measured cross-ref: `ddm_pz1` (shared-`D` resize; null-space field loses annihilation
  under the pose warp, 1.662× attenuation) — see §5 PR134.
- CLAUDE.md: NO-FAKE #7, rule-118 "inflate.py is a FREE interpreter", public-PR-intake
  pristine-clone discipline, ALWAYS KEEP THE PAYLOAD.

## ARCHIVE CUSTODY

Retained under `/Volumes/APDataStore/pact/ddm_hx1/` and (for 137/138) the sibling
`/Volumes/APDataStore/pact/ddm_pq2/intake/`. Nothing was cloned into the working tree.

| artifact | bytes | sha256 (prefix) |
|---|---:|---|
| `pr138/archive.zip` (opal_v1) | 182,040 | `bd9a47149b52a8f4986758e9274e509836bfa9c89f9b5cb069e90837eeb18400` (full) |
| `pr137/archive.zip` | 866,558 | `b396b26279c3…` |
| `pr133/archive.zip` | 190,212 | `051baf408f57…` ✅ matches PR claim |
| `pr132/archive.zip` | 191,028 | `77af18025dde…` ✅ matches PR claim |
| `pr138_full.diff` / `pr137_full.diff` | 220,187 / 121,147 | `58038fc69c9f…` / `a3ff1963255f…` |
| `opal_v1/runtime/entropy/opal_model_impl.c` | 33,819 | `b38958e1eb1a…` |
| `opal_v1/runtime/entropy/rc64_backend.c` | 22,179 | `b249b77bb06a…` |
| `opal_v1/runtime/entropy/coefficient_predictor.py` | 4,027 | `7a5717db1464…` |
| `pr133/.../carrier_codec.py` / `inflate.py` | 13,770 / 28,304 | `d2f14402374b…` / `335369c9b3b2…` |
| `pr133/.../verification.json` / `CREDITS.md` | 4,557 / 5,484 | `ac2a1bbead01…` / `6ca316673e3b…` |
| `pr136/src/codec_rc.py` / `codec.py` | 4,881 / 8,223 | `06dd2ffba5a3…` / `981e07291313…` |
| `pr129/pack_base.py` / `pr134/inflate.py` | 4,818 / 9,380 | `02c0f2ca7fd9…` / `77fab33f4492…` |

Full per-file manifests: `/Volumes/APDataStore/pact/ddm_hx1/notes/{opal_v1_source_manifest.txt,
pr136_notes.md, pr133_pr132_notes.md, lessons_sweep_notes.md}`.

## EVIDENCE GRADE — who was actually measured

This wave is mostly **unverified self-claims**, and that has to gate every value estimate below.

| PR | claim | eval-bot? | axis | state |
|---|---:|---|---|---|
| **133** cbq_matched8 | **0.165780** | **YES — maintainer eval, accepted** | **[contest-CUDA T4]** | merged/accepted |
| **129** qlp_exactgrid | **0.190503** | **YES** | **[contest-CPU]** | CLOSED |
| 138 opal_v1 | 0.1591495384 | **no** (only the auto-ack) | author-reported | OPEN |
| 136 hnerv_rc | 0.19258 | **no** | author-reported | CLOSED (LLM-policy comment) |
| 134 metricwarp_av1 | 0.93821 | **no** | author-reported | CLOSED (LLM-policy comment) |
| 137 metric_shift_av1 | 2.04 | **no** | author-reported | OPEN |
| 132 veigapunk_hpac_ft | — | **no** | none at all | CLOSED |
| 131 Coolchic | — | **no** | never exported an archive | CLOSED |

---

## 1. The technique table

Forms: **MECHANISM-ADOPT-ours** (our implementation, honest attribution) · **RACE** ·
**LESSON-ONLY** · **N-A**.

| # | mechanism | receipt | target | form | projected value (honest bars) |
|---|---|---|---|---|---|
| **T1** | **Byte-frozen coefficient re-solve against the exact scorer.** Greedy coordinate search over *already-transmitted* integer codes; exact forward PoseNet as accept oracle; step ladder [1,2,4,8,16,32], batch 8. **Zero added bytes.** | PR133 `CREDITS.md` steps 1-5; `verification.json.attribution_control_matched_effort` | our pose/`dxi` carrier; pz4 | **MECHANISM-ADOPT-ours** | **Their −0.0057, eval-bot confirmed, author-ablated to 89.5 % of their total move.** On ours: **−0.001 to −0.006, ±1 order of magnitude** — depends entirely on how far from optimal our shipped coefficients already are, **which we have never measured.** That measurement is cheap and is NEXT_IF_RESUMED #1. |
| **T2** | **Rank-one maximal-projector / complement split** (see relay header). | `rc64_backend.c:293-294`, `:138-147`, `:402` | me1 leg-iii; any 5-class law we code | **MECHANISM-ADOPT-ours** | Their −4.08 % of the token stream (self-claimed). Ours: **unknown, plausibly 1–5 %** of whatever section carries a 5-class law; **≈0 if our base law is already well-calibrated.** |
| **T3** | **Compensability-aware bit-depth pricing.** Price a shared parameter's bit depth *after* the free downstream channel absorbs the induced error, not before. | PR133 `CREDITS.md`; `carrier_codec.py:15,54-140` | pz4 (basis/scale vs per-frame coefficients) | **MECHANISM-ADOPT-ours** | Theirs: 828 B of 191 KB (0.43 % rate) at near-zero distortion cost. Ours: **unknown.** The transferable thing is the *pricing rule*, not their quantizer. |
| **T4** | **Scan order chosen to enlarge the causal neighbourhood.** | `opal_model_impl.c:72`, `:86-95`, `:306-307` | me1 leg-iii | **MECHANISM-ADOPT-ours** | Free (a permutation). **Unknown; order 0.5–2 %** of a context-coded plane. |
| **T5** | **Fisher-normalised online Newton per context slot**, same rule for the family mixer. | `rc64_backend.c:249-251`, `:262-277` | me1 leg-iii | **MECHANISM-ADOPT-ours** | Mostly *robustness*, not bits. **≤1 % vs a well-tuned SGD mixer**, but removes a tuning axis. |
| **T6** | **`min(brotli, adaptive-RC)` per stream, decided at encode time, 1 flag byte.** | PR136 `codec.py:4-8,12,114` (the absence of it) | packet | **MECHANISM-ADOPT-ours** | PR136's own two stated numbers imply they **lost ~400 B** by applying RC to a run-heavy plane where LZ77 wins. Order **10²–10³ B** on a mixed archive, wide bars. Cheap insurance. |
| **T7** | **Per-tensor context reset + never transmit the frequency table.** | PR136 `codec_rc.py:28-36`, `:68-69` | our coders | **MECHANISM-ADOPT-ours** | ~40 lines. Kills table overhead. **Unknown**, depends on our stream count / per-tensor spread. |
| **T8** | **STE targets the CONTAINER's grid, not a nominal quantizer.** Latents packed with per-dim **fp16** min/scale; the training STE replicates that fp16 grid bit-for-bit → zero train/pack gap. | PR129 `pack_base.py:88-92` | all our QAT surfaces | **MECHANISM-ADOPT-ours** | **Eval-bot-confirmed lineage (0.190503).** Independent confirmation of a law we hold. Value: eliminates a silent gap; **unknown in bytes, but it is a correctness fix, not a tuning knob.** |
| **T9** | **Boundary/margin seg loss** `sigmoid(−margin/τ)` (smooth argmax-flip fraction, τ annealed) optimised **through the exact inflate chain**. | PR129 seg loss | our codim-1 margin lever | **LESSON-ONLY (confirmatory)** | Independent eval-confirmed evidence that **the codim-1 margin loss is what buys the last seg decimal at the frontier** — our own §levers item. Not new; now externally corroborated. |
| **T10** | **Integer fixed-point AR(1) temporal predictor**: Q8 factors, round-half-away-from-zero in pure int64, reconstruction in a signed-int12 **modular** domain, **3 B metadata per dimension**. | `coefficient_predictor.py:24-31,19-22,118-127` | our token/coefficient path | **MECHANISM-ADOPT-ours** | Byte-exact temporal prediction at ~nil metadata. **Unknown; honest prior "small but nearly free."** Notable as the *contrast case* to §4. |
| **T11** | **Spacetime "holonomy" context** = `causal_word XOR causal_prev_word` — condition on *change*, not state. | `rc64_backend.c:175-176`, ids `:236,:243` | me1 leg-iii | **MECHANISM-ADOPT-ours** | Concentrates statistics on a near-static field. **Unknown; order 0.5–2 %.** One XOR + popcount. |
| T12 | **Symmetry-orbit quotienting of the context alphabet** (D4/C4/necklace orbit rep, 256-entry LUT) — collapses contexts ~8×. | `opal_model_impl.c:99-170` | our neighbourhood contexts | **RACE** | Attacks context dilution for free — **but dashcam statistics are not rotation-symmetric** (road/sky break vertical symmetry), which is presumably why they swept it and shipped it only for the *temporal* word (`rc64_backend.c:174`). Race it; do not assume it. |
| T13 | **Per-channel variable bit-width via a nibble side-table**; `bits=0` prunes the channel. | PR133 `integer_model_io.py:15-92` | adaptive-quantization toolbox | **RACE** | Unknown. |
| T14 | **Trade archive bytes for decode compute: 49.4 MB of adaptive state, ZERO archive bytes**, regenerated from the decoded prefix. | opal README; `sectors·8 = 6,175,440·8 = 49,403,520` (`opal_model_impl.c:650,:201`) | strategic | **LESSON-ONLY (confirmatory)** | **Our own rule-118 doctrine, independently discovered by a competitor and cashed for −4,684 B.** The score has no time term. Validates the axis; adds none. |
| T15 | **Two-stage design: 25-argv offline mode sweep → freeze winners as literals in the shipped decoder**, production backend `#include`s the sweep harness so the model is one source of truth. | `rc64_backend.c:1-3`; modes `opal_model_impl.c:229-276`; frozen constants `:127,145,250,254,276` | our DSL-as-SSoT | **LESSON-ONLY** | Convergent validation of triality: "the thing I swept is bit-identically the thing I ship." They do it by textual inclusion; we by DSL compile. |
| T16 | **Generated priors selected by a 2-bit index** (4 hard-coded geometric shapes; archive stores the index, not the table). | `adaptive_ans.py:23-33`; `renderer_weight_codec.py:16-19` | packet | **LESSON-ONLY (confirmatory)** | Rule-118 again — their comment literally narrates moving a byte prefix out of the counted payload into the free schema. |
| T17 | Adaptive order-0 rANS, count-halving rescale, stable-sort tie-break. | `adaptive_ans.py:44-72` | — | **N-A** | Textbook. **Anti-pattern worth naming:** it re-normalises with a numpy sort **per symbol** — fine for a small stream, 100–1000× too slow for a large one. |
| T18 | Combinatorial unranking of a frame subset to one integer. | `frame0_selector.py` | — | **N-A (known)** | Same colex-rank trick as PR101 L31, already in our lesson set. |
| T19 | PR132's fine-tune. | — | — | **N-A** | No measured outcome, no shipped loss, no ablation. §6. |

---

## 2. PR138 opal — where it sits, and what is actually there

| | archive B | exact S | axis |
|---|---:|---:|---|
| PR135 (their parent) | 186,724 | 0.1622684217 | author-reported |
| **PR138 opal_v1** | **182,040** | **0.1591495384** | **author-reported, NOT eval-bot confirmed** |
| **ours** | **181,161** | **0.15853325034789678** | **[contest-CUDA T4 n600]** |

Their arithmetic is internally consistent — DERIVED:
`100(0.00029639) + √(10·0.00000688) + 25(182040/37545489) = 0.029639 + 0.008295 + 0.121213
= 0.159147` vs their stated `0.1591495384`. Consistent to reported precision.

Their whole delta is **−4,684 B of pure entropy coding on a frozen token stream** — distortion is
bit-identical to PR135 by construction (they decode the same 117,964,800 tokens). A clean,
isolated, attributable rate win, which makes it unusually easy to reason about. Mechanisms T2,
T4, T5, T11, T12, T14, T15 above.

## 3. PR133 — the confirmed mover, and its author deflated his own headline

**What `cbq`/`matched8` actually are — not what the name suggests.** `cbq` =
*Compensability-aware Basis Quantization*. `matched8` = the **effort-MATCHED control** at **8**
full-600 coefficient-search passes. **Not 8-bit, not 8 levels, not a matched filter.**

The CPR1 pose carrier (`inflate.py:33-37,601-643`): the PoseNet-scored slave frame is
`127.5 + 64·(coeff @ basis)/√12` — neutral grey plus a **rank-12 linear image basis** (12 atoms ×
3ch × 24×32, bicubic-upsampled, per-atom zero-mean + RMS-normalised). Basis = 27,648 signed 5-bit
codes → zigzag → **one global canonical Huffman over 32 symbols, no context**
(`carrier_codec.py:15,54-140`). Coefficients = 600×12 signed int12, delta-along-time + zigzag
(`inflate.py:265-270`), then per-dimension Rice with `k` by exhaustive argmin of actual bit count
(`carrier_codec.py:221-224`).

**Quantizer is plain uniform** — int code × one fp32 per-atom scale. Grep across all five modules
for `lloyd|kmeans|codebook|trellis|tcq|deadzone` → **0 hits.** No Lloyd-Max, no dead zone, no
non-uniform spacing, no TCQ, no soft assignment. **RD co-design: NO** in the Lagrangian sense;
**YES** in the accept loop — rate enters as *real packaged bytes* and distortion is measured
*after* compensation.

**The decomposition, from the author's own published control** (`verification.json`
`attribution_control_matched_effort`, with four explicit `false`s in `claim_boundary`):

| component | ΔS | share | bytes |
|---|---:|---:|---|
| pose | −0.005802 | 91.2 % | 0 |
| rate | −0.000559 | 8.8 % | −828 |
| seg | 0 | 0 | — |
| **total 0.172141 → 0.165780** | **−0.006361** | | |
| ↳ CBQ's honest share | ≈ −0.00067 | **10.5 %** | −828 |
| ↳ **coefficient re-solve** | **≈ −0.0057** | **89.5 %** | **0** |

Neither branch converged. **Publishing the ablation that deflates your own headline is rare and
it is why the 2.55 % number is trustworthy.** T1 and T3 come from here.

## 4. State-sync — the hardest engineering in the class, half-solved

Read twice; this is where the class kills you.

**Right, structurally.** Encoder and decoder call *literally the same two functions* —
`opal_prepare()` then `opal_update()` — in the same order from the same source
(`rc64_backend.c:396-403` decoder, `:496-502` encoder). `opal_update` runs **after** the symbol
is coded, so the decoder holds identical information at identical points. No adaptive table
ships. Textbook backward-adaptive design.

**The guard they DID build, and it is clever and cheap:**
```c
b = (double)(float)log(wrong_mass / (1.0 - wrong_mass));   // rc64_backend.c:143
```
The `log` result is **rounded through float32 before use** — discarding ~29 mantissa bits, so any
libm agreeing to better than half a float32 ULP yields the *same* `b`. You do not need a
bit-identical `log`, only an agreeing one. Same trick guards stored state: `gradient`/`curvature`
are `float` arrays, updated `+= (float)(…)` (`:269-270`), re-quantising every step.

**The hole.** The meta-layer is unguarded double and it compounds:
```c
f->meta_gradient  += gradient;                                                   // :273  double
f->meta_curvature += curvature;                                                  // :274  double
f->multiplier = clamp(f->multiplier - gradient/(f->meta_curvature + 0.75), ±8.0);// :275-277
```
55 `double` accumulators + 55 `double` multipliers evolve across **117,964,800 symbols** with no
re-quantisation, each a function of `q = sigmoid(…)` — i.e. of `exp()`. `exp` is not required to
be correctly-rounded and is **not bit-identical across libm implementations or versions**.
`inflate.sh:19-22` compiles the backend with `${CC:-cc} … -lm` on the *evaluation* machine while
the encoder ran on the author's. One 1-ULP divergence propagates into `multiplier`, changes every
subsequent `q`, changes `(uint64_t)(value·2³¹)` in `opal_adjust_frequencies` (`:300`), and
**desynchronises the arithmetic decoder catastrophically** — not gracefully; everything after is
garbage.

They are almost certainly fine in practice on x86-64 glibc, and the decoded-token SHA-256 in
their README is their check. But the design is **not proven bit-exact** and is one libm change
from total decode failure.

**PR136 shows the same class, safe by accident.** Its docstring (`codec_rc.py:6-8`) claims
"identical **integer** count tables"; the code is float64. It is safe only because every count is
exactly `1.0 + 8k ≪ 2⁵³`, so the sum is exact under any summation order. Any non-integral
`PRIOR`/`INC`, any multiplicative decay, or counts past 2⁵³ silently desyncs the decoder.

> **LAW for me1 leg-iii:** in a backward-adaptive coder, quantise **every** value that enters the
> probability path — not only the stored state. The cheapest correct design keeps the entire
> probability path in integers / fixed-point; if a transcendental is unavoidable, evaluate it
> through a **shared deterministic LUT compiled from the same table on both sides**, never libm.
> Their own `coefficient_predictor.py` (T10) proves the team knows how — pure int64, explicit
> rounding rule, modular domain. The C path simply did not get the same discipline.

## 5. Decode wall-clock vs the 30-minute budget — DERIVED, not measured

No runs permitted, so this is arithmetic with wide bars. Per symbol the model does ~165 double
divisions (55 in `opal_prepare`, 55 **recomputed** in `opal_update`, 55 meta) and ~110 scattered
accesses into 49.4 MB. Over 117,964,800 symbols: **≈1.95 × 10¹⁰ divisions**, **≈1.3 × 10¹⁰ random
accesses** into a working set far larger than any L3.

- Divisions alone at ~4-cycle `divsd` throughput, 3 GHz: **≈26 s** (best case).
- Memory dominates: ~6.5 × 10⁹ likely-missing accesses at 8–40 ns → **≈1–4 min**, worse if
  memory-level parallelism is poor.
- ~2.4 × 10⁸ `exp()` calls ≈ 5 s.

**Estimate: 2–15 min single-threaded, most likely 4–8, DRAM-latency dominated** — against a
30-min budget that must *also* run the inherited F26 CUDA renderer over 600 frames (their README
concedes inflation still needs CUDA). Tight, probably feasible, and a real risk they carry
silently.

**The ablation they never ran, and the one I would run first:** 117.96 M symbols over 6.17 M slots
is **~19 observations per slot on average**, and the large families (`CELLS·GROUPS·256` =
2,334,720 slots) sit far below that. **Most of those contexts are starved.** A far smaller
context set would likely capture most of the gain. **If we adopt T2+T5, design the state to fit
in L3 (a few MB), not 49 MB.**

## 6. Falsifier honesty — what was empty

- **PR132 veigapunk_hpac_ft — EMPTY, closed honestly.** "HPAC" is a misnomer: the HPAC model
  bytes are **byte-identical to CPR1**. What changed is an 800-step AdamW fine-tune of the int4
  semantic renderer (`README.md` L3, `meta.json steps:800`). The claimed SegNet-CE loss is an
  **assertion — no training code ships and no loss line exists in the artifact.** The ft is real
  (semantic blob differs in 8,362/40,252 bytes = 20.77 %), but **this archive was never evaluated
  by anyone**; the only measured delta is a −24 B lossless repack = −0.0000160. **Do not cite any
  PR132 number.** Its one lesson is *negative*: quantize → ft → re-quantize → repack with **no
  compensation loop and no accept gate** — precisely the shape PR133 in the same lineage found
  fails ("direct quantization alone looked promising on a small screen, but it broke on larger
  batches").
- **PR131 Coolchic — EMPTY.** PR body is the unfilled template; no eval ever ran. Not a faithful
  contest port: stock Cool-Chic 5.0 (MSE-trained) run out-of-tree in a separate conda env, scored
  by an offline extrapolator that codes N frames and multiplies rate to 1200. **It never exports
  an `archive.zip` and has no inflate path** — research-only by construction, so it neither beats
  nor loses to AV1; it was never measured. One suggestive datum: `nn_bpp` (0.0115) is **4.9×**
  `latent_bpp` (0.0023) — at contest rates stock Cool-Chic spends most of its budget on
  synthesis/ARM weights, not the overfit latent. Suggestive, not a verdict. **The non-HNeRV
  neural lane is untested, not falsified.**
- **PR137 metric_shift_av1 — EXPECTED.** The "metric shift" is **(b) a colourspace/normalisation
  shift**: a per-frame quantised *mean-luma* correction applied after the resize back to camera
  resolution (`README.md:39-42`; `generate_sidechannel.py:570-573`; `inflate.py:177-182,501-515`).
  **Not** blind-spot targeting, **not** a bitrate ladder. Pose is ~2,700× worse than PR129's; the
  luma channel never touches the geometry the scorer reads. Nothing to take.
- **PR136 is largely off-the-shelf.** Stock `constriction` RangeEncoder/Decoder + stock
  `Categorical`; the novel surface is ~40 lines. The model is textbook Witten–Neal–Cleary 1987
  adaptive arithmetic coding, order-0, reset per tensor — **tensor identity is the entire context
  model.** Honest framing: a correct, minimal, well-scoped application of a standard coder, not a
  new coder. Its transferable content is T6/T7 and a lesson, nothing more.
- **opal's "55 causal context families" are shallower than the phrase suggests** (relay item 2),
  and **the "probability transport law" is T2's re-projection plus T5's Newton update** — not a
  separate mechanism. Naming it a "transport law" is presentation.
- **No decode wall-clock, no family-set ablation, no context-count ablation anywhere in PR138.**
  The three questions I most wanted answered are unanswered.

## 7. Defects found (recorded, not exploited)

- **PR138 `compress.sh` calls a file that does not exist**: `:26` runs
  `python3 "$HERE/verify_submission.py"`, absent from the 28-file submitted tree. Their
  compression script cannot run as shipped.
- **PR138 `compress.sh` does not compress** — it copies a frozen `archive.zip` and checks its
  hash; never reads the video, never trains, never runs their encoder. Independently confirms
  ddm_pq2's finding that our end-to-end rebuild chain is stronger than the field norm.
- **PR136 argues against itself in-tree**: `codec.py:56` celebrates *removing* the `constriction`
  dependency; `codec_rc.py` silently reintroduces it with **no install path** (`inflate.sh`
  installs nothing) — the PR106 missing-brotli replay-failure class. And `compress.sh:10` /
  `inflate.py:9` both point at `submissions.hnerv_muon`, not `hnerv_rc`: **the documented
  reproduce path does not run this PR's own code.**
- PR138 body says "182,020 bytes" in one sentence and 182,040 in the table/metadata. Typo;
  182,040 matches the archive sha.

## 8. Cross-PR structure worth knowing

1. **Clean split by family.** Classical AV1 sits at 0.94 / 2.04 (self-claimed); the HNeRV/CPR1
   neural lineage at 0.166–0.193. **Nothing classical is within 4×.**
2. **The CPR1 lineage is not competing on rate.** In PR133/PR132 the **token stream — 61.5 % of
   the payload, 116,980 B — is byte-identical and untouched by both.** The whole 0.172 → 0.166
   move came from re-solving a 22 KB pose carrier. And **seg is frozen at 0.00029660 across all
   three official archives — nobody in that lineage has moved the seg term at all.**
3. **The convergent lesson (the §ANSWER meta-finding).** Put the real scorer, with uint8 rounding,
   *inside* the search loop, and make the training quantizer bit-identical to the shipping
   container. PR129 via an STE on the fp16 pack grid; PR133 via exact-PoseNet accept on real
   packaged bytes; PR134 measured a **20×** penalty for rounding after instead of inside. Every
   entry that moved its number did this; none that skipped it did.
4. **PR134's constructive null-space result cross-refs our own `ddm_pz1`.** Both scorers resize
   1164×874 → 512×384 bilinear, `align_corners=False`; stride 1164/512 ≈ 2.273 > 2 makes the 2×2
   tap blocks **pairwise disjoint**, so writing one value into all four taps sets the net's input
   pixel *exactly* — resample loss zero, and **~23 % of full-res pixels are provably
   zero-weight** (`inflate.py:21-27`, `:89-99`). We measured a `D`-null-space field *losing*
   annihilation when the pose warp resamples it to a different lattice (1.662× attenuation);
   **PR134 sidesteps exactly that by warping in the 512×384 chart and grid-placing afterwards, so
   the lattice never changes.** That is a real, cheap answer to a wall we measured.

## 9. Borrowed-substrate accounting (NO-FAKE #7)

**This memo adopts nothing and lands zero code.** For each MECHANISM-ADOPT-ours row, the
accounting that must accompany any future landing:

- **T1, T3, T13** — ideas read from `JasonMo123`'s PR #133 (cbq_matched8) and its inherited CPR1
  lineage. **T2, T4, T5, T10, T11** — from `ccastillo1043`'s PR #138 (opal_v1) and its inherited
  CPR1/F26 lineage (PR #135, `codexblack`). **T6, T7** — from `JPL11`'s PR #136. **T8, T9** — from
  `ryanli0070`'s PR #129.
- Any landing must (a) be **our own implementation** against our own carrier and our own DSL,
  (b) carry an attribution line naming the PR number and the specific `file:line` above, (c) state
  plainly that every delta quoted here is **theirs on their vehicle**, never ours, until we
  measure our own.
- **T12** is RACE, not adopt: no code moves until a matched A/B says the orbit quotient helps on
  *our* statistics.
- **T9, T14, T15, T16** are LESSON-ONLY and confirmatory of doctrine we already hold; they add no
  borrowed substrate.
- **Copying opal's C is not on the table.** The bit-exactness hole in §4 is precisely the thing we
  should not inherit.

---

## NEXT_IF_RESUMED

1. **Measure T1's headroom before building anything.** How far from optimal are our *already
   shipped* pose/`dxi` coefficients? A greedy coordinate search over the existing integer codes,
   exact scorer as accept oracle, uint8 rounding inside the loop — **zero added bytes by
   construction.** This is the highest expected value in the wave and the cheapest thing to test.
2. **Ablate T2 offline, bits-only, before any coder work.** Compute `Σ −log2 p` under our current
   5-class law vs a rank-one-projector-corrected law with a **small** context set (e.g.
   `conf × cell × prev`, ~10 k slots, not 6.17 M — see §5). Pure numpy bit-accounting against
   retained tokens; no scorer, no launch. Answers "is our base law mis-calibrated enough to pay?"
3. **If (2) shows ≥1 %, build the integer / fixed-point probability path FIRST** (§4 LAW), then
   the coder. Race T11 (holonomy) and T12 (orbit quotient) as context-set variants inside (2)'s
   harness.
4. **Audit our QAT surfaces against T8**: does every STE target the *container's* grid (fp16
   min/scale as packed) rather than a nominal quantizer? PR129 got an eval-bot-confirmed row
   partly on this.
5. **T6 is cheap insurance**: add a per-stream `min(brotli, RC)` selector with a 1-flag byte.
