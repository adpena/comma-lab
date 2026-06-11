# GPU-ONLY INFLATE — is there a theoretical GPU candidate superior to the CPU path?

UTC 2026-06-11 · claude · **THOUGHT EXPERIMENT / RESEARCH** (operator-framed, 2026-06-11). Analysis +
derivation only. `[macOS-CPU advisory]` reasoning surface; `promotable=false`, `score_claim=false`, no
dispatch, no daemons touched, no MLX-scorer files touched, no /tmp, $0. Every byte/score number is
`[predicted]`/`[DERIVED]` with the mechanism shown (NO-FAKE); none is a measured exact-eval row. The CPU
inflate path remains the practical leaderboard route — this explores the 10-year class-shift edge of the
design space, NOT a claim the CPU path is wrong.

    S = 100·d_seg + √(10·d_pose) + 25·B/D        D = 37,545,489   (frozen authority, evaluate.py:92)

---

## 0. TL;DR — the gate, the verdict, the one novel idea

- **The strategic gate (HARD, quantified).** The contest leaderboard is the **CPU axis**
  (`README.md`: *"Final ranking will be based on the public leaderboard"*; the leaderboard is the CPU
  eval). `evaluate.sh` runs inflate AND `evaluate.py --device $DEVICE` **on the same instance with the
  same device flag** — so a **GPU-required inflate forces `--device cuda`**, which carries the measured,
  near-constant **+0.0330 CUDA−CPU tax** (drift-ladder RUNG C: pose 70% / seg 30%, σ≈4e-4, HNeRV-class).
  A GPU-only candidate therefore starts **+0.033 behind** the best CPU candidate **on score**, before it
  has done anything.
- **The byte threshold to overcome the tax `[DERIVED]`.** +0.033 score = **49,560 archive bytes** on the
  rate axis (= 0.033·D/25). To *tie* the current ~177 KB CPU frontier on score, a GPU candidate must get
  its archive **below ~128 KB** purely to pay the tax; to *beat* it meaningfully it must go well below
  that. That is achievable in principle (a 20-KB seed beats the CPU frontier by −0.072 net even after the
  tax) — **but only if the shrink is real.**
- **The decisive subtlety that collapses most of the case.** **The rate saving is device-INDEPENDENT.**
  `rate = compressed_size/uncompressed_size` is read from `archive.zip`'s file size (evaluate.py:63-65),
  not from inflate compute. So *any archive-shrinking decode that is CPU-feasible in 30 min should just
  run on CPU* — capturing the identical rate saving AND avoiding the +0.033 tax. **The GPU is uniquely
  valuable ONLY in the narrow regime where the shrinking decode is computationally INFEASIBLE on 4 CPUs
  in 30 min but feasible on a T4 in 30 min** (a ~16–40× compute wall, ≈ 15–20 CPU-hours of work).
- **The verdict.** A *theoretically plausible* GPU-superior candidate **exists** (the
  GPU-uniquely-feasible massive-decode regime, §3) — but it is **dominated for the leaderboard** by the
  honest reality that (a) our decode is seconds-long, so CPU has ~1800 s of unused budget and the GPU's
  feasibility advantage is moot until a decode is ~30× too slow for CPU, and (b) the +0.033 tax means a
  GPU candidate must be ~0.033 *better on its own axis* just to tie. **The right move is to repurpose the
  GPU's power at COMPRESS time** (where we already have unlimited GPU and no tax) to build the smaller
  CPU-feasible amortizer, not to move compute to inflate. **Pursue: COMPRESS-time GPU; DEFER: GPU-only
  inflate** (reactivation criterion in §5).
- **The single most-novel GPU-inflate idea (§3.2):** the **GPU-as-expensive-decompressor running a
  massively-parallel preimage/fixed-point solve** that reconstructs the witness from a Kolmogorov-tiny
  program — legal *only* if the equivalent CPU decode genuinely cannot finish in 30 min. It is the
  literal "weird Turing on its head": tiny program (archive) + huge GPU computation → the witness. It is
  the most interesting idea and the one with a real (if narrow) door.

---

## 1. THE GATE, FORMALIZED (no hand-waving)

### 1.1 Where the device enters the score — and where it does NOT

```
evaluate.sh:  bash inflate.sh archive/ inflated/ names   # produces 1200 .raw frames
              python evaluate.py --device $DEVICE         # scores those frames
```
Three score terms, three device dependencies:

| term | formula | depends on inflate device? | depends on eval device? |
|---|---|---|---|
| rate | `25·compressed_size/D` | **NO** (reads `archive.zip` file size, evaluate.py:63) | **NO** |
| d_seg | `100·mean(argmax flip)` | only via the .raw frames it produced | **YES** (SegNet kernel numerics) |
| d_pose | `√(10·mean(pose MSE))` | only via the .raw frames | **YES** (PoseNet kernel numerics) |

The crucial structural facts:
1. **`inflate.sh` and `evaluate.py` share one `$DEVICE`.** You cannot inflate on GPU and score on CPU in
   the official harness — the same flag drives both. A GPU-required inflate ⇒ `--device cuda` scoring.
2. **rate is device-free.** Whatever decode produces the archive, the bytes are the bytes. A small
   archive scores the same low rate on CPU or CUDA.

### 1.2 The +0.0330 CUDA tax (drift-ladder RUNG C, MEASURED)

From `local_to_contest_scorer_drift_ladder_and_correction_20260611.md` §1 (5 paired contest-CI bot
comments, the contest's own `--device cuda`/`--device cpu` on identical archives):

```
Δscore(CUDA − CPU) = +0.0330 ± 0.0004   (R_pose=5.04, R_seg=1.17; pose 70% / seg 30%; rate 0%)
```
This is a **structural property of the scorer** (additive fp-precision noise amplified through the
PoseNet regression head, √-softened in the score; SegNet argmax is stable), near-constant across the
HNeRV medal-band cluster. **A GPU-only candidate is scored on CUDA, so it pays this tax in full.**

### 1.3 The byte threshold `[DERIVED]`

Rate axis: 1 score point = `D/25 = 1,501,820` bytes. So:

| ΔS to offset | archive bytes |
|---|---:|
| **+0.0330 (the full tax)** | **49,560 B** |
| +0.0100 | 15,018 B |
| +0.0500 | 75,091 B |

Current CPU frontier archive ≈ **177,169 B** (rate term 0.11797 = 100% of the measured S_floor; the
frontier is RATE-bound). For a GPU candidate scored on CUDA to **tie** the CPU frontier *on final score*:

```
break-even seed = 177,169 − 0.033·D/25 = 127,609 B   ⇒ archive must drop below ~128 KB just to TIE.
```

Beyond break-even it wins net (rate saving > tax):

| GPU seed archive | rate Δ | +tax | NET vs CPU frontier | verdict |
|---:|---:|---:|---:|---|
| 50,000 B | −0.0847 | +0.033 | **−0.0517** | beats CPU |
| 30,000 B | −0.0980 | +0.033 | **−0.0650** | beats CPU |
| 20,000 B | −0.1047 | +0.033 | **−0.0717** | beats CPU |
| 10,000 B | −0.1113 | +0.033 | **−0.0783** | beats CPU |

**So the gate is passable on paper:** a ~20-KB seed scored on CUDA beats the 177-KB CPU frontier by
−0.072 net even after the +0.033 tax. The whole question reduces to: **can a ~20–50 KB archive
reconstruct the witness — and does it NEED a GPU to do so?**

---

## 2. WHY THE GPU IS (ALMOST ALWAYS) UNNECESSARY — the device-independence collapse

The seductive framing is "GPU does powerful things CPU can't." But the score only rewards **bytes** (and
distortion). The byte saving from any clever decode is **device-independent** (§1.1). Therefore:

> **If a ~20-KB-archive decode is feasible on 4 CPUs in 30 min, run it on CPU — you get the identical
> −0.072 rate win AND you dodge the +0.033 tax, netting an extra +0.033 over the GPU version.**

The capstone inflate is a **single numpy forward measured in seconds** (sibling memo §0). We are using a
*tiny fraction* of the 1,800 s × 4-core CPU budget. So for essentially every byte-shrinking lever in the
sibling "inflate-compute as free rate lever" menu — iterative/unrolled decode (L1), scorer-free TTO (L2),
super-res synthesis (L3), heavier arithmetic coding (L4), even a modest generative micro-prior (L5) — the
CPU has *orders of magnitude* of headroom. **None of them needs a GPU.** The CPU path captures the rate
win tax-free.

**Conclusion:** the GPU is uniquely valuable **only** when the byte-shrinking decode is *computationally
infeasible on CPU within 30 min* but feasible on a T4. That is the entire remaining theoretical territory.

### The compute wall that makes the GPU necessary `[DERIVED, order-of-magnitude]`

- T4 ≈ 8.1 TFLOP fp32; 4 modern CPU cores ≈ 0.2–0.5 TFLOP fp32 effective ⇒ **~16–40× raw throughput**,
  plus the GPU's far higher parallel width for independent/batched work.
- CPU budget ≈ 1,800 s. To make the GPU *necessary*, the decode must cost **> 1,800 CPU-s** in its
  CPU-feasible form, i.e. the byte-shrinking computation must be **~30–40× too slow for CPU** ⇒ the
  decode is ≈ **15–20 CPU-hours** of work that a T4 compresses into < 30 min.

What genuinely needs 15–20 CPU-hours but ~25 GPU-min and shrinks the archive?
- a **many-step, full-resolution, all-1200-frame iterative fixed-point or generative refinement** whose
  per-step cost × steps × frames exceeds the CPU wall;
- a **massively-parallel combinatorial preimage/search solve** (find the tiny program/latent whose decode
  lands in the evaluator cell) that is embarrassingly parallel (GPU-friendly) but serial-bound on CPU.

These are real candidates — §3 examines them against the gate.

---

## 3. THE THEORETICAL GPU-SUPERIOR CANDIDATES (the territory the gate leaves open)

### 3.1 Training-at-inflate (gradient descent in 30 GPU-min from a tiny seed)

**Idea.** Store a few-KB seed (coarse latent + a small fixed prior). At inflate, run GPU gradient descent
to *overfit* an implicit representation (INR/NeRV) of the witness, descending a **deterministic
scorer-free** objective (the no-scorer rule forbids SegNet/PoseNet at inflate — CLAUDE.md "Strict scorer
rule"). Many GPU steps "finish the fit" the small seed under-specifies.

**Does it pass the gate?**
- *Legal objective:* must be a **stored deterministic target** — temporal self-consistency (warp
  frame_{2k+1}→frame_{2k} via stored pose), reconstruction self-consistency against a stored low-rank
  reference, edge/structure preservation. **NEVER** "descend toward what the scorer wants." This is the
  sharp compliance edge.
- *GPU-necessary?* Only if the descent genuinely can't finish on CPU in 30 min. A few-K-param INR overfit
  to 1200 frames for, say, 2,000 steps **is plausibly CPU-feasible** (it's small) ⇒ **GPU not needed ⇒
  run on CPU, dodge the tax.** GPU becomes necessary only at a large prior / full-res / many-thousand-step
  scale — which then risks the 30-min *GPU* wall too (1200 frames × many steps × full-res FLOPs).
- *Kolmogorov framing:* yes — this is "tiny program, huge computation → witness." But the deterministic
  objective can only recover information the **stored statistic constrains**; you cannot conjure d_seg/
  d_pose-correct frames from a seed that doesn't encode them. The seed must already carry the
  cell-landing information; descent only *amortizes its decompression*. So the achievable seed size is
  bounded by the witness's true conditional entropy given the prior — the same floor the CPU faces.

**Verdict:** legal and interesting, but **rarely GPU-necessary** (the small-seed regime is CPU-feasible)
and **always tax-burdened** when it is. The information bound is device-blind. **DEFER.**

### 3.2 ⭐ The GPU-as-expensive-decompressor: massively-parallel preimage/fixed-point solve (MOST NOVEL)

**Idea — "weird Turing on its head."** The archive stores a **Kolmogorov-tiny program / spec** (a few KB:
a generator's seed + a deterministic fixed-point operator + a target invariant). The inflate runs an
algorithm whose **output is the witness** but whose **computation is GPU-only-feasible in 30 min**:
- a **large iterative fixed-point** `x_{n+1} = F(x_n; seed)` run to convergence over all 1200 frames at
  full res (a neural-ODE / deep-equilibrium / Anderson-accelerated solve) where F is cheap but the
  iteration count × frames is the wall;
- a **massively-parallel search/solve** over a huge candidate space for the per-frame state that satisfies
  a stored deterministic invariant (cellular-automaton-style generation, parallel preimage of a stored
  hash-like constraint, simultaneous over millions of pixels/candidates);
- a **deterministic multi-step generative denoise** (sibling L5) at a scale that is feasible on T4 but
  blows the CPU wall.

**Why it is the genuinely novel door.** It is the *only* mechanism where the GPU's **parallel-compute
capability — not its bytes** — does something a CPU **provably cannot finish in 30 min**, mapping huge
free GPU computation onto a tiny archive. It is the closest thing to "training at inflate / weird Turing"
that survives the rate-is-device-free collapse: it lives *exactly* in the §2 GPU-necessary regime by
construction.

**Does it pass the gate? (the four hard tests)**
1. **Tax-survivable?** Yes *if* the program is < ~50 KB → rate saving > tax (§1.3). A few-KB spec easily
   clears this. ✓ (on paper)
2. **GPU-necessary (the whole point)?** This is the **design constraint, not a side-effect**: the solve
   must be one whose CPU form exceeds 1,800 s. Embarrassingly-parallel preimage / wide fixed-point fits
   (16–40× GPU advantage). **The risk is the inverse:** if the solve is *also* tight on the **T4's** 30
   min for 1200 full-res frames, it fails the budget on GPU too. Feasibility is a two-sided HARD gate.
3. **Scorer-free + deterministic?** Required — fixed seed + fixed schedule/operator ⇒ bit-reproducible.
   The invariant the solve targets must be a **stored deterministic target**, never the scorers. ✓ with
   care.
4. **Lands in the cell?** The OPEN question (Kolmogorov-uncomputable lower bound): does *any* few-KB
   program decode to a witness inside the SegNet-argmax × PoseNet-tube cell at near-zero distortion? The
   floor report explicitly leaves "does any amortizer beat 177 KB" UNRESOLVED. This is the real risk,
   and it is **identical to the CPU class-shift risk** — the GPU does not make a smaller amortizer
   *exist*, it only changes which decodes are *runnable*.

**Verdict:** the most novel and the only theoretically-defensible GPU-superior candidate. But its
*existence* hinges on the same open compression question the CPU faces, and its GPU-necessity must be
*engineered in* (and not break the T4 budget). **DEFER with a sharp reactivation criterion (§5).**

### 3.3 Honest non-starters (catalogued so we don't relitigate)

- **"GPU is faster so the score is better."** False — speed buys nothing the score rewards; rate is
  byte-only and distortion is device-numerics-only (the tax direction is *against* GPU).
- **"Run the existing decode on GPU."** Pure loss: identical bytes, identical-or-worse frames, +0.033 tax.
- **"GPU inflate, then re-score on CPU."** Not available in the harness — `evaluate.sh` couples one
  `$DEVICE` to both inflate and eval (§1.1). A GPU-required inflate cannot be CPU-scored officially.
- **"Inflate-time scorer-guided optimization."** FORBIDDEN (no SegNet/PoseNet at inflate). Any
  inflate-time objective must be a deterministic stored target.

---

## 4. THE NET CALCULUS — when (if ever) GPU-inflate wins

A GPU-only candidate beats the CPU frontier `S_cpu*` iff:
```
  S_gpu_cuda  =  100·d_seg^gpu + √(10·d_pose^gpu) + 25·B_gpu/D    <    S_cpu*
  with d_seg^gpu, d_pose^gpu carrying the +0.033 CUDA-vs-CPU inflation already baked in.
```
Re-arranged against the *same scheme's* hypothetical CPU-feasible counterpart `S_cpu(B_gpu)`:
```
  S_gpu_cuda  ≈  S_cpu(B_gpu) + 0.033            (the tax)
```
So GPU-inflate is **strictly dominated** whenever the same byte-shrinking scheme is CPU-feasible
(`S_cpu(B_gpu)` is attainable on CPU, tax-free). GPU-inflate is **viable only** when `S_cpu(B_gpu)` is
**unattainable** because the decode can't run on 4 CPUs in 30 min — i.e. the §2 / §3.2 regime. And even
there it must clear `B_gpu < 128 KB` (to pay the tax) **and** keep its CUDA-inflated d_seg/d_pose small
enough that the distortion terms don't eat the rate win.

**Bottom line:** the GPU's only winning move is to enable a byte-shrink that is *impossible on CPU*. Every
byte-shrink that is *possible on CPU* is better done on CPU.

---

## 5. VERDICT + WHERE THE GPU POWER ACTUALLY BELONGS

### Verdict
- **Does a theoretical GPU-superior inflate candidate exist?** **Yes, narrowly** — the
  GPU-as-expensive-decompressor (§3.2): a Kolmogorov-tiny archive whose witness-producing solve is
  *infeasible on 4 CPUs in 30 min but feasible on a T4*. Mechanism: massive parallel preimage / wide
  fixed-point / deterministic generative solve. Byte threshold: archive **< ~128 KB to tie, < ~50 KB to
  win net** after the +0.033 tax.
- **Is it the right place to spend?** **No, not now** — it is dominated by the CPU path because (a) our
  decode is seconds-long, so the CPU has ~1,800 s of unused budget and the GPU's feasibility edge is moot
  until a decode is ~30× too slow for CPU; (b) the +0.033 tax is a permanent handicap; (c) its *existence*
  rides on the same open "smaller amortizer" question the CPU faces — the GPU doesn't make the amortizer
  exist, only changes what's runnable. The operator's half-anticipated honest answer is correct: **the
  CPU-fast-enough reality + the tax dominate.**

### Where the GPU power belongs: COMPRESS time
Everything compelling about "GPU does powerful things" — gradient descent, massive search, generative
refinement, INR overfit — we can do **at compress time, where we already have unlimited GPU (M5 Max MLX +
Modal) AND no tax AND the scorer is legally available.** The correct repurposing of every §3 idea:

| GPU-inflate idea | COMPRESS-time form (the right place) |
|---|---|
| training-at-inflate (3.1) | **score-aware compress-time training of the small amortizer/seed** (has the scorer; bakes a fixed CPU-cheap decoder) — this is the sibling L1/L2 program |
| parallel preimage solve (3.2) | **compress-time GPU search for the smallest program/latent** whose CPU-feasible decode lands in the cell (the evaluator-equivalence quotient compiler, GOAL lever A) |
| generative denoise (3.2/L5) | **compress-time GPU training of a tiny fixed prior + seed**; inflate runs a *CPU-feasible* fixed-schedule denoise (sibling L5, CPU-feasibility-gated) |

The GOAL_v3 levers A (quotient compiler) + B (score-native carrier) + C (fresh-init score-aware training)
+ E (micro-prior) are **exactly** "use unlimited compress-time GPU to build a smaller CPU-feasible
amortizer." That is the class-shift door below S_floor 0.118 — and it pays **zero tax** because it ships a
CPU inflate. **The GPU thought-experiment resolves into: put the GPU at compress time, ship a tiny CPU
archive.**

### Reactivation criterion for GPU-only inflate (DEFER, not kill)
Re-open §3.2 IF AND ONLY IF a compress-time result produces a **< 50 KB** archive whose witness-producing
decode is **measured infeasible on 4 CPUs in 30 min** (a real CPU-wall, not a convenience) **and**
feasible on a T4 in < 30 min — at which point the +0.033 tax is worth paying because no CPU counterpart
exists. Until then, every byte-shrink is CPU-routed.

---

## 6. WIRE-IN (Catalog #125)

1. **sensitivity-map** — ACTIVE. New prior: *"inflate device is a score lever only via the +0.033 CUDA
   tax; rate is device-free; therefore byte-shrinks belong on CPU unless CPU-infeasible."* Down-ranks all
   GPU-inflate levers below their compress-time twins.
2. **Pareto constraint** — ACTIVE. A GPU-inflate atom is admitted only if `B<128KB` AND its CUDA-inflated
   distortion terms + tax still net `ΔS<0` AND a measured CPU-infeasibility wall exists (the
   GPU-necessity Pareto gate).
3. **bit-allocator** — N/A (no bytes emitted; this is a design/derivation surface).
4. **cathedral autopilot dispatch** — N/A (no archive).
5. **continual-learning posterior** — N/A here; the §5 reactivation measurement would reseed it.
6. **probe-disambiguator** — ACTIVE. The disambiguator: *"is there a byte-shrinking decode that is
   CPU-infeasible-in-30-min yet T4-feasible AND lands in the cell at <50 KB?"* — §3.2 is the probe; its
   answer also answers GOAL's open "smaller amortizer" question (the device-axis projection of it).

## 7. CROSS-REFERENCES

`inflate_compute_as_free_rate_lever_20260611.md` (sibling — the CPU-side levers L1–L5; THIS memo is the
GPU-only extreme + the device-independence collapse that re-routes them to CPU) ·
`local_to_contest_scorer_drift_ladder_and_correction_20260611.md` §1 (RUNG C, the +0.0330 CUDA tax,
MEASURED) · `GOAL_standing_v3_20260610.md` (levers A/B/C/E = the compress-time GPU class-shift;
S_floor=0.11797 RATE-bound; "smaller amortizer is the only door below 0.118") ·
`information_theoretic_floor_report_v1_20260610T102335Z.md` (the rate-bound floor + the open amortizer
question §3.2 inherits) · `upstream/evaluate.sh` (inflate+eval share one $DEVICE — §1.1) ·
`upstream/evaluate.py:63-65,92` (rate = archive bytes / D, device-free; score formula) ·
`upstream/README.md:114` (30-min budget; GPU→T4 16GB, else CPU 4-core/16GB) · CLAUDE.md "Strict scorer
rule" (no SegNet/PoseNet at inflate — bounds §3.1/§3.2 objectives to deterministic stored targets).
