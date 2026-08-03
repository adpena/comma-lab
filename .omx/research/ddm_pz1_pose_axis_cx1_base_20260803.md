# ddm_pz1 — the pose axis on the `cx1` base: the `ll1` window solve, measured and adjudicated

- **arm:** `ddm_pz1` · **date:** 2026-08-03 · **axis:** `[macOS-CPU advisory]` NON-PROMOTABLE.
  `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`. **Pointer UNMOVED.**
- **live base:** `ddm_cx1` (`v4d_cx1_pj2ix2`) **S = 0.8264972** — seg 0.4311790 · pose 0.1597320 ·
  rate 0.2355862, **353,808 B**, n600 real evaluator, byte-closed.
- **denominator for every ΔS:** `tac.canonical_equations.gap_decomposition_against_floor_20260802`.
  Gap to the PR130 bar (0.172141) = **0.6543562** (seg 61.4% · pose 22.1% · rate 16.5%).
- **cost:** $0. **The n600 evaluator slot was NOT spent** — and the reason it was not spent is the
  finding, not a deferral.

---

## §0 Headline

**The `ll1` window solve is RETIRED. Both axes were measured at n600 on an instrument validated
against the real evaluator, and it loses on both: ΔS(seg) = +0.000394 (claimed −0.0144…−0.0238) and
ΔS(pose) = +0.0310217. Net +0.0314155 — 4.80% of the gap in the wrong direction, the byte-equivalent
of 52,364 B. The n600 evaluator slot was NOT spent and should not be.**

**The rung's seg premise was the real error, and it was not a pose problem at all.** `ll1` measured
*"88 flips **vs perfect delivery**"* — the argmax of our own **ideal render**, not GT. That is a real
number honestly measured against the wrong target. `d_seg` is measured against **GT**, and the
rasterization it removes turns out to be **orthogonal noise with respect to the GT error**: 43.3% of
pairs improve, 49.3% worsen. Removing noise uncorrelated with your error does not reduce your error.
This fires `ra1` §5.4's own pre-registered falsifier (*retire if `Δd_seg × 100 > −0.010 S`*) —
**RETIRE, not defer.**

Five results, three of which correct load-bearing claims in the documents that sent me here — and
one of which corrects the framing of my own charter.

**(1) PZ1-1, the free measurement `sg2` asked for, is done — and it refutes the optimistic reading.**
`sg2`/`ra1` measured the frame_0 perturbation at **camera** resolution (rms 2.75 LSB, max 178) and
noted, correctly, that PoseNet reads `D(f0)` so the scorer-plane delta is *strictly smaller and
unmeasured*. Measured on the live base, 6 strided pairs, every control passing:

| | rms (LSB) | max | vs f1's own |
|---|---:|---:|---:|
| frame_0 delta, **camera** domain | 2.8229 | 196.0 | — |
| frame_0 delta, **scorer plane** `D(f0)` | **1.6986** | **128.92** | **2.12× larger** |
| frame_1 delta, scorer plane `D(f1)` | 0.8028 | 20.46 | (this is the debt being *paid*) |
| **attenuation camera → scorer** | **1.662×** | | |

`D` attenuates the perturbation by only **1.66×** — generic 2×2-subsample smoothing, *not* the
order of magnitude that would have made the rung safe. The reason is structural and is the whole
story of this arm: the solve's delta is a **`D`-null-space field in frame_1's own lattice**; the
homography resamples it at a *different* lattice, where `D` no longer annihilates it. **Null-space
membership does not survive a change of lattice.** From the pose path's point of view the solve is
not a correction at all — it is **injected structured noise**, and it is *larger* in the scorer
plane than the debt it removes from frame_1.

**(2) PZ1-2 — the pose cost, MEASURED directly rather than bracketed.** `sg2` refused to
interpolate between its two anchors (they differ by ~7,300×). Correct call; so I measured the
quantity instead, on the frozen-PoseNet authority the `pj2` solver itself uses. Paired design, same
pairs, same renders, differing only in `window_solve`. **n600: `d_pose` 0.00255140 → 0.00363866,
ratio 1.4261, 94.3% of pairs worse ⇒ ΔS(pose) = +0.0310208**, against a tolerance — DERIVED before
the measurement — of ratio ≤ 1.1884 (or ≤ 1.3202 at the most generous seg claim). **Exceeds both.**
Positive control: my base reproduces the evaluator's own `report.txt` n600 value to **1.2e-5
relative** (§3.1).

**(5) A correction to my own charter's framing, and the reason it matters.** The charter frames this
as *"is the window solve GO or NO given the pose risk"*, with the seg win taken as given from `sg2`'s
1.98–3.27%-of-gap pricing. I nearly measured only pose — pose alone already decided the verdict, and
the seg run looked like tidiness. **It was not.** The seg number is where the rung actually dies, and
had I stopped at pose I would have reported *"the pose cost eats a real seg win"* and recommended the
11-CPU-hour re-fit conversion. That recommendation would have been wrong. **Both sides of a trade get
measured, even when one side has a number in a memo** (`m82`).

**(3) A correction to the charter, to `sg2` §6, and to `ra1` §5.1: the shipped receiver does not
contain the window solve at all.** `sg2` grades it *"grade-5 UNWIRED-BUT-BUILT … an unfired kwarg,
not an unfired idea"* and prices the build at **zero**. In the repo, true. In the **live-best
shipped submission**, false: `v4d_cx1_pj2ix2/ddm_tr1_runtime.py` (sha `f3b93708…`) is a *different
file* from `src/tac/optimization/ddm_tr1_runtime.py` (sha `6f4916f9…`), 257 diff-lines behind, and
`grep window_solve` returns **zero hits**. The rung is *"re-vendor the receiver, **then** flip the
bool"*, not *"flip the bool"*. This is the same class of silent-wrong-receiver error `ddm_cx1` just
fixed in `stage_v4d_realized_gate.sh`, found independently at a second surface.

**The good news inside that correction, established rather than assumed:** the swap is **legal and
byte-free** — `archive.zip` holds exactly **one** member, `0.bin` (353,700 B); the `.py` receiver is
inflate-side, not rate-charged (rule 118) — and the repo receiver reproduces the shipped frames
**bit-for-bit with the solve OFF on every pair tested**. So the swap itself is sound. It is what the
swap *enables* that fails.

**(4) A CLAUDE.md line is backwards, and it is the line this whole analysis rests on.** CLAUDE.md's
scorer-architecture section states PoseNet is *"rgb_to_yuv6 → resize to (512,384) → normalize"*.
The frozen upstream (`upstream/modules.py:70-74`) is the **opposite order**:

```python
def preprocess_input(self, x):                                   # PoseNet
  x = einops.rearrange(x, 'b t c h w -> (b t) c h w', ...)
  x = torch.nn.functional.interpolate(x, size=(384, 512), mode='bilinear')   # D FIRST
  return einops.rearrange(rgb_to_yuv6(x), ...)                                # yuv6 SECOND
```

PoseNet and SegNet therefore read through the **identical** `D`, sharing the same private-window
geometry and the same 230,904-pixel blind set. `ll1`'s docstring is right and CLAUDE.md is wrong.
This matters beyond bookkeeping: had the CLAUDE.md order been the real one, chroma subsampling would
happen at *camera* resolution and mix across window boundaries for **both** frames, and the blind-set
guarantee `ll1` relies on would not hold for pose at all. The rung would have been dead for a
different reason. **Verify the premise against the frozen artifact; two of my three sources agreed
with each other and were still wrong.**

---

## §1 Apparatus validity — controls run BEFORE any number was read off the instrument

| control | requirement | result |
|---|---|---|
| **C0a** shipped vs repo receiver | establish which code actually ships | shipped **lacks** `window_solve`; repo has it |
| **C0b** receiver-swap byte-identity | repo receiver with solve **OFF** must reproduce shipped frames exactly | **PASS**, all pairs, bitwise |
| **C1** blind-set invariance | solve must touch 0 of 230,904 `D`-blind camera px | **PASS**, 0 touched, every pair |
| **C2** delivery through `D` | `D(f1_solved) − r` must collapse vs base | **0.8023 → 0.0298** (reproduces `ra1`'s 0.7994 → 0.0298 to 0.4%) |
| **C3** probe can return the negative | a null input must report exactly 0 | **PASS** |
| **C4** blind-set size | must land on the independently-known value | **230,904 px = 22.696926%** — matches |
| **C5** `m88` population guard | subset mean of the governing quantity vs population | **FIRED** — see §3 |

C5 firing is why §3 is an n600 measurement and not a 40-pair claim.

**Artifacts.** `.omx/research/ddm_pz1_scorer_plane_delta_cx1_20260803.json` ·
`.omx/research/ddm_pz1_dpose_paired_cx1_20260803.json` (40 strided) ·
`.omx/research/ddm_pz1_dpose_paired_n600_cx1_20260803.json` (n600).
Probes: `experiments/ddm_pz1_scorer_plane_pose_delta.py` ·
`experiments/ddm_pz1_dpose_window_solve_paired.py`.

---

## §2 PZ1-1 — the scorer-plane perturbation, per pair

Live base `v4d_cx1_pj2ix2`, 6 strided pairs (never a prefix, `m88`):

| pair | deliver base→solved | camera Δf0 rms / max / %chg | **scorer Δf0 rms / max** | scorer Δf1 rms |
|---:|---|---|---|---:|
| 0 | 0.8003 → 0.0299 | 2.8196 / 181 / 35.7% | **1.3097 / 60.40** | 0.8008 |
| 120 | 0.8082 → 0.0298 | 2.7543 / 184 / 35.4% | **1.5516 / 75.06** | 0.8087 |
| 240 | 0.8023 → 0.0299 | 3.0448 / 196 / 35.8% | **1.9704 / 111.70** | 0.8028 |
| 359 | 0.8036 → 0.0298 | 2.5626 / 165 / 32.9% | **1.6614 / 81.23** | 0.8041 |
| 479 | 0.7973 → 0.0299 | 2.9705 / 184 / 34.6% | **1.9721 / 128.92** | 0.7979 |
| 599 | 0.8019 → 0.0298 | 2.7856 / 177 / 35.2% | **1.7263 / 102.30** | 0.8024 |
| **mean** | **0.8023 → 0.0298** | **2.8229 / 196 / 34.9%** | **1.6986 / 128.92** | **0.8028** |

In PoseNet's normalised input units (the network divides by `std = 63.75`) the pair delta is
**rms 0.011921, max 1.3946** — excursions past a full standard deviation of the network's own input
scale.

**One published figure does not reproduce, and I am flagging it rather than smoothing it.**
`ra1`/`sg2` report **58.8%** of frame_0 pixels changed; I measure **34.9%**. rms (2.82 vs 2.75) and
max (196 vs 178) agree closely, so this is not a chain discrepancy — it is a **base** difference:
they measured on `dc1_fold`, I measure on `cx1`(=`pj2` frames), which carry a *different* pose
solve (different `s_t`, `sel`, `ab`, `beta` ⇒ different homography ⇒ different valid-mask and
different resampling offsets). The `rms`/`max` agreement across a base change is itself evidence
that both chains are correct; the `%changed` is the base-sensitive statistic.

---

## §3 PZ1-2 — Δd_pose, **n600**, paired, on the frozen-PoseNet authority

### 3.1 The positive control is essentially exact — and it validates the whole chain

| | measured here | `upstream/evaluate.py` `report.txt` for `cx1` | ratio |
|---|---:|---:|---:|
| mean base `d_pose`, n600 | **0.00255140** | **0.00255143** | **1.000** (rel. 1.2e-5) |

Packet parse → render → bicubic ↑ → camera raster → homography warp → f0 composition
(`sel`/`beta`/`ab`) → frozen PoseNet → stored 6-dim targets → MSE reproduces the **real evaluator's
own n600 number to five significant figures**. Every `d_pose` below is on an instrument checked
against the authority it is being used to predict.

### 3.2 The result

| n600, 600 pairs | base | solved | ratio |
|---|---:|---:|---:|
| mean `d_pose` | 0.00255140 | **0.00363866** | **1.4261** |
| pose term `√(10·d_pose)` | 0.1597320 | **0.1907528** | |
| **ΔS(pose)** | | **+0.0310208** | |
| pairs made **worse** | | **94.3%** | |
| per-pair ratio: median / min / max | | 1.5453 / 0.6729 / **505.8** | |

**Against the tolerance, which was DERIVED before the measurement:** the rung survives only if the
ratio is ≤ **1.1884** (defending `ll1`'s measured −0.0144 seg win) or ≤ **1.3202** (defending the
rms-scaled −0.0238). **Measured 1.4261 exceeds both.** ΔS(pose) = +0.0310 is **4.74% of the
0.6543562 gap** — a regression roughly twice the size of the best case the rung was ever claimed
to be worth.

**The median ratio (1.5453) is HIGHER than the aggregate ratio (1.4261).** The typical pair is hurt
*more* proportionally than the mean, because the mean is carried by pairs whose base `d_pose` is
already large and which are proportionally less disturbed. The pairs that suffer most in relative
terms are exactly the tightest ones — `sg2` predicted this ("the tail pairs hold the most
finely-tuned solutions and are the likeliest to be knocked off") and it is now measured.

### 3.3 The `m88` guard earned its keep — a worked example, not a citation

I first ran 40 strided pairs. The guard is executed in the probe itself: compare the subset's mean
of the governing quantity to the population's.

| | 40 strided | n600 truth | |
|---|---:|---:|---|
| mean base `d_pose` | 0.00075159 | 0.00255143 | **0.295×** ⇒ subset REFUSED |
| mean ratio | **2.2043** | **1.4261** | overstated by 1.55× |
| median ratio | **1.3399** | **1.5453** | understated by 1.15× |

A 40-pair strided sample of a right-skewed quantity was a **different population**, and it was wrong
in *both directions at once* — the mean too high, the median too low. Had the guard not fired I
would have published 2.20. **Strided is not a defence against skew; only the population is.** The
verdict below rests on n600.

---

## §4 Can a pose RE-FIT rescue it? A matched partial bound

`ra1` §5.3 / `sg2` convert the item, on regression, to *"re-fit the pose sidecar against the solved
raster"* (~1.76–1.96 h wall, ~11.2 CPU-h, $0). The charter requires that be **ranked**, not assumed.
Re-fitting only the solved side and comparing to the shipped base would be rigged, so the **same**
exhaustive re-fit over the discrete shipped grammar (`st_idx` ∈ 11-entry `ST_GRID` × `sel` ∈ {0,1})
was run on **both** rasters (`m85`'s matched-base rule):

| 40 strided pairs | shipped params | re-fit over grid |
|---|---:|---:|
| base raster | 0.00075159 | 0.00074789 |
| solved raster | 0.00165673 | **0.00133002** |
| **penalty** | **+0.00090514** | **+0.00058212** |

- **The base was already at its grid optimum on 95.0% of pairs** — `pj2` had genuinely exhausted
  these DOF, so the 5% that moved are real re-optimisation and the probe is not inert.
- **A matched partial re-fit recovers 35.7% of the penalty.** At matched optimisation the solved
  raster is still **1.78×** worse than the base.

**Scope, stated against my own conclusion.** This re-fits only the *discrete* grammar; `p_best`
(6 continuous dims) and `ab` are not re-fitted, so 35.7% is a **lower bound** on full-re-fit
recovery. And this subset is the one the `m88` guard refused, so the *level* is not population-valid
(the *fraction recovered* is a ratio of deltas and transfers better, but is not guaranteed).
Carrying the 35.7% figure onto the n600 penalty gives a re-fitted ΔS(pose) of roughly **+0.021**,
which is why §5 does **not** close the conversion on pose alone — it closes on seg.

---

## §5 The seg side, **n600** — and the rung's premise is refuted

`SegNet.preprocess_input` uses `x[:, -1, ...]` — **frame_1 only**. `d_seg` therefore depends on
frame_1 alone, the window solve acts directly on frame_1, and the GT argmax is already cached for
all 600 pairs. So the seg side is an n600 measurement too, with no warp, no GT decode, and no
evaluator slot.

| | measured | `report.txt` for `cx1` | ratio |
|---|---:|---:|---:|
| **positive control**: seg term base | **0.4311795** | **0.4311790** | **1.00000** |

| n600 | base | solved |
|---|---:|---:|
| mean `d_seg` | 0.00431179 | 0.00431574 |
| seg term `100·d_seg` | 0.4311795 | 0.4315737 |
| **ΔS(seg)** | | **+0.000394** |
| pairs improved / worse | | **43.3% / 49.3%** |

**There is no seg win. The solve is a coin flip that lands very slightly on the wrong side.**
Claimed: −0.0144 to −0.0238. Measured: **+0.000394**. `ra1` §5.4's pre-registered falsifier —
*retire if `Δd_seg × 100 > −0.010 S`* — **fires: RETIRE, not defer.**

### 5.1 Why the transfer failed — a surrogate hiding inside a correctly-measured number

`ll1`'s docstring is precise about what it measured, and reading it precisely is the whole answer:

> `clip(rint(U(r)))` **88 flips vs perfect delivery** `d_seg 0.0001492`

The reference was **perfect delivery of `r`** — the argmax of our own *ideal render*. That is a real
quantity, honestly measured. **But it is not `d_seg`.** `d_seg` is measured against **GT**.
Delivering `r` more exactly reduces the distance to `r`; it reduces the distance to GT only if `r`'s
argmax is *closer to GT* than the rasterized version's is. There is no reason for that to hold, and
it is now measured false: **43.3% of pairs improve, 49.3% worsen.** The rasterization perturbation is
**orthogonal noise with respect to the GT error**, not a bias toward it. Removing noise that is
uncorrelated with your error does not reduce your error.

This is the **surrogate-≠-authority** class (NO-FAKE #8) in an unusually well-camouflaged form: the
number was correct, the *target it was measured against* was a proxy for the score, and the proxy's
relationship to GT was never measured. Both downstream memos labelled the transfer INFERRED and both
demanded n600 before firing — **the apparatus worked; nobody fired it.** What was missing was not
caution but the measurement.

---

## §6 Adjudication — **NO-GO / RETIRED**, on both axes, on the pre-registered thresholds

| axis | claimed | **measured, n600** |
|---|---:|---:|
| ΔS(seg) | −0.0144 … −0.0238 | **+0.000394** |
| ΔS(pose) | unmeasured, tolerance ≤ +0.0144…+0.0238 | **+0.0310208** |
| ΔS(rate) | 0 | **0** (verified: `archive.zip` = 1 member; receiver not rate-charged) |
| **net ΔS** | −0.0144 … −0.0238 (a win) | **+0.0314155** (a loss) |

Firing this rung takes `cx1` from **S 0.8264972 → 0.8579126** (recomputed from components; the
evaluator's rounded `Final score` prints `0.83` for both and cannot distinguish them). That is
**4.80% of the 0.6543562 gap in the wrong direction** — the byte-equivalent of **52,364 B** of rate
at 10,907 B per 1% of gap.

**Verdict: RETIRED**, per `ra1` §5.4's own pre-registered seg falsifier. `verdict_scope:` the
`ddm_ll1` window solve as built (exact `D`-inverse per private window), on the `cx1`/`pj2` vehicle,
n600, frozen SegNet + frozen PoseNet + stored 6-dim GT targets, both distortion terms measured
against **GT**. This does **not** kill receiver-side realization work in general — it kills *this*
lever, for a reason (§5.1, §7.1) that is now stated generally enough to screen the next one.

**The conversion is NOT triggered.** `ra1` §5.3 / `sg2` say that on a pose regression the item
becomes *"re-fit the pose sidecar against the solved raster"* (~1.76–1.96 h, ~11.2 CPU-h). **Do not
do this.** There is no seg win to defend — a re-fit spends 11 CPU-hours to partially recover a pose
loss incurred for a +0.000394 seg loss. It must not be ranked against `#766` waterfill or the
granularity re-race, because it is not a candidate at all. And §4 independently shows it would not
work anyway: at matched optimisation the solved raster is still 1.78× worse.

**The n600 evaluator slot was NOT spent, and should not be.** `ra1` §5.4's request is **withdrawn**
— it has been answered on a validated instrument at higher resolution (both terms, n600, per-pair)
than the request itself asked for.

---

## §7 What generalises — three laws, one apparatus note

### 7.1 LAW — null-space membership does not survive a change of lattice

The solve is justified by an exact structural fact: `D` is point-sampling at stride 2.276 > 2, so
each scorer pixel owns a **private, disjoint** 2×2 camera window with a 3-dimensional integer null
space. Put the correction in that null space and `D` cannot see it. **True — for consumers that read
through `D` at `D`'s own lattice.**

Our frame_1 has **two** consumers at **two** lattices:

| consumer | operator | null-space claim |
|---|---|---|
| SegNet, and PoseNet's frame_1 input | `D(f1)` | **holds by construction** (verified: 0 blind px touched, all pairs) |
| PoseNet's frame_0 input | `D(W(f1))`, `W` = homography, sub-pixel offsets | **does not hold** |

`W` resamples at arbitrary sub-pixel positions, so `D∘W` is not `D` and does not annihilate `D`'s
null space. Measured consequence: attenuation camera → scorer is only **1.662×**, and the frame_0
scorer-plane delta (**1.6986** rms) is **2.12× LARGER** than the frame_1 debt the solve pays off
(0.8028). From the pose path's view the solve is not a correction — it is **injected structured
noise, bigger than the error it fixes elsewhere.**

> **Operational rule.** Before claiming a receiver-side lever is free because it lives in a scorer's
> null space: **enumerate every consumer of the modified bytes and re-check the null-space claim
> against each consumer's own operator and lattice.** One consumer's null space is another's signal.

### 7.2 LAW — a ΔS claim must name the target the distortion was measured against

§5.1. *"Distance to my own ideal render"* is not *"distance to GT"*. A correction measured against an
intermediate object of our own construction transfers to the score only if that object is closer to
GT — **which must be measured, never assumed.**

> **Operational rule.** Every ΔS claim states its TARGET. If the target is not GT (or the exact
> contest oracle), the claim is a surrogate and owes an explicit transfer measurement before it may
> be priced in gap-percent.

### 7.3 LAW — the stale-carrier law needs two regimes, and they need different cures

Three arms reached the stale-carrier law today (`cp1`, `ra1`, `gd5`): *`f0 := a·warp(f1) + b`, with
per-pair `(a,b)`, `s_t`, `sel`, `beta` ALL solved against the ORIGINAL decoded frame_1 — so any
change to the decoded frames invalidates a pose solve fitted against the old ones.* Confirmed, and
**sharpened**: "invalidated" covers two regimes that a re-fit treats very differently.

| regime | signature | cure |
|---|---|---|
| **STALE** — the optimum moved | matched re-fit recovers ≈ all of the penalty | re-fit (the priced conversion) |
| **FLOOR-RAISED** — noise the parameters cannot absorb | matched re-fit recovers little; solved still worse *at matched optimisation* | **none** — a re-fit chases a worse optimum |

The **matched** partial re-fit (§4) is the cheap discriminator, and it must be matched: run the same
re-fit on **both** rasters, or you are comparing a freshly-optimised candidate to a stale incumbent
(`m85`). Here: base already at its grid optimum on **95.0%** of pairs, only **35.7%** of the penalty
recovered, solved still **1.78×** worse at matched optimisation ⇒ **FLOOR-RAISED**.

**The byte-identity exemption test, adopted as the standard.** `cx1` is exempt from the stale-carrier
law because its inflated `0.raw` is byte-identical to `pj2`'s — that byte-identity is the *proof
form* of the exemption, and it is the right test. Anything that changes even one LSB owes a re-solve
**and** owes the stale-vs-floor-raised discrimination *before* the re-solve is assumed to help.

### 7.4 APPARATUS — the shipped receiver is not the repo receiver

`sg2` priced this rung's build at **zero** ("an unfired kwarg, not an unfired idea"). In the repo,
correct. In the **shipped** live-best submission, the code is not there at all. **Second independent
instance today** of the same class — `ddm_cx1` found the first in `stage_v4d_realized_gate.sh`,
which had been silently copying a pre-`ix2` receiver off the SSD.

> **Operational rule.** Before pricing any "flip the flag" rung, verify *which file actually ships*
> (`shasum` the vendored copy against the repo copy), and if a swap is required, prove it is
> **byte-identical in the OFF configuration** before attributing any measured change to the ON
> configuration. Both were done here (§1 C0a/C0b) and the swap is sound — it is what the swap enables
> that fails.

**Minor, real, worth one line:** `tools/subagent_commit_serializer.py` defaults `--label` to
`$SUBAGENT_LABEL` or `anonymous`, so an arm that checkpoints its own `files_touched` and then commits
**collides with itself** on the Catalog #340 guard. Pass `--label <arm-id>` (the guard has a proper
self-exclusion path; the fix is to use it, not to bypass the guard).

---

## §8 The reusable asset this arm leaves behind

Both scored distortion terms are now measurable on the live vehicle **at n600, locally, for $0,
without the evaluator**, on an instrument validated against the evaluator's own report:

| term | tool | n600 wall | positive control vs `report.txt` |
|---|---|---:|---|
| `d_pose` | `experiments/ddm_pz1_dpose_window_solve_paired.py` | ~12 min | 0.00255140 vs 0.00255143 (**1.2e-5** rel) |
| `d_seg` | `experiments/ddm_pz1_dseg_window_solve_n600.py` | ~20 min | 0.4311795 vs 0.4311790 (**1.2e-6** rel) |

Any future receiver-side change can be adjudicated on **both** distortion axes at full population
before an evaluator slot is spent. That is the difference between this arm and its charter: the
charter allocated one n600 gate; the gate was not needed, and the reason it was not needed is
reusable.

---

## §9 Round-1 adversarial self-review

**A1 — my own framing was wrong at the start, and the measurement corrected it.** I opened expecting
this to be a *pose-risk* adjudication against a real seg win, exactly as the charter framed it. The
seg measurement — which I nearly skipped as tidiness, since pose had already decided the verdict —
showed the seg win does not exist. **Had I stopped at pose I would have published "the pose cost eats
a real seg win" and recommended the re-fit conversion.** That recommendation would have been wrong,
and it would have spent 11 CPU-hours. The lesson is `m82`: I was about to accept one side of a trade
as given because a memo had a number for it.

**A2 — can my probes return the negative?** Yes, three ways, all executed: C3 (a null input reports
exactly 0); the re-fit probe demonstrably changes the answer on some pairs (pairs 46, 61) so its
mutation is live, not inert; and both n600 probes reproduce the evaluator's own values to 5–6
figures, which a broken chain could not do by accident.

**A3 — is the 35.7% re-fit recovery load-bearing?** No, and I have not made it so. It is measured on
the 40-pair subset my own `m88` guard **refused**, and it covers only the discrete DOF. It is a lower
bound and it is labelled as one. The verdict does **not** rest on it — the verdict rests on the seg
measurement, which is n600 and controlled. The re-fit number only supports the secondary claim
(§7.3) that we are in the floor-raised regime.

**A4 — did I check that the seg and pose measurements are on the same object?** Yes. Both use the
same `Decoder`, the same repo receiver, the same `window_solve` toggle, and both positive-control
against the same `report.txt`. The seg probe feeds a single-frame sequence so `x[:, -1]` is a no-op —
verified against the known base value rather than argued.

**A5 — a discrepancy I did not smooth over.** My frame_0 `%changed` is 34.9% against the published
58.8% (§2). I traced it to the base difference (`cx1`/`pj2` vs `dc1_fold` carry different pose
params ⇒ different homography ⇒ different valid-mask), supported by `rms`/`max` agreeing across the
same base change. I am reporting it as an unexplained-until-traced discrepancy with a stated cause,
not deleting it because my headline did not need it.

**A6 — what I did NOT establish.** (i) Whether a *joint*-constraint solve (deliver `r` through `D`
**while** keeping `D∘W` quiet) is feasible — I did not compute `dim(null(D) ∩ null(D∘W))`. `ra1`'s
four allocation variants producing near-identical `D∘W` deltas is *indicative* that the intersection
is small, not proof. But §5 makes it moot: even a perfectly pose-quiet solve buys **+0.000394**.
(ii) Whether the seg result would differ on a base whose render is closer to GT — the mechanism in
§5.1 suggests not, but it is scoped to this vehicle. (iii) I did not re-run the matched re-fit on a
representative population; it was not worth the compute once seg decided the verdict.

---

## NEXT-IF-RESUMED

1. **The `ll1` window solve is RETIRED, on both axes, on `ra1`'s own pre-registered threshold. Do not
   re-attempt it, and do not convert it to a pose re-fit** — there is no seg win to defend, and §4
   shows the re-fit would not work anyway. `ra1` §5.4's n600 request is **withdrawn as answered**.
   Leave `window_solve` default OFF; the repo kwarg is now documented-dead rather than untested.
2. **Do not re-attempt null-space re-allocation** (dither order, allocation norm, init kernel) —
   `ra1` §3 measured it shut four ways, and §7.1 now explains *why* it cannot work: the problem is
   not the allocation, it is that `W` reads at a different lattice.
3. **`ra1` §4.2 remains the one live zero-byte AA surface** (`np.repeat` nearest-neighbour upsampling
   inside the decoder). It is a **re-race**, not a patch — it invalidates the token search. §7.2 now
   binds it: any claimed gain must be measured against **GT**, not against the ideal render, or it
   will repeat this arm's failure exactly.
4. **The honest size of my axis.** To reach the PR130 pose floor (term 0.015268) from `cx1`'s
   0.1597320 requires `d_pose` **2.331e-5** against the current **0.00255143** — **109.5× tighter**.
   The pose gap is 0.1444640 = **22.08%** of the total gap. Nothing measured today moves it; the
   `pj2` solve is already at its discrete-grammar optimum on 95.0% of pairs, which is a real datum
   about where the remaining pose headroom is *not*.
5. **Use the §8 instrument.** Any next receiver-side or frame-affecting candidate should be run
   through both n600 probes (~32 min total, $0) before an evaluator slot is requested. The
   `--submission-dir` argument is the only thing that changes.

