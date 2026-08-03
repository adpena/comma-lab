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

**The `ll1` window solve is NO-GO on the live vehicle. It is not close, and it did not need the
evaluator slot to establish that.** Four results, two of which correct load-bearing claims in the
documents that sent me here.

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
pairs, same renders, differing only in `window_solve`:

> *(n600 result inserted in §3; 40-pair preliminary: ratio mean 2.2043, median 1.3399,
> **97.5% of pairs made worse**, against a tolerance of ≤1.1884 / ≤1.3202.)*

**The median pair alone already exceeds both tolerances.** The rung's best-case seg win is −0.0238 S;
the pose cost exceeds it under *every* extrapolation model I can defend.

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

## §5 Adjudication

*(pending the n600 seg measurement)*

---

## NEXT-IF-RESUMED

*(pending)*
