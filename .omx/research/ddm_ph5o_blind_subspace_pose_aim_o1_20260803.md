# ddm_ph5o — O1 ANSWERED: the D-blind subspace CAN be aimed, and the descent is ADDRESS-LIMITED

- **arm:** `ddm_ph5o` · **date:** 2026-08-03 · **axis:** `[macOS-CPU advisory]` NON-PROMOTABLE.
  `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`.
  **Contest pointer 0.1910828242 UNMOVED.**
- **cost:** $0. Held the single scorer slot; released at the end of this memo.
- **baseline, named (memory `m46` — a ΔS without its baseline is unanchored):** `ddm_pu2`,
  archive sha `c72ef357`, **353,805 B**, n600 `report.txt`: d_seg 0.00431179 · d_pose 0.00154519
  · S **0.7910689** (seg 0.4311790 · pose 0.1243057 · rate 0.2355842).
- **floor:** PR130 **0.1721413**. Gap decomposition computed by the canonical equation
  `gap_decomposition_against_floor_20260802`, not by hand: **total gap 0.6189276**,
  **seg 0.4015190 (64.87%) · pose 0.1090381 (17.62%) · rate 0.1083705 (17.51%)**,
  **9,295.2 B per 1% of gap**.

---

## §0 Headline — the answer has two halves and they point opposite ways

> **ALIGNMENT: YES.** The 692,712-dimension-per-pair D-blind subspace contains a real, usable
> descent direction for `d_pose`. Measured at n600 through the shipped receiver and the frozen
> CPU-torch PoseNet, with a sign-asymmetry positive control that a single-sign sweep could not
> have produced.
>
> **RATE: NO, by roughly an order of magnitude.** The descent direction is a handful of
> **isolated pixels** picked out by a scorer-derived saliency. **Its address IS its information**,
> that address is per-pair private (measured pairwise Jaccard **0.0056**), and naming the pixels
> costs **3.5×–62×** more than the pose they buy is worth — at every support size measured, n600.

**`ddm_ph4`'s KILL(seg) HELD EXACTLY, and is now proven one level deeper.** Every treated frame
in every cell came back with `max|D(f1') − D(f1)| = 0.0e+00`, and the C7 end-to-end control ran
the **real frozen SegNet** and got **argmax bit-identical at amplitudes 1, 8 and 64** — 692,712
camera values changed per frame, **zero** argmax pixels moved — against a cardinality-matched
D-visible control that moves 13–27 argmax pixels every time. ph4's proof is not just intact; it
now rests on the network, not only on a reading of `modules.py`.

**The structural reason the cheap form cannot work, in one sentence:** the free, generic,
deterministically-generated basis that rule 118 lets us run for zero bytes is *smooth and
delocalised*, and the useful blind direction is a *~12-pixel spike* — **the cheapness of the
basis and the localisation of the descent are the same property with opposite signs.**

---

## §1 Control ledger — everything below rests on these, and two of them fired on me

| # | control | result | verdict |
|---|---|---|---|
| C1 | blind mask reproduces 230,904 px = 22.696926% | 230,904 / 22.696926% | **PASS** |
| C2 | ph4's exact zero: `max\|D(f1+δ_blind) − D(f1)\|` | **`0.0e+00`** | **PASS** |
| C3 | cardinality-matched **D-visible** edit at amp 3 moves `D` | **3.0000** exactly | **PASS** |
| C4 | warp adjoint dot-product identity `⟨Lu,v⟩ == ⟨u,Lᵀv⟩` | rel err **1.895e-15** | **PASS** |
| C5a | re-implemented forward warp vs shipped `warp_rgb`, float | 8.53e-14 = **1.51 ULP** @255 | **PASS** |
| C5b | full re-implemented chain vs shipped `Decoder.f0`, uint8 | **`0.0e+00`** bit-identical | **PASS** |
| C6 | `d_pose` bit-reproducible on repeated identical input | exact | **PASS** |
| C7 | **real frozen SegNet argmax** identity under a blind edit | **0 px differ**, amps 1/8/64, 4 pairs | **PASS** |
| C7c | C7's positive control: visible edit **must** move the argmax | 13/18/27/18 px | **PASS** |

### 1.1 Two instrument defects I found in my OWN code, recorded rather than smoothed

**(a) I set C5 wrong and it fired on the first run.** I demanded the re-implemented forward warp
be *bit-identical* to the shipped `warp_rgb`. It is not, and it cannot be: I accumulate
`Σ_j I_j·w_j` while `warp_rgb` computes `(Ia(1−wx)+Ib·wx)(1−wy) + (Ic(1−wx)+Id·wx)·wy` —
algebraically identical, different association, so they agree to **1.51 ULP at magnitude 255**,
not to the bit. The leg that is actually load-bearing — **the uint8 frame the receiver emits** —
*is* bit-identical at 0.0. The gate now has two legs with two different tolerances, and the
reason is in the source. *A gate demanding bit-identity from a re-associated float sum is a
false blocker (`#828`'s class), and mine was one.*

**(b) My first solver had the `m50` vacuity defect inside it.** It returned the **base** `d_pose`
whenever no candidate improved — making *"stepped and got worse"* and *"never stepped at all"*
**the same symbol**. The first smoke duly reported `rel +0.0000%, improved 0.0%` on 8/8 pairs
and 2 independent treatments, which I read as a suspiciously clean null rather than a result.
The fix is structural: **every row now carries an anti-vacuity probe** — a unit constant-mode
blind field whose changed-pixel count is recorded — so no reported number can be a fallback.
With it, the same pairs report the *actual* behaviour: **12/12 frames provably changed, mean
d_pose +109.09%**. The "null" was a *catastrophic rise* wearing a zero's clothes.

---

## §2 The measurement — the support sweep at fixed ±1 LSB

**Why support and not amplitude.** A full-field ±1 LSB sign step — the smallest amplitude a uint8
camera raster can express, applied to all 230,904 blind pixels at once — **raises** `d_pose` by
**4.263e-01**. That is **153×** the entire first-order drop the measured gradient promises
(`Σ|g| = 2.784e-03`) and **4.5× the base `d_pose` itself**. A linearisation whose promised drop
exceeds the whole objective is simply outside its trust region, so that step could not establish
"misaligned". With the quantum pinned at ±1 by uint8, the **only** remaining free parameter is
the **support** — how many blind pixels move.

**The instrument.** `experiments/ddm_ph5o_blind_pose_sparsity.py`. Rank blind pixels by `|g|`,
where `g = ∂d_pose/∂f1_blind` is obtained by autograd through the frozen PoseNet to f0 at camera
resolution and then pulled back through the **exact adjoint of the shipped warp** (C4, 1.9e-15).
Apply `−sign(g)` at ±1 LSB to the top-n. Score through the real `Decoder.f0` → real `_to_uint8`
→ frozen PoseNet. **n600, all 600 pairs, never a prefix.**

**The chain's own positive control.** My instrument's n600 base `d_pose` is **0.0015451713**
against the shipped `report.txt` **0.00154519** — **1.21e-05 relative**, the same reproducibility
floor `pz1` §3.1 quotes. The whole path (shipped receiver → shipped warp → uint8 → frozen
PoseNet → `d_pose_u8`) reproduces the archive's own row before it is asked to measure a delta.

**MEASURED, n600, all 600 pairs. `KILL(seg)` bit-identical in every one of the 7,200 cells
(600 pairs × 6 supports × 2 signs): `max|D(f1′) − D(f1)| = 0.0e+00`.**

| support n | descent `−sign(g)` | ascent `+sign(g)` | pairs improved | **asymmetry** |
|---:|---:|---:|---:|:--|
| 1 | **−0.2704%** | +0.2542% | 88.3% | descent wins |
| 2 | **−0.4911%** | +0.5510% | 85.5% | descent wins |
| 5 | **−0.8112%** | +1.4166% | 69.5% | descent wins |
| **12** | **−1.2181%** ← best fixed | +3.5889% | 49.2% | descent wins |
| 27 | −0.6525% | +8.6051% | 28.8% | window closing |
| 61 | **+4.6591%** | +22.4014% | 18.0% | window closed |

**Per-pair ORACLE support selection** (each pair takes its own best n, or none): `d_pose`
0.00154517 → **0.00145373 = −5.9177%**, **92.7% of pairs improve**, mean support **13.68 px/pair**.
That is the upper bound on this actuator.

**The descent window closes at a support of order ten pixels out of 230,904.** That single fact
is the whole result: the useful object is a *spike*, and §3 is what a spike costs to name.

**The positive control on the aiming itself is the ascent column.** If the response were purely
second-order the two signs would be symmetric and the gradient would carry no information. They
are not symmetric, and the asymmetry runs the right way at every support. **This is the evidence
that O1's alignment question is answered YES**, and no single-sign sweep could have produced it.

---

## §3 The byte arithmetic — where it dies

The decoder cannot run PoseNet (CLAUDE.md, "no scorers at inflate time"), so a correction that
names individual blind pixels must carry **both the address and its sign**:
`log2(230,904) = 17.817` bits + 1 sign bit = **2.352 B/pixel/pair**. This is a **combinatorial**
cost, not a statistical estimate — it does not shrink with more measurement.

| support n | bytes (600 pairs) | ΔS_pose | ΔS_rate | **ΔS joint** | % of total gap | **byte cut needed** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,411 | −0.000168 | +0.000940 | **+0.000772** | +0.125% | **5.6×** |
| 2 | 2,823 | −0.000306 | +0.001879 | **+0.001574** | +0.254% | 6.1× |
| 5 | 7,056 | −0.000505 | +0.004699 | **+0.004193** | +0.678% | 9.3× |
| 12 | 16,935 | −0.000759 | +0.011276 | **+0.010517** | +1.699% | 14.8× |
| 27 | 38,104 | −0.000406 | +0.025372 | **+0.024966** | +4.034% | 62.5× |
| 61 | 86,087 | +0.002863 | +0.057322 | **+0.060185** | +9.724% | — (pose worse too) |
| **oracle** | 19,514 | −0.003734 | +0.012994 | **+0.009260** | +1.496% | **3.5×** |

**Every configuration is net negative.** The pose gain is real and it is *small in score units*
— the largest is the oracle's **−0.003734**, which is 3.42% of the pose gap — while the bytes
needed to name the pixels cost **+0.012994**. The rate term wins by **3.5× at best** and by
**5.6×** for the best shippable fixed support.

*(The equation reproduces this table independently from the measured relative-`d_pose` values and
the combinatorial address cost — two code paths, same numbers.)*

**Against the pre-registered break-even.** The charter pre-registered *2.20% of the pose gap at
k=6 / 3,600 B*. Recomputed by the canonical equation: 3,600 B costs ΔS_rate **+0.0023971** =
**2.1984%** of the live pose gap — the charter's figure reproduces **exactly**. Turned into the
requirement it implies: **a 3,600 B correction must cut `d_pose` by 3.8196%.**

---

## §4 The two escapes, both measured, both refuted

**(1) "Ship the address table once, ship only per-pair signs."** This is the escape that would
have moved the arithmetic by ~19×, so it had to be measured rather than argued. Top-12 supports
over 16 strided pairs: **180 distinct pixels of a possible 192** (93.75% unique), only **10**
pixels chosen by more than one pair, **mean pairwise Jaccard 0.0056**, only **10%** of pair-pairs
share a single pixel, a global consensus set covers **12.5%** of any pair's own support, and
sign agreement on it is **0.18** (near-random). **The support is per-pair private. The address is
the information.** Artifact: `ddm_ph5o_support_sharing_20260803.json`.

**(2) "Use a free generic basis — rule 118 makes it zero bytes."** Measured directly: a rank-6
separable-DCT basis on the blind mask, solved per pair by Gauss-Newton on the 6-scalar pose
residual with real evaluations, then quantised to 1 B/coefficient. **100% of pairs solved to the
all-zero integer coefficient vector** (12 strided pairs — labelled, not an n600 claim). That is not a solver failure, it is an inner product: the
descent is a ~12-pixel spike and a smooth low-frequency basis is maximally delocalised.
Artifact: `ddm_ph5o_rank6_generic_basis_20260803.json`.

Both refusals are now **executable**, not just written down — `descent_support_byte_cost(...,
shared_addresses=True)` refuses without a measured overlap above 0.5, and
`generic_basis_can_carry_descent` refuses a smooth family absent a fresh measurement. A future
arm cannot re-assume either escape.

---

## §5 What I refute in my own charter

1. **The charter's stated pose gap does not reproduce, and it is internally inconsistent.** It
   says *"Pose gap = 0.2120 S (29.2% of the 0.6189279 total)"*. But 0.2120/0.6189279 = **34.25%**,
   not 29.2%, and 29.2% of the total is **0.1807**, not 0.2120 — the two figures contradict each
   other. Neither reproduces from the live `report.txt`: the measured live pose gap is
   **0.1090381 = 17.62%**. **The charter's break-even survives** — its 2.20% reproduces to
   2.1984% against the *live* gap, so the number was computed correctly and only the label is
   stale. Name the baseline, not just the delta (`m46`).
2. **The pre-registered instrument does not exist at its pre-registered price.** "k=6 / 3,600 B"
   presumes a rank-6 generic-basis correction. That exact object is **empty** on this vehicle
   (100% zero coefficients, §4.2). The actuator that *does* carry the descent is address-based
   and costs ~2.35 B/pixel — so the break-even was set for an instrument that turns out to be
   inert, and the real instrument is priced in a different currency.
3. **I did not start with pu2's ranked tail, and starting there would have been the `m88` trap.**
   The tail is a skewed subset; its mean base d_pose differs from the population by ~3× (I
   watched exactly this on my own interim prefix: 81 rows read −8.29% at n=61 against a subset
   **3.19× harder** than the population). I measured **all 600** instead, which *subsumes* the
   tail — the top-N byte/benefit rows in §3 are the tail answer, computed exactly rather than
   extrapolated.
4. **`ddm_ph4`'s own canonical-equation module is an orphan** (`m55`):
   `src/tac/canonical_equations/ddm_ph4_blind_set_seg_free_pose_actuator_20260803.py` is not
   imported by `src/tac/canonical_equations/__init__.py`, so none of its four functions is
   reachable from the package surface. **Owed**, and named here so it is not rediscovered.

---

## §5.5 OWED, and why — the package wiring I deliberately did NOT land

**My module ships UNWIRED, and that is a choice, not an oversight.** I wired it into
`src/tac/canonical_equations/__init__.py`, then found that file already carried a **sister arm's
uncommitted edit** (the `ddm_cr1_seg_only_base_pose_degradation_20260801` import block, absent at
`HEAD`). Committing the file would have **absorbed another arm's in-flight work into my commit**
— the Catalog #314/#340 absorption anti-pattern. I restored `__init__.py` byte-for-byte to its
pre-my-edit state (`git diff` is again exactly cr1's 5 lines and nothing else) and dropped it
from this landing.

*Two things I also caused and undid:* my `ruff --fix` on that shared file re-sorted ten `__all__`
entries belonging to other arms and auto-added cr1's three names — churn on a hot shared file
that would have ridden into someone else's next commit. **Do not run an autofix on a hot shared
file.** Both are reverted.

**OWED — paste-ready, ~30 seconds, for whoever next commits `__init__.py` legitimately:**

```python
from tac.canonical_equations.ddm_ph5o_blind_descent_is_address_limited_20260803 import (
    break_even_d_pose_cut,
    descent_support_byte_cost,
    generic_basis_can_carry_descent,
    joint_delta_s_of_support,
    shared_address_pricing_is_admissible,
)
```

Until then the module is reachable by its full path
(`tac.canonical_equations.ddm_ph5o_blind_descent_is_address_limited_20260803`), which is how the
numbers in §2–§3 were cross-checked. **I am calling my own module an orphan (`m55` grade) rather
than buying reachability with a sister's bytes.**

*Side effect worth recording:* `__init__.py`'s broad reachability is what made the pre-commit
CI-blind hook select **36 MLX tests** for this landing, one of which (`test_compact_renderer_mlx_
spine_runner.py` inside a 237-node combined session) **hard-crashes with a `Bus error` in
`src/tac/substrates/_shared/mlx_score_aware/adapter.py::_score_aware_loss_part_metrics`**. That
module passes **309/309 solo**, and neither it nor the adapter references `canonical_equations`
or `ddm_ph5o` at all — so it is a **combined-session** crash, not a consequence of any change
here. It is a live false-blocker (the `#828` class) owed to whoever owns the MLX suite;
reproduction log: `/Volumes/VertigoDataTier/pact/ddm_ph5o_20260803/mlxfull.log`. **No gate was
bypassed in this landing** — dropping `__init__.py` takes the selection to zero honestly.

---

## §6 The SPEC — what would make this live (a dominated row is a specification, not a kill)

The actuator is **not retired**. Its physics is sound and its seg-freedom is exact and rare. It
fails on **one** axis, and the axis is nameable:

> **Required: a 3.5× reduction in the cost of NAMING which blind pixels to move** (or,
> equivalently, a 3.5× larger pose gain at the same support). 3.5× is the ORACLE figure; the
> best *shippable fixed* support needs **5.6×**. The exact factor per support is
> the `byte_cut_factor_needed` column in §3 and is emitted by
> `joint_delta_s_of_support(n)["byte_cut_factor_needed"]`.

Three directions that would attack the *address* cost rather than the physics, none of them
measured here:

- **A generic saliency the decoder can recompute.** The address is expensive only because it is
  scorer-derived. If a *generic* predictor of the top-|g| pixels — computable in `inflate.py`
  from the decoded frames alone, with no scorer weights — reproduced even a modest fraction of
  the ranking, the address collapses to a rank index within a generically-derivable candidate
  set. The measured hook: a consensus set covers 12.5% of per-pair support, so a generic proxy
  needs to beat that by a lot to matter.
- **Joint solve with the 11 pose knobs.** Everything here holds the shipped knobs FIXED, which is
  the realised receiver chain. A joint re-solve is unmeasured and could shift the gain — it would
  *not* shift the addressing arithmetic.
- **A localised generic basis.** The rank-6 DCT is empty because it is delocalised. A generic
  *wavelet/curvelet* dictionary is localised and still deterministically generable — the same
  rule-118 free lunch with the right support geometry. This is the one direction that could
  attack both terms at once.

---

## §7 Honest limits

- **`[macOS-CPU advisory]`, not contest-CPU.** `d_pose` here is a frozen-PoseNet forward on our
  own frames through the canonical `d_pose_u8` path, at n600 — the same instrument the pj2/pu2
  solver used. It is not `upstream/evaluate.py`, and nothing here is a score claim.
- **The gradient is a linearisation with an STE through `_to_uint8`.** It is used only to *choose*
  a direction; every reported `d_pose` is a real forward evaluation on a real uint8 frame.
- **Pose knobs held fixed** (see §6). This is the realised chain, not necessarily the best one.
- **The sharing probe is 16 strided pairs, not n600.** A Jaccard of 0.0056 with 180/192 distinct
  pixels is not a marginal call, but it is a 16-pair measurement and is labelled as such.
- **The C7 SegNet control is 4 pairs × 3 amplitudes.** The D-identity that implies it is exact and
  was checked on *every* treated frame at n600; C7 is the end-to-end confirmation on a sample.

---

## §8 Artifacts

| artifact | what |
|---|---|
| `.omx/research/ddm_ph5o_n600_support_sweep_20260803.json` | **the n600 row** — per-pair sweep, per-support aggregates, full byte arithmetic |
| `.omx/research/ddm_ph5o_rank6_generic_basis_20260803.json` | the rank-6 generic-DCT solve (100% zero coefficients) + all 7 controls |
| `.omx/research/ddm_ph5o_support_sharing_20260803.json` | the shared-address refutation (Jaccard 0.0056) |
| `.omx/research/ddm_ph5o_c7_segnet_argmax_20260803.json` | C7 end-to-end SegNet argmax identity + its positive control |
| `experiments/ddm_ph5o_blind_pose_solve.py` | controls, adjoint, T1 aimed step, T2 generic-basis GN |
| `experiments/ddm_ph5o_blind_pose_sparsity.py` | the support sweep + byte arithmetic |
| `src/tac/canonical_equations/ddm_ph5o_blind_descent_is_address_limited_20260803.py` | the law, with both escapes made un-assumable |

Bulk per-pair JSONL rows stay on the SSD tier at
`/Volumes/VertigoDataTier/pact/ddm_ph5o_20260803/` (rebuildable from the committed instruments +
the custodied `pu2` archive; the committed JSON summaries carry every number this memo cites).
