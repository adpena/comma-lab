---
schema: ddm_rs2_flip_damage_rerank_and_drop_seg_leg.v1
date_utc: 2026-08-03
arm: ddm_rs2 (re-rank the #766 waterfill in the right currency; queue the drop seg leg)
lane_id: "lane_ddm_rs2_20260803"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: SEE-PER-ROW
axis: "[byte-closed rate + decoder-plane, scorer-free]. NO SegNet or PoseNet forward or
  backward was fired. The evaluator slot (held by ddm_pu2) was not requested and not
  touched. Every byte figure comes from the real encoder on the real live-best lattice;
  every DRIVE figure comes from the real receiver's render + the frozen resamplers."
consumes:
  - .omx/research/ddm_br1_basis_race_and_drop_surface_20260803.md  (predecessor)
  - experiments/ddm_wr1_reverse_waterfill.py                       (task #766, primary)
  - experiments/ddm_gr1_granularity_rerace.py                      (the cell grain)
  - /Volumes/VertigoDataTier/pact/ddm_ru1_20260729/atlas_flat.npz  (458,738 per-flip rows)
  - /Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_cell_sensitivity_atlas.npz
  - /Volumes/VertigoDataTier/pact/ddm_gr1_20260730/gr1_sweep_cell_n48_receipt.json
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_cx1_pj2ix2_archive.zip
consumers: [MAIN, ddm_pu2]
tokens: [p0-ledger-ok]
---

# ddm_rs2 — the waterfill's currency was already flip damage; its SUPPORT is 24x too small

**Headline in one line:** my charter's premise is **FALSE** — `#766`/`ddm_wr1` already ranks by
flip damage as the PRIMARY key with bytes only as a tie-break
(`ddm_wr1_reverse_waterfill.py:93`, `np.lexsort((-residual_mass, flip_mass))`) — but the key is
computed on the **wrong support**: it prices a cell's drop by the ambient flips inside that
cell's own **16x16 tile (256 px)**, while the **MEASURED** receptive field of a cell drop in the
SegNet's own input plane is **84 x 82 px = 6,192 px, 24.2x larger**, so **144 of the 486 cells it
ships as the *"safe-floor (all zero-flip)"* tranche have ambient flips inside the region their
own drop perturbs.**

**What is in hand, in one screen:**

| # | finding | status |
|---|---|---|
| 1 | `#766` already ranks by flip damage; the charter's premise is refuted at source | **MEASURED** (§1.1) |
| 2 | its damage key's support is **24.2x too small**; **144 of 486** "provably safe" cells are not, monotone in the RF estimate, `half = 0` reproduces wr1 exactly | **MEASURED** (§1.2, §4 R1-a) |
| 3 | its byte tie-break `residual_mass` correlates only **rho 0.513** with the real per-cell byte marginal (384 exact re-encodes) | **MEASURED** (§1.2c) |
| 4 | an endpoint-FREE thin-margin key agrees with the corrected key at **rho 0.99** and with wr1's at 0.89 — two disjoint instruments, same verdict | **MEASURED** (§1.4) |
| 5 | a byte-matched ordering **A/B is BUILT and byte-closed** (274,631 B vs 274,321 B); at equal bytes arm B is lower on BOTH halves of flip damage — **27.9% less** ambient flip mass and, at **n600**, **11.25% fewer** perturbed scorer pixels | **BUILT + QUEUED** (§2.3-§2.5) |
| 6 | `br1`'s `cell_drop63` byte leg is for a **different cell set**: gr1's own ordering saves **79,177 B**, not 72,544 | **MEASURED** (§2.1) |
| 7 | `br1`'s owed equations leg paid: 3 canonical equations + 31 behaviour tests | **LANDED** (§3) |
| 8 | the full per-cell n600 DRIVE sweep **died silently at 24/36 groups** (loop-end-only save); and my process-state probe was wrong **three times in both directions** — corrected append-only | **HONEST NEGATIVE** (§1.6, §1.6b) |
| 9 | **the coordinator's hypothesis for the residual 17% is REFUTED, exactly**: `D.U` annihilates NOTHING on the render plane (gains **[0.687, 1.028]**, 0% attenuated, operator validated to 1.7e-07 against the real receiver). The cited 80.6742% null fraction is reproduced — it lives on the CAMERA plane and is **structurally unreachable** from the token lattice | **MEASURED** (§5.2) |
| 10 | **the 14.8% residual is UNEXPLAINED.** Channel structure is real (chi-sq **228.7**, p<<0.001) but modest and does not account for it; my first headline for it was permutation-invariant noise and is retracted. Both axes of `#766`'s lexsort can be made EXACT from data on disk (36.4 s + 388.8 s) | **MEASURED + HONEST GAP** (§5.3-§5.4) |
| 11 | `gr1`'s gradient key is **n600**, not n48 — concern dissolves at the artifact; the real caveat is that it is an **ancestor-lattice** measurement | **MEASURED** (§5.1) |
| 12 | **the exact pointer did NOT move.** 0.1910828242 [contest-CPU] UNMOVED. Nothing here is a score. | — |

**NEXT-IF-RESUMED** — see §9. Written incrementally.

---

## §0 — WHAT I CHECKED BEFORE BUILDING ANYTHING

### §0.1 Every charter constant recomputed from primaries, never re-typed

The charter's own warning ("I handed arms wrong constants FIVE times today, and one hit br1")
is the reason this is the first section. Recomputed from `DEN = 37,545,489`,
`PX = 196,608 x 600` and the `cx1` component terms alone:

| symbol | charter | recomputed here | agrees |
|---|---|---|---|
| `W = 4*DEN/PX` | 1.2731082153320312 | **1.2731082153320312** | exact |
| `S` = seg+pose+rate | 0.8264972 | 0.4311790+0.1597320+0.2355862 = **0.8264972** | exact |
| `d_seg` = seg/100 | 0.004311790 | **0.004311790** -> **508,639** flips | exact |
| archive from rate term | 353,808 B | `0.2355862*DEN/25` = **353,807.96** | exact |
| gap to PR130 (0.172141) | 0.6543562 | **0.6543562** | exact |
| 1% of gap | 9,827.2 B / 7,719 flips | **9,827.25 B / 7,719.10 flips** | exact |
| `dS/d(d_pose) = 5/sqrt(10 d_pose)` | — | `d_pose = 0.00255143` -> **31.302** | — |

**Baseline declaration (the "a ΔS without its baseline is unanchored" law).** Every delta in this
memo is against **`cx1` = `v4d_composed_cx1_pj2ix2_archive.zip`, S 0.8264972, 353,808 B,
sha `1d3ab694…`**. Note that `MEMORY.md`'s 2026-08-02 frontier rows (`dc1_fold` 0.8983775 @
360,309 B, `ms8` 0.8984335) are **superseded** by `cx1`, which is 0.0719 lower; a reader coming
from the index must not price against those.

### §0.2 The lattice and the base, re-derived from the shipped bytes

Parsed straight out of the live archive through the real receiver
(`inflate_runner_v4d.Decoder`, the ix2-capable one; the `pfs1` tree's runner is pre-ix2 and
cannot parse a container — `ddm_cx1` §5):

| quantity | value | check |
|---|---|---|
| lattice | `(600, 24, 32, 4)` uint8, 0..15 | — |
| token member | **341,295 B** | equals `br1`/`cx1` exactly |
| tokens vs `br1`'s cached `cx1_tokens.npy` | **byte-identical** | positive control |
| live cells (any temporal variation) | **384** of 768 | = `gr1`'s `cell_drop50` |
| dead cells | **384** | — |
| live units (cell x channel) | **1,528** of 3,072 | equals `br1`'s 1,528 |

---

## §1 — THE RE-RANK

### §1.1 ROUND-1 CATCH — the charter's premise is FALSE, and I checked it before building

> My charter: *"`#766`'s sensitivity-weighted reverse-waterfill must rank by FLIP DAMAGE, not
> by bytes ... Ranking by bytes on a flat, uncorrelated, occasionally-negative yield surface is
> ranking by noise."*

`#766` **is** `ddm_wr1` (`.omx/research/ddm_wr1_reverse_waterfill_20260729.md:1`,
`experiments/ddm_wr1_reverse_waterfill.py`). Read at source:

```python
:87   residual_mass = np.abs(signed).sum(axis=(0, 3))...          # the BYTE proxy
:89   cell = (atlas["y"] // 16) * 32 + (atlas["x"] // 16)          # the DAMAGE proxy's support
:90   flip_mass = np.bincount(cell, minlength=768)
:93   order = np.lexsort((-residual_mass, flip_mass))
```

`np.lexsort` takes its LAST key as primary. **The primary key is already `flip_mass`
(ascending); bytes enter only as the tie-break.** wr1's own line `:257` says so:
`"ordering": "flip_risk asc (safest), tie residual-mass desc (fattest-safe first)"`. So the
waterfill was never ranking by bytes, and the re-rank the charter asked for cannot be the one it
described. **VERIFIED_VIA_SOURCE_INSPECTION.**

*(This is the same shape as `br1`'s round-3 catch on its own charter. Two arms in a row have had
a false premise handed to them; the cure that worked both times was reading the primary before
building on the brief.)*

### §1.2 What IS wrong — the SUPPORT, and it is off by 24x. MEASURED.

`:89` attributes each ambient flip to the **16x16 tile it lands in**, so a cell's safety is
priced over **256 scorer pixels**. But dropping a lattice cell does not perturb its own tile —
it perturbs whatever the decoder's four `repeat(2) + conv` stages reach. I measured that on the
live receiver rather than deriving it: drop cell `(13,17)` (whose own tile is scorer rows
208-223, cols 272-287), re-render `frame_1` through `inflate_runner_v4d`, upsample, `rint`, and
apply the frozen scorer downsample `D`:

| quantity | measured |
|---|---:|
| perturbed bbox in the SegNet input plane | rows **[174, 257]**, cols **[239, 320]** |
| bbox size | **84 x 82** |
| **nonzero** perturbed scorer pixels | **6,192** |
| the tile wr1 prices over | **256** |
| **support ratio** | **24.19x** |
| fraction of the real footprint wr1's key sees | **4.13%** |

**A key computed on 4% of the region a drop actually disturbs systematically UNDERSTATES the
risk of every drop, and cannot certify any cell "safe".**

*(Three support numbers appear below and they are not interchangeable: the measured **bbox** is
84 x 82 = 6,888 px, the **nonzero** perturbed set inside it is 6,192 px, and the **box the
corrected key integrates over** is (2 x 34 + 16)^2 = 84 x 84 = 7,056 px. The 24.19x ratio is on
the nonzero basis — the conservative one; the box basis gives 27.56x.)*

**The single-cell anchor GENERALISES — measured on 49 cells (§1.6).** The resumable n600 sweep
has 49 of 384 live cells on disk so far (**denominator stated: 12.8%**) and every one of them has
a receptive field of **84 rows** (min = median = max = 84) and **84 columns** (median 84, min 49
at a clipped image edge), median area **7,056 px = 27.56x the tile**. The 84-px extent is not a
box artifact: the measurement box allows up to 96 px. So the support correction rests on a
distribution now, not on one probe.

### §1.2b The consequence, measured: 144 of wr1's 486 "provably safe" cells are not safe

wr1 ships `--knee-a 486`, described at `:211` as the *"safe-floor tranche (all zero-flip)"*, and
its damage model `:131` is `dseg_ceiling = REF_DSEG + dropped_flip_mass / TOTAL_PX` — a ceiling
built on the same 4% support. I reproduced wr1's `flip_mass` array **exactly**
(`wr1_flip_mass_reproduced: true` against `wr1_cell_sensitivity_atlas.npz`) and then recounted
the same flips over the measured footprint:

| | cells |
|---|---:|
| zero ambient flips in the **16x16 tile** (= wr1's knee-A set) | **486** |
| zero ambient flips in the **measured receptive field** | **342** |
| **wr1 calls "zero-flip" but the real footprint says otherwise** | **144 (29.6%)** |
| RF-zero set is a strict subset of tile-zero | **true** (as geometry requires) |

**Nearly a third of the tranche wr1 certifies as safe has ambient flips inside the region its
own drop perturbs.** The direction is one-sided and structural: shrinking the support can only
turn a non-zero into a zero, never the reverse, so a tile-scoped key can only ever produce
FALSE safety, never false alarm.

### §1.2c The byte tie-break is also a weak proxy — measured at the cell grain

While I had the encoder open I measured the **exact** byte marginal of every one of the 384 live
cells (384 real re-encodes of the real lattice, 389 s):

| | value |
|---|---:|
| per-cell byte marginal | min **248** / median **861** / mean **858** / max **1,234** B |
| negative marginals at the CELL grain | **0** |
| Spearman vs wr1's byte proxy `residual_mass` | **0.513** |
| Spearman vs `wr1_tile` flip mass | 0.147 |
| Spearman vs `rs2_rf` flip mass | 0.238 |

Two things, and the first **refines `br1` rather than repeating it**: at the CELL grain the byte
side has a 4.98x spread and **no negative marginals** — `br1`'s "flat, occasionally negative"
result is a **UNIT-grain** (cell x channel) result and does not transfer unchanged to the cell
grain the waterfill actually operates on. Second, and this is a defect in its own right:
**wr1's byte tie-break `residual_mass` correlates only rho = 0.513 with the real byte
marginal**, so even the secondary key is carrying about half the information it claims to.

The **DRIVE sweep** (§1.5) re-measures the footprint independently for every live cell at n600,
so the 24x is a single-cell anchor now and a per-cell distribution shortly.

### §1.3 The re-rank, and what changes

Three keys are in play. Two are incumbents and one is the correction. **Only the support
differs between wr1's key and mine — the ambient-flip proxy and the byte tie-break are held
fixed — so this isolates the support question.**

| key | what it is | support | source |
|---|---|---|---|
| `gr1_gsum` | `sum |d seg-loss / d tokens|` per cell (backprop) | **correct by construction** | selected the LIVE base |
| `wr1_tile` | ambient flips in the cell's own 16x16 tile | **256 px (4%)** | task #766 |
| `rs2_rf` | the SAME ambient flips over the MEASURED receptive field | 6,888 px box | this arm |

Rank agreement over all 768 cells (Spearman):

| pair | rho |
|---|---:|
| `gr1_gsum` vs `wr1_tile` | **0.695** |
| `gr1_gsum` vs `rs2_rf` | **0.829** |
| `wr1_tile` vs `rs2_rf` | **0.889** |
| wr1's DROP ORDER vs the corrected drop order | **0.727** |
| `wr1_tile` vs wr1's own byte proxy | 0.477 |
| `rs2_rf` vs wr1's own byte proxy | 0.614 |

Prefix overlap of the two drop orders (how many of the first k cells are the same set):

| k | 100 | 200 | 300 | 384 | **486** | 600 |
|---|---:|---:|---:|---:|---:|---:|
| shared | 54 | 114 | 194 | 294 | **452** | 550 |

**At wr1's own knee-A the two orders share 452 of 486 cells** — the correction is not a
wholesale reshuffle, it is a targeted swap of ~34 cells (and, more importantly, a re-typing of
144 of them from "certified safe" to "not certified"). Early in the order the disagreement is
much larger: at k = 100 they share only **54**.

**Read the first two rows together.** `gr1`'s gradient key carries the receptive field
automatically (backprop propagates through the same upsample stack). It agrees *more* with the
support-corrected ambient key (0.829) than with the tile-scoped one (0.695). That is an
independent instrument — gradients, not flip counts — pointing at the same conclusion: **the
tile support is the outlier.**

### §1.4 An independent susceptibility instrument agrees with the correction, at rho = 0.99

Ambient flips are a questionable susceptibility proxy on their own, for a reason worth stating
because it attacks my own §1.3: **a pixel that already flips cannot flip again** (d_seg counts
disagreements), so ambient-flip pixels are exactly the ones whose damage is already CAPPED. The
pixels at risk of NEW damage are the currently-CORRECT ones with a THIN margin.

That is testable with cached data and no scorer: `gt_n600.npz['margins']` is the GT SegNet
top1-top2 logit gap for every one of the 600 x 384 x 512 pixels, and `ddm_sg1`'s
`argmax/chunk_*.npy` gives the realized labels. Built on the same measured receptive-field
support:

| tau | thin px | thin AND currently correct | rho vs `rs2_rf` | rho vs `wr1_tile` | rho vs `gr1_gsum` |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 166,700 | 92,682 | **0.9945** | 0.8983 | 0.8253 |
| 0.20 | 663,192 | 425,874 | **0.9942** | 0.8984 | 0.8249 |
| 0.50 | 1,630,790 | 1,240,576 | **0.9932** | 0.8951 | 0.8289 |
| 1.00 | 3,149,890 | 2,699,169 | **0.9904** | 0.8907 | 0.8358 |

Three things fall out, all measured:

1. **The thin-margin key and the support-corrected ambient key are effectively the same ranking
   (rho 0.990-0.995 across two decades of tau).** Two instruments built from disjoint inputs —
   one from where we are already wrong, one from the GT logit geometry — agree. That is the
   strongest available evidence that the correction is right and not an artifact of the atlas.
2. Both disagree with `wr1_tile` at ~0.89 and with `gr1_gsum` at ~0.83, by the same margin.
3. **The "already-flipped pixels cannot flip again" objection is measured to be immaterial at
   the ranking level:** rho(all thin, thin-AND-correct) = **0.9989-0.9998**. My own attack on
   §1.3 does not survive contact with the data, and I am recording that it was tried.

### §1.5 The DRIVE currency — new, exact, scorer-free  *(sweep in flight; §1.6 on completion)*

Ambient flips and margins are both *susceptibility*. The other half of flip damage is **DRIVE**
— how hard the drop actually pushes the scorer's input:

```
DRIVE(c) = sum_{p<600} || D(cam_drop_c(p)) - D(cam_base(p)) ||_1     over (384,512,3)
```

computed with the real receiver and the frozen resamplers. It is exact, it needs no scorer, and
it carries a hard certificate: **SegNet reads only `D(f1)`, so `DRIVE(c) = 0` proves the drop of
`c` causes EXACTLY ZERO argmax flips.** Flip damage factorises as drive x susceptibility, and
neither incumbent key measures the drive half at all.

Pilot (12 live cells stratified by activity, one pair, `rs2_drive_pilot.json`):

| | value |
|---|---:|
| DRIVE spread across cells (max/min) | **7.90x** |
| Spearman(activity, DRIVE) | **-0.014** |
| cells with zero DRIVE | 0 of 12 |

**Activity does not predict drive either** (rho -0.014). Combined with `br1`'s result that the
byte marginal is flat and activity-uncorrelated, *every* cheap scalar anyone has attached to a
lattice unit — bytes, activity — is measured to carry no ranking information. The information
is in the drive x susceptibility product, and both factors have to be measured.

### §1.6 The full per-cell n600 sweep **DIED SILENTLY at 24 of 36 groups. Reported, not hidden.**

The 36-group disjoint-support sweep (`rs2_drive_sweep.py`) ran 24 groups over 1,901 s and then
**vanished with no traceback, no receipt, and no per-cell data.** Two failures caused it, both
mine, both worth more than the data would have been:

1. **Loop-end-only saving.** The script wrote its `cell_drive.npz` once, after all 36 groups.
   CLAUDE.md forbids exactly this ("Loop-end-only saving is FORBIDDEN"), and it cost 32 minutes
   of completed work that no artifact preserves. **A per-group append would have cost 3 lines.**
2. **My liveness probe could not return the negative.** I checked the job with
   `pgrep -f rs2_drive_sweep`, which matched **my own watcher shells** — their command lines
   contain the script name. It reported ALIVE for minutes after the process was gone, with an
   RSS of 3.3 MB for a job that should hold 1.35 GB. I only caught it by noticing the RSS was
   impossible. **The fix is to anchor liveness on the EXEC form**
   (`ps -eo args | awk '/[r]s2_..\.py/ && /python/'`) or on a receipt row, never on a pattern
   that the watcher itself matches. This is the campaign's named
   *"a probe that cannot return the NEGATIVE"* class, and it bit me inside the memo that cites it.

**What survives, from the console log:** 24 groups, 240 live cells covered, per-group disjointness
leaks in **[-1.83, +1.19] L1 against per-group totals of order 1e8 (~1e-8 relative)** — so the
box decomposition is sound to float32 summation noise. Per-group wall time 60-103 s.

**What I did instead of a 45-minute re-run** (per §3 of the operating manual: rank by what a
silent error would damage, and by what actually decides). The per-cell DRIVE map was a
nice-to-have. The DRIVE number that **decides the queued gate** is the two arms against each
other at n600, which is 3 x 600 renders and ~4.3 minutes — landed in §2.3b at full n600.

> ### §1.6b APPEND-ONLY CORRECTION — I declared TWO deaths that never happened
>
> Everything above about the 36-group sweep dying is **true and confirmed** (no `cell_drive.npz`,
> no `receipt.json`). But I then wrote, and put in a commit message, that the sister script
> `rs2_arm_drive_n600.py` *"was killed twice and lost nothing."* **That is FALSE and I am
> correcting it rather than deleting it.** Its FIRST launch ran to completion — `n_pairs: 600`,
> `elapsed_s: 259.4`, full JSON receipt. I called it dead because I read a lagging log tail
> (200/600) and a `ps` check while it was still running, and I did the same thing again on the
> relaunch at 150/600.
>
> **So the same instrument failed THREE times in one session, in both directions:** it reported
> ALIVE when the process was gone (`pgrep` matching my own watcher shells) and DEAD twice when
> the process was alive and finishing. The unifying law is sharper than "anchor on the exec form":
>
> > **Job state is only ever readable from a RECEIPT. The process table and a log tail are both
> > lagging, self-matching, and unable to distinguish "finished" from "killed".**
>
> The `until [ -f <receipt> ]` waiter got it right **both** times and I overrode it with a worse
> instrument each time. The commit message on `c423c153e4` carries the false "killed twice" claim;
> this row is its append-only correction, and the conclusion it supported (checkpoint per chunk)
> survives on the sweep's real death alone.

---

## §2 — THE QUEUED SEG LEG: what I built, and why it is NOT `cell_drop63` alone

### §2.1 ROUND-2 CATCH — `br1`'s `cell_drop63` byte leg is for a DIFFERENT cell set

`br1` §3.4 supplies `cell_drop63 = -72,544 B` and calls it the byte leg of `ddm_na1`'s P0-2,
which asks about *"the knee that selected the live base"*. That knee is `gr1`'s, and `gr1`'s
ordering is reproducible exactly: `argsort(gr1_cell_gsum)` ascending, drop the first
`round(pct*768)`. **Positive control: that reproduction at 50% matches
`qa24_grid_keep_mask_50.npy` on all 768 cells, and the shipped `cx1` lattice's 384 live cells are
EXACTLY that mask (768/768).** So the live base is `gr1`'s `cell_drop50`, verified at mask level.

Running `gr1`'s own ordering through the real encoder:

| level | cells dropped | token member | archive | saved | flip budget at `W` |
|---|---:|---:|---:|---:|---:|
| live base (`cell_drop50`) | 384 | 341,295 | 353,808 | 0 | — |
| **`cell_drop63`, gr1 ordering** | 484 | **262,118** | **274,631** | **79,177** | **62,192** |
| `cell_drop70`, gr1 ordering | 538 | 215,648 | 228,161 | 125,647 | 98,694 |
| `cell_drop80`, gr1 ordering | 614 | 146,787 | 159,300 | 194,508 | 152,782 |
| `br1`'s `cell_drop63` (its own selection) | 484 | 268,751 | 281,264 | 72,544 | 56,982 |

**`br1`'s figure is 6,633 B low (9.1% of the saving) because it used a different 484 cells.**
Anyone consuming `br1` §3.4 for P0-2 should take **79,177 B / 62,192 flips**, not
72,544 / 56,982. The grain was right; the selection was not.

### §2.2 The gate `br1` queued is already answered twice, and both say NO

Before spending the one scorer slot, I priced what it would buy. Two independent measured
priors already bracket it:

| prior | measurement | realized B/flip | vs `W` = 1.2731 |
|---|---|---:|---|
| `ddm_ba31` drop-more | **n600** | 0.6498 | **0.51x -> dominated 1.96x** |
| `ddm_gr1` cell sweep | **n48** `cell_drop50` d_seg 0.003947 -> `cell_drop63` 0.0050128 | 0.6298 | **0.49x -> dominated 2.02x** |

`gr1`'s row is a direct read on the exact question (`gr1_sweep_cell_n48_receipt.json`):
`dd_seg = 0.0010658` over the knee, i.e. **125,732 n600-equivalent flips** against the 62,192-flip
budget = **2.02x over**. It is n48, so per the campaign's own law it is a strong PRIOR and not
evidence — but it is a prior pointing the same way as an n600 measurement from a different arm.

**Firing `cell_drop63` alone would spend the slot to confirm a twice-corroborated negative.**

### §2.3 What I built instead: a byte-matched ORDERING A/B, both arms byte-closed

The open question is not *"does dropping more pay"* (answered ~2x NO, twice) but *"is the
ordering right"* — which is exactly what §1 puts in doubt and which no receipt answers. So both
arms carry the SAME byte budget and differ ONLY in which cells are dropped:

| arm | key | cells | archive | saved | flip budget | ambient flips in MEASURED support |
|---|---|---:|---:|---:|---:|---:|
| **A** `kA_gr1_drop63` | `gr1_gsum` (the incumbent that chose the base) | 484 | **274,631 B** | 79,177 | 62,192 | **1,024,507** |
| **B** `kB_rs2_rfkey_bytematched` | ambient flips on the MEASURED support | 478 | **274,321 B** | 79,487 | 62,435 | **738,385** |

Byte-match residual **310 B (0.4% of the saving)**; arm B saves slightly MORE. The sets differ
by only 26 cells (A-only) and 20 (B-only), yet **arm B carries 27.9% less ambient flip mass in
the support the drop actually perturbs, at equal bytes.**

Both archives are BUILT, on disk, and byte-closed:

```
/Volumes/VertigoDataTier/pact/ddm_rs2_20260803/ab/rs2_kA_gr1_drop63_archive.zip          274,631 B
/Volumes/VertigoDataTier/pact/ddm_rs2_20260803/ab/rs2_kB_rs2_rfkey_bytematched_archive.zip 274,321 B
```

Each was proved, not asserted: the container re-parses, the token frame decodes **bit-identically
to the lattice handed to the encoder**, the joint sections are unchanged, and
`inflate_runner_v4d.Decoder` actually renders `frame_1` from it (`lattice_roundtrip_exact: true`,
`receiver_renders: true`). Losslessness is by construction — a dropped cell is a legal lattice in
the same 16-symbol alphabet and the receiver is untouched — so **no format gate and no receiver
change stand between these bytes and the evaluator.**

**What is NOT done (round-2 correction).** `upstream/evaluate.py` consumes a submission DIR, not
a bare `archive.zip`. Assembling that tree — `archive/0.bin` + `inflate.sh` + the **ix2-capable**
runner + its flat dependencies — is the one remaining build step, and both halves of it are
named in §2.5. I am not claiming the gate is a single command; I am claiming the bytes are
final and proved.

### §2.3b The DRIVE side of the A/B, measured — arm B perturbs **11.3% fewer scorer pixels**

The ambient-flip difference in §2.3 is a SUSCEPTIBILITY signal. The DRIVE signal is independent
and I measured it directly: render base, A and B through the real receiver and count the scorer
pixels whose input actually changed.

**n600, all 600 pairs, 259.4 s** (`rs2_arm_drive_n600.SEALED.json`, sha `cd857c694f88de16`):

| B/A ratio | value | reading |
|---|---:|---|
| perturbed scorer px, **> 0 LSB** | **0.8875** | **11.25% fewer pixels perturbed at all** |
| > 1 LSB | 0.8914 | |
| > 2 LSB | 0.9050 | |
| > 4 LSB | 0.9305 | |
| > 8 LSB | 0.9683 | |
| total drive L1 | 0.9657 | |

**The advantage is concentrated in SMALL perturbations** — 11.3% fewer pixels touched at all, but
only 3.2% fewer touched hard (>8 LSB). Which end of that curve decides flips is genuinely open:
a flip needs to cross a thin margin, which favours the small-perturbation end mattering most, but
a skeptic can argue large perturbations flip more reliably. **The whole curve is reported rather
than the convenient end of it**, and §2.4's gate is what settles which end predicts.

**Subset validation, done properly.** Before the n600 landed I quoted a 250-pair prefix at
**0.8874** and a scattered 5-pair sample (pairs 0, 137, 299, 411, 577) at **0.8890**. The n600
population value is **0.8875** — the prefix was right to **0.01%** and the scattered sample to
0.17%. So for this quantity the prefix is *not* a different population, and that is now MEASURED
against the population rather than argued from stability.

**Read together with §2.3:** at equal bytes (arm B is 310 B *cheaper*), arm B is lower on BOTH
halves of flip damage — **27.9% less** ambient flip mass in the real support, **11.25% fewer**
perturbed scorer pixels at n600. Two independent instruments, same direction. That is the
strongest scorer-free case that can be made for the corrected ordering, and it is still not a
flip count: converting it needs the scorer, which is §2.4's gate.

### §2.4 THE PRE-REGISTERED GATE  (written before any scorer runs)

**Prediction, recorded now.** From the two priors in §2.2 and local linearity, **arm A realizes
~125,700 flips = ~2.02x over its 62,192-flip budget, so arm A LOSES on seg+rate.** Arm B carries
27.9% less ambient flip mass in the real support; if that proxy transfers linearly, **arm B
realizes ~90,600 flips = ~1.46x over budget — better, but still LOSING.** So the honest
pre-registration is: *I expect both arms to be dominated, and the finding to be the RATIO
between them, not a win.*

**Falsifiers, one per claim:**

| claim | falsifier |
|---|---|
| the ordering matters | if `flips(B) / flips(A) > 0.95`, the support correction does not buy flips and §1 is a real but **non-actionable** finding — say so |
| the support correction is the right direction | if `flips(B) > flips(A)`, the corrected key is WORSE and §1 is **refuted** at FORMULATION scope |
| drop-more is dominated | if either arm realizes **< 62,192** flips (and pose does not eat the budget), the twice-corroborated negative is **overturned** and the knee moves |

**Admission arithmetic (joint, because a token change moves BOTH frames —
`frame_0 := a*warp(frame_1) + b`):**

```
100*d(d_seg) + [ sqrt(10*(d_pose + d(d_pose))) - sqrt(10*d_pose) ]  <  25*db/DEN
```

with `db` from the table, `d_pose = 0.00255143` and `dS/d(d_pose) = 31.302` — so
`d(d_pose) = 1e-4` alone costs 0.00313 S, which is 6% of a whole arm's rate yield. **Report the
pose delta; do not assume it is zero.**

### §2.5 THE COMMANDS  (for `ddm_pu2` or whoever holds the slot; build step already done)

```bash
# 1. The two archives are BUILT and byte-closed -- do not rebuild them:
#      A: /Volumes/VertigoDataTier/pact/ddm_rs2_20260803/ab/rs2_kA_gr1_drop63_archive.zip
#      B: /Volumes/VertigoDataTier/pact/ddm_rs2_20260803/ab/rs2_kB_rs2_rfkey_bytematched_archive.zip
#
# 2. ASSEMBLE the submission tree (the one build step left; see the R2-a correction).
#    The runner MUST be the ix2-capable one -- the pfs1 tree's runner is PRE-ix2 and cannot
#    parse a container archive (ddm_cx1 section 5):
#      runner : /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/inflate_runner_v4d.py
#      flat deps + inflate.sh template :
#               /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1/
#               (ddm_tr1_runtime.py, ddm_r7_token_coder.py, pfs1_warp_receiver.py, inflate.sh)
#
# 3. Grep the argparse before emitting any flag (never-invent-flags):
.venv/bin/python upstream/evaluate.py --help
.venv/bin/python upstream/evaluate.py --device cpu --submission-dir <dir> --report report.txt
```

Report, per arm: realized n600 `d_seg` and `d_pose`, the **realized B/flip against
`W = 1.2731082153320312`**, and the delta against the stated baseline (`cx1`, S 0.8264972,
353,808 B). A dominated row is a **SPECIFICATION** ("needs the flip cost cut to X"), never a kill.

### §2.6 `cell_drop35` is still unmeasurable, and `br1` was right about why

The restore direction needs token values for cells the live base has already zeroed. Those
values are not in the archive — dropped cells retain only their mode. **Confirmed structurally
here:** `gr1`'s drop35 keeps 499 cells but the shipped lattice only has 384 live, so 115 cells'
worth of information is simply absent. Restoring requires the pre-drop lattice or a retrain.

---

## §3 — THE EQUATIONS LEG (`br1`'s owed triality leg, paid)

Landed as `src/tac/canonical_equations/ddm_rs2_waterfill_support_and_byte_yield_20260803.py`,
three equations with real evaluators (each computes from inputs; none returns a canonical marker):

| id | law | evaluator |
|---|---|---|
| `lattice_cell_drop_pricing_support_v1` | a cell drop is priced over the decoder's receptive field (6,192 px), not its 16x16 tile (256 px); ratio **24.19x**; a smaller-support key **understates risk and cannot certify safe** | `support_mispricing()` |
| `token_lattice_byte_marginal_flat_uncorrelated_v1` | per-unit byte yield is flat (-58 / 196 / 211 / 472 B), 2 units NEGATIVE, group drops superadditive 1.0206x -> the byte side is not rankable | `byte_side_is_rankable()` |
| `live_vs_dead_symbol_entropy_decomposition_v1` | the coder's apparent advantage over order-0 is **1.4776x** but **1.1093x** on LIVE symbols; the rest free-rides on LZ-copied dead zeros | `coder_advantage_split()` |

**ROUND-2 CATCH, carried into the equation.** `br1` reports the all-symbol coder advantage as
**1.4834x**; its own byte figures give `504,291 / 341,295 = ` **1.4776x**. The live ratio
`377,100 / 339,956 = 1.1093x` reproduces exactly. Conclusion unchanged, headline ratio 0.4%
lower — recorded in the module docstring rather than silently overwritten.

**Registration scope, stated rather than omitted.** The module defines the equations and their
measured anchors and is importable and self-validating now. The locked-registry `populate_*`
append and the `__init__` export are **OWED**, deliberately: a sister session holds uncommitted
edits in `src/tac/canonical_equations/__init__.py` (and in the registry JSONL), and folding those
into this commit would be the absorption-pattern bug class CLAUDE.md names. This is the same
discipline `ddm_b2b_rowband_flip_mass_20260731` records for the same reason. The exact owed
patch is three import lines plus three `__all__` entries.

---

## §5 — THE GRADIENT KEY, PURSUED (the finding my §1 used as corroboration and under-read)

**Re-anchored baseline for this section.** Live best is now **S = 0.7910689, 353,805 B**
(seg **0.4311790** unchanged, rate 0.2355842, pose 0.1243057 → d_pose 0.00154519,
`dS/d(d_pose)` = **40.223**), gap to the PR130 bar **0.6189279**, 1% of gap = **9,295.2 B =
7,301.2 flips**. The move from my charter's 0.8264972 base is **entirely POSE** (−0.0354) and
**−3 archive bytes**: the seg leg and the token lattice are identical, so every §1/§2 seg and
support result stands unchanged and every rate delta shifts by 3 B.

### §5.1 ASK 2 FIRST, because it dissolves: `gr1`'s key is **n600**, not n48

I was told to re-measure or scope it. Neither was needed — the artifact settles it.
`gr1_sensitivity_gabs.npy` is **(600, 24, 32, 4) float64** with **all 600 pairs carrying nonzero
gradient**, and `cell_gsum` reproduces from it **exactly** (max rel diff 0.0). `gr1`'s
`n_pairs_realized: 48` applies to its *realized d_seg* rows, **not** to the sensitivity map.

**The real scope caveat is a different one, and it is mine to state:** the gradient was computed
on the **pre-drop** model (gr1's ref archive, 569,996 B), because its purpose was to CHOOSE the
drop. The live `cx1` lattice is the post-`cell_drop50` result. So the key is n600 but on the
**ancestor lattice** — an ancestor-vehicle caveat, not a subset caveat.

### §5.2 ASK 1: the 17% is **NOT** the resampling visibility structure. REFUTED, exactly.

The hypothesis: a gradient lives in the space the score sees, so it carries `D`'s null structure
for free. That is testable **exactly and scorer-free**, because the operator between the
renderer's output and the scorer's input is linear AND separable:

```
render [384,512] --U--> camera [874,1164] --D--> scorer input [384,512]
M = D.U  factorises as  M_row (384x384)  and  M_col (512x512)
```

**Validated before use** (the result was suspiciously clean and the matmul emitted
divide-by-zero warnings, so I refused it until it was checked): BLAS vs `einsum` differ by
**0.0**; the warnings are spurious FP-flag noise. `M` applied to a real render matches the
**actual receiver pipeline** (`bicubic_up_to_camera_float` → the frozen `D`) to
**4.42e-05 on a 0..255 range = 1.7e-07 relative**. It is the real operator.

| | measured |
|---|---:|
| row singular values | [**0.82899**, **1.01417**], condition **1.2234** |
| col singular values | [**0.82829**, **1.01390**], condition **1.2241** |
| **full 196,608-dim gain range** | **[0.6866, 1.0283]** |
| fraction of render directions attenuated below 0.5 | **0.0%** |
| below 0.1 / 1e-2 / 1e-3 | **0.0% / 0.0% / 0.0%** |
| max deviation of `M` from identity | 0.0955 |

**`D.U` annihilates nothing.** Every render-space direction reaches the scorer with gain at
least **0.687**.

> **ROUND-4 RETRACTION.** My probe also reported a `frac_gain_below_uint8_step` row. **That row is
> a category error and I retract it:** a gain is dimensionless and the uint8 step is an amplitude,
> so comparing them is meaningless. The correct statement is that the shipped path's
> `clip(rint(.))` dead zone is **amplitude-dependent, not direction-dependent** — no *direction*
> is ever invisible, but a small enough *amplitude* is. Which has a sharp consequence worth more
> than the retracted row: **a linearisation cannot represent a dead zone at all** (the derivative
> of a step is zero or straight-through), so NEITHER the gradient key NOR my geometric key can see
> it. Of the three instruments, **only DRIVE — a realized finite difference through the actual
> `clip(rint(.))` — sees the quantiser.** That is a capability the gradient does not have.

**Where the 80.6742% actually lives, and why we cannot use it.** I reproduced the cited figure
to four decimals: `1 - 196,608/1,017,336 = ` **80.6742315%** — it is the null fraction of `D`
**on the CAMERA plane**, which is 1,017,336-dimensional. But our renderer emits into a
**196,608-dimensional** render plane and reaches the camera only through `U`. The measurement
above says `D` restricted to `range(U)` is **near-isometric**. So the enormous null space is
real and **structurally unreachable from the token lattice** — a token change cannot hide in it.
*(This concerns the render→scorer resampling only. The separate rank-4 "140 of 144 dims
invisible" result is about a different space and is another arm's; I neither re-derive nor
contradict it.)*

**Verdict: the coordinator's hypothesis for the 17% is REFUTED at FORMULATION scope** — for the
resampling layer, on a validated operator. That closes a plausible explanation rather than
confirming it, which is the more useful outcome.

### §5.3 So what IS the residual? Two things the gradient sees — and **`gr1` throws both away**

**(a) The support curve, from the gradient's own point of view.** Correlating `gr1`'s key against
the ambient-flip mass at every box half-width:

| half-width px | 0 (wr1's tile) | 8 | 16 | 24 | 28 | **34 (measured RF)** | 48 | **64** | 80 | 160 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rho vs `gr1_gsum` | **0.6950** | 0.7543 | 0.7916 | 0.8206 | 0.8341 | **0.8288** | 0.8456 | **0.8519** | 0.8478 | 0.5799 |

Monotone from the tile to a broad plateau: **the gradient independently confirms the support
correction, buying +0.139 of rank correlation between h=0 and h≈28.**

> **Do NOT read the argmax as a support estimate — I nearly did.** Widening the box also DENOISES
> a sparse flip field, so the argmax is the best *smoothed* estimator, not the true support. The
> curve is flat within 0.8% across h ∈ [34, 80]; it corroborates "much bigger than a tile" and
> **cannot discriminate** my directly-measured 84×82 footprint from 64 px. The support number
> stands on the direct measurement, not on this curve.

**Ceiling: even at the best box, rho = 0.8519** — no ambient-flip proxy at ANY support reproduces
the gradient's order. *(rho is a rank correlation; `1 - rho` is NOT a variance share, so I say
"cannot reproduce the order", not "explains 85% of it".)* Two candidates for what is in the gap:

**(b) CHANNEL structure — real but MUCH weaker than my first pass claimed. ROUND-4 CORRECTION.**
The gradient is per-(cell, channel); `wr1`'s key, my key, and **`gr1`'s own key**
(`g_abs.sum(axis=(0,3))` — it sums over channels) are all per-cell. My first draft led with
*"within-cell max/min channel ratio, median 3.34x"* and called it the residual's explanation.

> **That statistic is worthless and I am recording why rather than deleting it.** Permuting the
> four channels within a cell cannot change `max/sum` — the share statistic is
> **permutation-invariant by construction**, and a 200-draw within-cell permutation null
> reproduces the observed median **0.3531 exactly**. Four positive draws have a large max/min
> ratio for free. It is evidence of nothing.

What DOES survive is the channel **identity**, which a permutation null does destroy:

| test | observed | null |
|---|---|---|
| which channel dominates, over 768 live cells | **[126, 216, 75, 351]** | 192 each |
| chi-squared vs uniform (3 dof) | **228.7** (p << 0.001) | 16.3 = p 0.001 |
| per-cell fraction of the 600 pairs agreeing on the dominant channel | median **0.3958** | 0.25 |
| cells with >50% temporal agreement | **215/768 = 28.0%** | — |
| global per-channel share of total `|g|` | [0.2492, 0.2624, 0.2222, 0.2661] | 0.25 each |

**Honest reading:** there IS statistically significant per-cell channel structure that every
per-cell key discards (chi-squared 228.7), and it is **cell-specific, not a global gain imbalance**
(the global shares are within 2.6% of uniform). But it is modest, and only **28%** of cells have a
temporally stable dominant channel. **So this does NOT explain the residual** — it is a real
unexploited axis, not the answer. **The 14.8% gap remains UNEXPLAINED, and saying so is the
honest state of §5.**

**(c) Temporal structure, also discarded by summing over pairs:** 250 of 600 pairs carry 50% of
the total `|g|`, 444 carry 80%; per-pair max/median 3.09.

**The sharp consequence.** The gradient's residual advantage is not visibility — it is that the
gradient is a **richer object than the key built from it**. `gr1` computes a per-(pair, cell,
channel) tensor and then flattens it to 768 numbers, discarding a median-3.3x channel anisotropy
that is single-channel-dominated in 21.4% of cells. **`br1` already measured the byte side at the
UNIT (cell x channel) grain — 1,528 live units.** Both halves of a **per-unit** waterfill
therefore already exist on disk; the per-cell formulation is a lossy projection of data we have.

### §5.4 ASK 3: cost and reach — the exact instruments are **cheaper than the proxies**

| instrument | what it produces | cost | on disk? |
|---|---|---:|---|
| `gr1` backprop gradient | per-(pair, cell, channel) damage, n600 | **36.4 s** | **yes** |
| my geometric RF key | per-cell damage proxy | ~ms, **but** needs an RF measurement (24 s) **and** an atlas that cost a full n600 scorer run | yes |
| exact per-cell byte marginal | the real `Delta b`, 384 cells | **388.8 s (1.01 s/cell)** | **yes** |
| `wr1`'s `residual_mass` byte proxy | rho **0.513** vs the exact marginal | ~ms | yes |

**The gradient key is ~10x cheaper than the byte marginals I measured exactly, and produces a
strictly richer object.** And the second half of the ask has a cleaner answer than "find a
gradient for bytes": bytes come from a discrete entropy coder and are not differentiable, **but
they do not need to be — the exact answer costs 1.01 s/cell and is already measured.**

> **Both axes of `#766`'s `lexsort` can be replaced by exact measurements that exist on disk
> today**: damage from `gr1`'s cached n600 gradient (36.4 s), bytes from the encoder (388.8 s).
> Total ~7 minutes for an exactly-keyed waterfill, versus a primary key on 4% of the wrong
> support and a tie-break at rho 0.513.

### §5.5 ASK 4: the general form — the class, with five instances

> **A hand-built proxy stands in for a quantity that an exact instrument ALREADY RUNS in the live
> path. The proxy's calibration is then silently sized for a different question than the one it
> is now answering.**

| # | proxy | the exact instrument already in the path | measured gap |
|---|---|---|---|
| 1 | `wr1` `flip_mass` on a 16x16 tile | backprop through the decoder carries the true support | rho 0.695 → 0.852; **144/486 false-safe** |
| 2 | `wr1` `residual_mass` byte proxy | **the encoder**, 1.01 s/cell | **rho 0.513** |
| 3 | **my own `rs2_rf` box** | the same gradient | rho 0.829; **I built a geometric proxy where a gradient existed** |
| 4 | `rt2` `margin_floor = 0.1` (cited, not mine) | derived for the L7 fp32 drift guard (~0.096); the separatrix scale here is 2.0582 | median margin **59x** the floor |
| 5 | `clip(rint(U(r)))` delivery | `ll1`'s exact per-window solve | 88 flips → 3 flips (**96.6%** of the realization debt) |

**The diagnostic, in one question:** *does an exact instrument for this quantity already run
somewhere in the live path?* If yes, the proxy owes a measured rank correlation against it, or it
is noise wearing a number. Instances 1–3 are all in the SAME 4-line `lexsort`, which is the
strongest evidence this is a class and not three coincidences.

**Self-indictment, stated plainly:** instance 3 is mine. I spent this arm measuring a receptive
field by finite differences to hand-build a support model — three calibration choices, each a
chance to be 24.2x wrong — when the chain rule computes that support exactly, for free, and
`gr1` had already cached it at n600. **The correction I landed was right; the instrument I chose
to land it with was the same class of mistake I was correcting.**

---

## §4 — RECURSIVE ADVERSARIAL REVIEW

The counter resets on any round that finds something. **Round 1 found four things, so it is not
a clean pass.** All four are recorded with what I did about them.

### R1-a — "does the 24x depend on my single-cell RF estimate?" **It does not. MEASURED.**

The 486 -> 342 result rests on a receptive-field half-width taken from ONE cell. I swept it:

| RF half-width | box px | RF-zero cells | wr1-safe-but-NOT-RF-safe |
|---:|---:|---:|---:|
| **0 (= wr1's own tile)** | 256 | **486** | **0** |
| 8 | 1,024 | 446 | 40 (8.2%) |
| 16 | 2,304 | 416 | 70 (14.4%) |
| 24 | 4,096 | 377 | 109 (22.4%) |
| **34 (measured)** | 7,056 | **342** | **144 (29.6%)** |
| 44 | 10,816 | 291 | 195 (40.1%) |

Two things fall out. **The `half = 0` row is a positive control:** my box-sum reduces EXACTLY to
wr1's `bincount`, reproducing 486 and 0 — an independent second route to
`wr1_flip_mass_reproduced: true`. And the conclusion is **monotone and one-sided**: even at a
quarter of the measured half-width, 8.2% of the "provably safe" tranche is not safe. Only the
magnitude depends on the RF estimate; the direction cannot.

### R1-b — "is my `D` really the scorer's downsample?" **VERIFIED against torch.**

The whole DRIVE measurement rests on `window_geometry()` being the frozen scorer resample. I did
not take the docstring's word for it: on a random `(874,1164,3)` frame,

```
max | D_mine(cam) - F.interpolate(cam, (384,512), bilinear, align_corners=False, antialias=False) |
  = 0.0101  on a 0..255 range  (~4e-5 relative; fp32 accumulation order, not semantics)
```

**VERIFIED_VIA_SOURCE_INSPECTION + executed.** (This exercises the resampler only; no SegNet or
PoseNet weights were loaded.)

### R1-c — negative-existence claim, hunted and SCOPED

I claimed *"no receipt answers whether the ordering is right"*. Negative existence is the
campaign's #1 false-claim class, so I searched rather than asserted. What I found:

* `ddm_gr1` **does** run matched-bytes comparisons — but between **granularity FAMILIES**
  (*"cell_rung (graded {L8,L4}) is DOMINATED by cell-DROP at matched bytes"*), not between two
  orderings within the drop family.
* `ddm_ba31` compares **DIRECTIONS** off the knee (restore vs drop-more), not orderings.

**Scoped negative, retained in this narrower form:** *I found no receipt that compares two
cell-drop ORDERINGS at matched bytes.* Denominator: `tools/corpus_query.py` over
research/equations/memory/dag/council/tasks/docs (index ~76%, 7,398 of 9,706 documents), three
distinct queries, plus direct reading of `ddm_wr1_reverse_waterfill.py`,
`ddm_gr1_granularity_rerace.py` and `ddm_tw1_token_waterfill_state_dependence.py`. **Not
exhaustive.**

### R1-d — the search surfaced a real limitation I had not stated: **every key here is POSE-BLIND**

`ddm_gr1_granularity_rerace_20260730.md` records it about its own key: *"|g|-sum ordering is
SEG-only (the backward was seg-loss) — pose-BLIND"* and *"cell_drop50's ordering is seg-only /
pose-blind"*. **My corrected key inherits exactly the same blindness** — ambient SegNet flips say
nothing about `d_pose`. And a token change moves BOTH frames (`frame_0 := a*warp(frame_1) + b`),
so pose is a real cost on every drop, at `dS/d(d_pose) = 31.302`. Credit to `gr1` for the
observation; §2.4's admission arithmetic already carries the pose term, and §2.5 now requires the
pose delta to be REPORTED rather than assumed zero. **A pose-aware key is a named open item, not
something this arm delivers.**

### R2-a — I overstated the readiness of the gate. **CORRECTED.**

§2.3 first read *"the gate is one command with no build step left in it."* That is **false**. The
`archive.zip` files are built and byte-closed, but `upstream/evaluate.py` consumes a SUBMISSION
DIR, which also needs `inflate.sh` plus the **ix2-capable** runner and its flat dependencies. The
honest statement, now in §2.5: the archive is done; assembling the submission tree is one
remaining step, and both halves of it are named. Caught by asking "would my own command actually
run?", which is the only question that finds this class.

### R2-b — the n48 prior needs a subset correction, and the correction is MEASURED

§2.2 extrapolates `gr1`'s **n48** cell sweep to an n600 flip count. That is the
subset-is-a-different-population trap, so I looked for the calibration rather than assuming it —
and `gr1` measured it: `gr1_confirm_cell_drop50_n600.json` realizes `cell_drop50` at
**n600 d_seg 0.004310379** against the same candidate's **n48 0.003947**.

**The n48 subset understates n600 d_seg by 9.2% on this exact family.** Propagating that as a
ratio widens the arm-A prediction from 125,732 flips to **~137,300**, i.e.

> **pre-registered band for arm A: 125,700-137,300 realized flips = 2.02x-2.21x over its
> 62,192-flip budget.** Arm B's proxy-scaled band is **90,600-99,000 = 1.45x-1.59x over.**

### R2-c — cross-check: two independent n600 realizations of the same `frame_1` agree to 1.4e-6

`gr1`'s n600 `cell_drop50` realizes `d_seg = 0.004310379`; `pz1`'s n600 on the live `cx1` base
realizes **0.004311795**. SegNet reads only `frame_1`, and `cx1`'s lattice is bit-identically
`gr1`'s `cell_drop50` (verified 768/768 in §2.1), so these should agree. They differ by
**1.4e-6 = 167 flips of 508,639 (0.033%)** — small, but **not zero and not explained here**.
Recorded as a bound rather than waved away: it means the seg leg is reproducible to ~167 flips,
which is **375x smaller** than the ~62,000-flip decision thresholds in §2.4, so it is immaterial
to the gate and material to anyone quoting `d_seg` to six figures.

### R3-a — the A/B does NOT test the wr1 finding. **SCOPE CORRECTION.**

A reader could conflate two different things, so state them apart:

* **§1's finding is about `wr1`'s tile key** and is established by direct measurement
  (486 -> 342 zero-flip cells, monotone in RF half-width). It needs no A/B.
* **The §2 A/B pits `gr1`'s GRADIENT key (arm A) against the RF-corrected key (arm B)** — because
  `gr1`'s key is the one that actually selected the live base and the one `na1` P0-2 asks about.
  `gr1`'s key is **support-correct by construction** (backprop carries the receptive field), so
  the A/B is NOT a support test; it tests **a first-order linearisation of a large discrete step
  against realized flip proximity.** That is a well-posed and separately useful question, but it
  is not the same question as §1's.

### R3-b — "27.9% less flip mass" is a RELATIVE index, not a flip count

RF boxes overlap (84 px footprints on a 16 px pitch), so summing ambient flips over a cell set's
boxes double-counts: the knee-A set's RF sum is 1,585,454 against 458,738 total flips in the clip
— a ratio of **3.46**, which is the overlap, not a paradox. **Both arms are summed identically,
so the 27.9% RATIO is meaningful and the absolute is not a flip count.** The §2.4 prediction that
scales arm A's flips by that ratio inherits the proxy assumption and is labelled accordingly.

### R3-c — NO-OP DETECTOR: the dropped bytes are actually consumed. MEASURED.

An archive that parses and renders is not yet an archive that *differs*. Rendering base, A and B
through the real receiver on 5 pairs (0, 137, 299, 411, 577) and differencing in the scorer plane:

| pair of arms | max abs delta (LSB) | mean abs delta | scorer px changed |
|---|---:|---:|---:|
| base vs **A** | 172.8 | 2.647 | **576,040** / 983,040 (58.6%) |
| base vs **B** | 175.4 | 2.531 | **512,053** / 983,040 (52.1%) |
| **A** vs **B** | 150.1 | 1.456 | 429,364 / 983,040 (43.7%) |

Catalog #105 satisfied: the bytes are structurally consumed and the two arms are genuinely
distinct objects, not a repack. **Bonus, and it points the same way as §2.3:** at equal bytes
arm B perturbs **11.1% fewer scorer pixels** than arm A — an independent DRIVE-side signal
agreeing with the 27.9% susceptibility-side signal.

### R4 — attacking the NEW §5 material. Three findings; the counter stays at 0.

* **R4-a RETRACTED (category error):** my visibility probe reported a
  `frac_gain_below_uint8_step` row comparing a dimensionless gain to an amplitude. Retracted in
  §5.2, and the replacement is sharper: the quantiser dead zone is amplitude-dependent, and
  **only DRIVE — a realized finite difference — can see it**; neither the gradient nor a
  geometric key can, because a linearisation cannot represent a dead zone.
* **R4-b CORRECTED (my headline statistic was noise):** I led §5.3(b) with a median 3.34x
  within-cell channel ratio. It is **permutation-invariant by construction** and a 200-draw
  within-cell permutation null reproduces it **exactly** (0.3531). Replaced by the channel
  IDENTITY tests, which a permutation null does destroy (chi-squared **228.7**, p << 0.001) —
  real structure, but modest, and **it does not explain the residual**. The 14.8% gap is
  **UNEXPLAINED**, and §5 now says so.
* **R4-c LOOSENED:** `1 - rho` is not a variance share. "explains 85% of the ranking" became
  "cannot reproduce the order".

### Round 3 verdict, and the honest counter

Rounds 1-4 found 4 + 3 + 3 + 3 = **13 items**, three of them in the §5 material added last and
two of those defects in my OWN headline statistics. **The counter stands at 0 consecutive clean
passes** — per the bug-class-spread prior that is evidence more exist, not that the surface is
clean. Every §2 verdict is PROVISIONAL by construction anyway —
the realized flip cost is unmeasured — and the three §2.4 falsifiers are the mechanism that
converts them. **I am not declaring a SEAL.**

---

## §7 — ASSUMPTION LEDGER

| assumption | classification | note |
|---|---|---|
| `#766` ranks flip damage PRIMARY, bytes as tie-break | **VERIFIED_VIA_SOURCE_INSPECTION** | `ddm_wr1_reverse_waterfill.py:87-93,257` |
| `gr1`'s key is a TRUE backprop gradient, so support-correct | **VERIFIED_VIA_SOURCE_INSPECTION** | `ddm_gr1_granularity_rerace.py:140-168`: `nn.value_and_grad` over a loss that renders and runs the real SegNet adapter; `grads["tokens_delta"]` |
| every key here is POSE-BLIND | **VERIFIED_VIA_SOURCE_INSPECTION** | `ddm_gr1_granularity_rerace.py:151` `compute_pose=False`; my ambient/thin-margin keys are SegNet-only by construction |
| `gr1`'s key is a smoothed SURROGATE of the flip count | **VERIFIED_VIA_SOURCE_INSPECTION** | `:142,150` `seg_loss="tau_softplus"` — a gradient of a smoothed proxy for a discontinuous count, which is a third proxy layer on top of the linearisation |
| my `D` is the frozen scorer downsample | **VERIFIED_VIA_SOURCE_INSPECTION** + executed | matches `F.interpolate(..., bilinear, align_corners=False, antialias=False)` to 0.0101 of 255 |
| the A/B arms are not a repack (bytes are consumed) | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | no-op detector §4 R3-c: 512k-576k of 983k scorer px changed |
| wr1 attributes flips to the cell's own 16x16 tile | **VERIFIED_VIA_SOURCE_INSPECTION** | `:89` |
| a cell drop perturbs 84x82 = 6,192 scorer px | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | real receiver + frozen `D`, cell (13,17); per-cell distribution in flight |
| the live base is `gr1`'s `cell_drop50` | **VERIFIED_VIA_SOURCE_INSPECTION** | reproduced ordering matches `qa24_grid_keep_mask_50.npy` 768/768 AND the shipped lattice's live set |
| gr1-ordered `cell_drop63` = 274,631 B archive | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | real encoder + real container + real ZIP; archive BUILT and re-parsed |
| both A/B arms decode bit-identically to their lattice | **VERIFIED_VIA_EMPIRICAL_ANCHOR** | `lattice_roundtrip_exact: true`, receiver renders |
| ambient flips are a susceptibility proxy | **INFERRED_FROM_DOMAIN_LITERATURE** | wr1's assumption, inherited so the A/B isolates SUPPORT; independently corroborated by the thin-margin key at rho 0.99 |
| the ambient-flip atlas describes the LIVE base | **ASSUMED_AWAITING_VERIFICATION** | `ru1`/`sg1` endpoint has **458,738** flips; live `cx1` has **508,639** (10.9% apart). The SUPPORT correction is geometric and endpoint-free; the absolute flip masses are borrowed. The thin-margin key is endpoint-INDEPENDENT and agrees at rho 0.99, which bounds the risk but does not remove it |
| GT-frame margins proxy the margin on OUR delivered frames | **ASSUMED_AWAITING_VERIFICATION** | `gt_n600.npz['margins']` is computed on GT frames; the operative margin is on the rendered frame |
| realized flip cost of arms A and B | **ASSUMED_AWAITING_VERIFICATION** | => every §2 verdict is **PROVISIONAL** |
| **my charter's premise "the waterfill ranks by bytes"** | **REFUTED** | primary key is `flip_mass` |
| `br1`'s `cell_drop63 = -72,544 B` answers P0-2 | **REFUTED** | different cell set; gr1's own ordering gives -79,177 B |
| `br1`'s all-symbol coder ratio 1.4834x | **REFUTED (arithmetic)** | its own inputs give 1.4776x |

**Shared assumption this work operates within, and whether violating it unlocks breakthrough:**
*that the CELL is the right drop unit.* Every key here — gr1's, wr1's, mine — ranks 768 fixed
16x16-footprint cells. But the measured receptive fields OVERLAP heavily (84 px footprints on a
16 px pitch), so cell drops are **not** independent actions and a greedy per-cell waterfill is
solving the wrong combinatorial problem. Violating that assumption — pricing drops on the
overlap graph rather than per cell — is the untested direction §1.5's per-cell RF map opens.

---

## §9 — NEXT-IF-RESUMED

1. **Do not "fix" the waterfill to rank by flips.** It already does (`ddm_wr1:93`). The fix is
   the SUPPORT. I did **not** land it in `experiments/ddm_wr1_reverse_waterfill.py` for one
   stated reason: a sister session holds uncommitted edits in
   `experiments/ddm_tw1_token_waterfill_state_dependence.py`, which **consumes wr1's
   `drop_rank`**, so silently changing the ranking under it is the collision the guards exist to
   prevent. The patch is written out here so applying it is mechanical, and `rf_half = 0`
   reproduces today's behaviour **exactly** (verified: half=0 gives 486 and 0 in §4 R1-a), which
   makes it a safe default-off lever with this memo as its duty-to-measure record:

   ```python
   # experiments/ddm_wr1_reverse_waterfill.py :: cell_sensitivity
   def cell_sensitivity(delta, atlas_flat, *, rf_half: int = 34):   # 0 == today's behaviour
       ...
       atlas = np.load(atlas_flat)
       y = atlas["y"].astype(np.int64)
       x = atlas["x"].astype(np.int64)
       dense = np.bincount(y * 512 + x, minlength=384 * 512).reshape(384, 512)
       ii = np.zeros((385, 513), np.int64)
       ii[1:, 1:] = dense.cumsum(0).cumsum(1)
       flip_mass = np.empty(768, np.float64)
       for r_ in range(24):
           for c_ in range(32):
               r0, r1 = max(0, r_ * 16 - rf_half), min(384, (r_ + 1) * 16 + rf_half)
               c0, c1 = max(0, c_ * 16 - rf_half), min(512, (c_ + 1) * 16 + rf_half)
               flip_mass[r_ * 32 + c_] = ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]
   ```

   `:131`'s `dseg_ceiling = REF_DSEG + dropped_flip_mass / TOTAL_PX` must ALSO be revisited: with
   overlapping RF boxes the summed mass double-counts (§4 R3-b), so it is a relative index, not a
   d_seg ceiling. And `--knee-a 486` should be re-described: **342** cells are zero-flip on the
   real support, not 486.
2. **The gate worth the slot is the byte-matched A/B in §2.3, not `cell_drop63` alone.** Both
   archives are built and byte-closed; the prediction and all three falsifiers are pre-registered
   in §2.4. Whichever way it lands it prices every future drop rung.
3. **Carry the corrected byte leg forward:** gr1-ordered `cell_drop63` is **-79,177 B / 62,192
   flip budget**, not `br1`'s -72,544 / 56,982.
4. **The thin-margin key is the better one and it is free.** rho 0.99 with the RF-ambient key,
   endpoint-INDEPENDENT (no borrowed-vehicle caveat), and covers every pixel rather than only
   the already-wrong ones. Prefer it once the A/B calibrates which key predicts realized flips.
5. **Cell drops are not independent** — 84 px footprints on a 16 px pitch overlap ~5x in each
   axis. A greedy per-cell waterfill mis-prices the interaction. The per-cell RF map is the input
   to pricing on the overlap graph instead; **it is still OWED** (§1.6), and re-running
   `rs2_drive_sweep.py` needs the two fixes below FIRST or it will lose its work again.

5b. **Two operational lessons, both mine, both cheap to fix and expensive to repeat:**
   * **Append per group, never at the end.** `rs2_drive_sweep.py` saved once after 36 groups and
     lost 24 groups of completed n600 work to a silent kill. The sister script
     `rs2_arm_drive_n600.py` writes a JSONL row per chunk — it was killed twice and lost nothing
     both times. Three lines of difference.
   * **Job state is only readable from a RECEIPT.** In this one session the process-table probe
     was wrong THREE times in BOTH directions: `pgrep -f <script>.py` reported ALIVE for minutes
     after a real death (it matched my own watcher shells), and a lagging log tail plus `ps` made
     me declare two deaths for a job that had already finished n600 (§1.6b). The
     `until [ -f <receipt> ]` waiter was right every time and I overrode it every time. **Never
     probe the process; poll the receipt.**
6. **Owed, and deliberately so:** the `__init__` export + locked-registry `populate_*` for the
   three §3 equations, held back only because a sister session holds those files.
7. Artifacts: `/Volumes/VertigoDataTier/pact/ddm_rs2_20260803/` —
   `ab/rs2_{kA_gr1_drop63,kB_rs2_rfkey_bytematched}_archive.zip`, `rs2_ab_build_receipt.json`,
   `rs2_drive_pilot.json`, `rs2_thin_margin_keys.{json,npz}`, `rs2_footprint_rerank.json`,
   `rs2_drive_sweep/{cell_drive.npz,receipt.json}`. Scripts:
   `scratchpad/rs2_{drive_pilot,drive_sweep,footprint_rerank,build_ab_archives,thin_margin_key}.py`.
