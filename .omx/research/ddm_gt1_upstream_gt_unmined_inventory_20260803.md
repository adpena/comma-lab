# ddm_gt1 — GT never ships; its only job is to tell the encoder WHICH FREE BASIS to buy coefficients for

**UTC** 2026-08-03 · **arm** `ddm_gt1_upstream_gt_unmined_inventory` · **axis**
`[macOS-CPU advisory]` / `$0 source inspection + exact linear algebra + re-reduction of landed
n600 receipts`. **No SegNet or PoseNet forward or backward was fired. The evaluator slot (held by
`ddm_qd2` / `ddm_de1` / `ddm_cu1`) was not requested and not touched.** `score_claim=false`,
`promotion_eligible=false`. **Pointer UNMOVED.**

**Baseline for every ΔS below** (charter, recomputed from components, never a rounded field):
`S = 100·d_seg + √(10·d_pose) + 25·B/37_545_489`. Live base `ddm_cx1` **S = 0.8264972**
(seg 0.4311790 · pose 0.1597320 · rate 0.2355862, 353,808 B). **Target** = PR130 demonstrated
floor **0.172141** (archive **191,052 B**, rate 0.127214). **Remaining gap = 0.6543562**
— seg 0.4015190 (61.4%) · pose 0.1445 (22.1%) · rate 0.1083722 (16.6%).
**1% of the gap = 9,827 B** (re-derived here; `tac.canonical_equations.gap_decomposition_against_floor_20260802`).

**Reproducers — every number below regenerates from these, all $0 and scorer-free:**
`upstream/modules.py`, `upstream/frame_utils.py`, `upstream/models/*.safetensors` (source
inspection); `reports/ddm_bp2/reach_n600.jsonl` (the n600 re-reduction in §5).

---

## §0 THE HARD CONSTRAINT, stated correctly (MAIN's 2026-08-03 correction adopted)

My charter's first framing — *"GT-derived content is video-derived ⇒ COUNTED"* — **conflated two
different sets and I am not using it.** The rule-118 test is **VIDEO-SPECIFIC**, not "learned by
looking at GT." The operative taxonomy is three-way:

| class | verdict | live precedent |
|---|---|---|
| property of the **fixed operators** (`D`, `R`, `rgb_to_yuv6`, uint8 lattice, openpilot geometry) | **GENERIC → FREE in `inflate.py`** | `#401` blind-coordinate fill rests exactly on "this is a property of `D`, not of the video" |
| property of the **frozen scorer weights** | video-INVARIANT; status is **ECONOMIC, not legal** — priced in §4 | — |
| property of **THIS CLIP** (`lstars`, the 6 pose scalars, any fitted table) | **COUNTED** | — |

**The law that organizes every row in this memo — and the answer to the operator's question:**

> In every exploit this campaign has landed, the **BASIS / SPAN / GEOMETRY is GENERIC and FREE**
> (`inflate.py` recomputes it from its own state at zero counted bytes), and the **COEFFICIENTS
> are COUNTED** (this clip's projection onto that basis). **GT's entire role is ENCODE-side: it
> tells us which basis is worth carrying coefficients for. GT itself never ships.**
> Corollary that ranks everything below: *the value of a GT study = (quality of the generic basis
> it selects) × (coefficient bytes saved by having the right basis).*

This is not my invention — it is `ddm_pb3` §3 stated generally (*"Only the SPAN has to be right,
because the encoder fits the six coefficients freely"*). Naming it as the general law is what lets
the table below have a non-empty GENERIC/FREE column, which is what MAIN asked for.

**What does NOT change:** rule-118 binds however the content is spelled; hide-data-in-code is the
fake (`#417` receiver-consumption bijection); the pinned upstream snapshot is immutable. The
correction WIDENS what counts as generic; it does not weaken the video-specific test by one byte.

---

## §1 THE ANSWER FIRST

1. **The single most valuable thing this arm did was NOT find something — it caught itself about
   to rediscover.** I measured the SVD of `hydra.final_layer.pose.weight[:6]`
   (σ = [8.769, 0.988, 0.700, 0.512, 0.420, 0.353], **cond 24.8**, output dim 0 carrying 97.4% of
   ‖W₆‖²_F) and was one paragraph from presenting it as the arm's headline. **It is PRIOR by 101
   days**: `project_posenet_rank1_discovery.md` (~2026-04-24, "Jacobian rank 1.008; dim0 = 99.80%
   of variance"), then `ddm_pi2` (07-30, QA51 — **cond 24.8 and the 26-dim head-null explicitly
   named**), `ddm_ua1` (07-31, re-derived 24.8), `ddm_gc16` (07-31, which had already
   self-REFUTED the same claim as its own discovery). **I claim none of it.** Recorded because the
   charter named rediscovery as this arm's most likely failure and it very nearly happened.
2. **GT1-1 is CLOSED with no freedom to gain.** `ddm_pb3`'s `rank(J) ≤ 6` is measured on the
   **SCORED** 6 (`d_pose = ‖e‖²/6`, `e` a 6-vector), and every repo consumer slices `[..., :6]`.
   The pose-null subspace is already computed at the correct, larger, codim-≤6 size. §2.
3. **GT1-3 is already-done and NEGATIVE, by our own arm yesterday.** `ddm_sg2` §2 IS the
   allocation-side measurement: the margin-defined flip-capable set is 0.4312% of pixels but
   carries 1.17% of through-R leverage — a tilt of **2.713×, not 238×**, and *flat* across margin
   buckets. The margin field does not concentrate allocation the way the diagnosis suggested. §2.
4. **The full-weights decode-side scorer is DEAD BY ARITHMETIC, and I can now say by how much.**
   The absolute ceiling for ANY decode-side seg apparatus — assuming it drives `d_seg` to
   **exactly zero** — is **647,553 B (632 KiB)**. SegNet alone is 38,502,892 B = **59.5× that
   ceiling**; shipping it nets **+25.21 S** even at `d_seg = 0`. **But MAIN is right that this
   does not kill the small end**, and the small end has a much better target than I expected. §4.
5. **CLAUDE.md's "~73 MB" for the scorer pair is STALE.** Re-derived from the files:
   segnet 38,502,892 + posenet **55,835,560** = **94,338,452 B (94.3 MB)**, 29% larger. §4.
6. **My one new n600 measurement returned an honest NEGATIVE that refutes my own hypothesis.**
   The pose tail is **not** a reachability tail — the hard pairs are the *more* reachable ones
   (tail/body directional-pullback ratio **5.44×**, `r(log d, log γ) = +0.52`). §5.

**Recall accounting (charter-required): `already-done 6 / newly-proposed 7 / total 13`.**

---

## §2 ADJUDICATION OF MAIN'S FOUR

### GT1-1 — the 6 unscored pose dims → **ALREADY-DONE / CLOSED. No freedom to gain.**

**Reconciled at source.** `upstream/modules.py:26` `HEADS = [Head('pose', 32, 12)]`; `:84`
`compute_distortion` slices `[..., : h.out // 2]` = the first 6. `ddm_pb3` writes
`d_pose = ‖e‖²/6` with `e` a 6-vector and `J = ∂p/∂δ` of shape `6 × 692,712` — **its rank bound is
on the SCORED 6, with the unscored 6 already correctly excluded.**

**Verified that nothing in the repo accidentally constrains all 12** (the only way the unscored
half could be silently costing us freedom). Scanned `src` + `tools` + `experiments`: 691 files
matching `out // 2`, 2,098 matching the `[..., :6]` family; the stored sidecar
(`src/tac/scorer_targets.py`) stores 6 by construction. **One stale artifact, code correct:**
`experiments/feedy_byteclosed_exact_row_probe.py:104` comments *"PoseNet 'pose' head out=6 ->
first 3"* — factually wrong (out=12, half=6) — while `:109` uses the correct dynamic
`half = h.out // 2`. **Comment wrong, code right; no behavioural defect.** Worth a one-line
correction by whoever next edits that file; not worth a commit from this arm.

**NEW datum, no lever attached (row H).** The unscored rows are **not** redundant with the scored
ones: principal angles between the two rowspaces are **62.3° / 62.6° / 64.2° / 69.4° / 81.0° /
85.3°**, and the unscored half carries *more* mass (‖W[6:]‖_F = 13.24 vs ‖W[:6]‖_F = 8.88). The
trained head spends more output capacity on what is never scored than on what is. This is a
curiosity about the frozen net, **not an actuator** — we cannot make the scorer read those
dimensions, and the head-null they imply (32 − 6 = 26-dim) was already named by `pi2`.

### GT1-2 — the reachable feature manifold → **OPEN, but ranked low. Honest reasons.**

The premise is sound and not obviously covered by `at1` (whose n600 gaze/Jacobian shards are
recorded BLOCKED) or `sn1` (telemetry asymmetry, a different statistic). GT's 1,200 frames are the
only in-distribution sample of the frozen scorers' own behaviour, and the frame-set landing in a
given argmax cell could be materially larger *along* that manifold than an ambient analysis says —
which bears on `sg2`'s **14.537× required**.

**Why I rank it low anyway, and this is the honest part:** (a) it needs feature extraction over
1,200 frames = a scorer slot, which this arm is forbidden and three live arms hold; (b) its output
is a **loosening of a bound**, not a byte — it changes what we believe is feasible, not what
ships; (c) `sg2` §3 already measured the two ambient operators it would be compared against and
found `R` has **no free subspace at all** (σ ∈ [0.6866, 1.0283], cond 1.4975, zero modes below
0.5), so the ambient analysis it would relax is already tighter than the "resize throws away
information" intuition assumed. A manifold argument has to beat that, not the intuition.

### GT1-3 — GT logit margin as per-pixel tolerance: allocation or diagnosis? → **ALREADY-DONE, NEGATIVE.**

Charter asked me to distinguish "wired into ALLOCATION" from "only DIAGNOSIS" with evidence.
**Evidence: `ddm_sg2` (2026-08-02) §2 is the allocation-side measurement, and it refutes the naive
transfer.** MEASURED n600 on the cached through-R reachability map: the flip-capable set
(margin < t\* = 0.153053) is **0.4312%** of pixels but carries **1.17%** of through-R input
leverage — **tilt 2.713×, not 238×** — and **74.91% of all leverage sits on pixels whose own
margin is ≥ 4**, structurally unable to flip themselves. The tilt is also flat (2.60–2.73 across
every margin bucket 0–2). So `#766`'s missing input is **not** margin-saliency: margin-weighted
waterfill and magnitude waterfill differ by under 3×, and `sg2` already priced that ceiling.
`ddm_sg2`'s separate finding that source-margin geometry predicts edge debt to **2%** remains a
strong *predictor* — it is the seed for §4's separatrix-student framing, not for reweighting.

### GT1-4 — uniform objective × heavy-tailed difficulty → **MOSTLY ALREADY-DONE; one new negative from me.**

The asymmetry is real and thoroughly measured: `pc2` `err_rate ∝ area^−1.26` (r = −0.934);
`pb3` top-1% of pairs carry 62.1% of pose reduction, median per-pair gain 0.198%; `v4d` §4 top-17
pairs = 74.3% of pose mass; `sg2` per-frame ρ CV 11.3% with 2.17× min-to-max. The objective is a
flat mean; the difficulty is not. **What was NOT known is WHY the tail is a tail — and I tested
the most actionable candidate mechanism and it failed.** §5.

---

## §3 THE GENERIC / FREE COLUMN — what row-1 structure GT study reveals

Charter (as corrected): *"your ranked table should now have a GENERIC/FREE column that is not
empty. If after honest adjudication it IS empty, that is a strong finding, but it must be the
measured conclusion."* **It is not empty. Here is the full inventory, with what is already
exploited marked, because an inventory that hides the already-exploited rows overstates the
opportunity.**

| generic operator fact | status | exploited by |
|---|---|---|
| `D` = the shared `interpolate(→512×384, bilinear)`; maps 1,017,336 → 196,608, **80.67% of shipped-error directions exactly invisible** | **EXPLOITED** | `#401` fill, `bp2`/`pb3` blind set, `sg2` §3 |
| disjoint 2×2 sampling (stride 2.276 > 2): 4 private camera px per scorer px, **22.70% blind to BOTH scorers** | **EXPLOITED** | `m86`, `bp2` `blind_verify.json` |
| `R` round-trip spectrum: σ ∈ [0.6866, 1.0283], cond 1.4975, **no free subspace** | **CLOSED (negative)** | `sg2` §3 |
| `rgb_to_yuv6` per-2×2-block null: 12 RGB → 6 outputs ⇒ **6 exactly-pose-null, seg-VISIBLE dims per block** | **FULLY PRIOR — including the count.** I derived 6 × 192 × 256 = **294,912**; `ddm_control_surface_exact_quartering_20260731` §Q3 MEASURED **294,912**, pose **5.684e-14**, seg **6.000**. Also prior: `upstream_scorer_alldim_reread_20260710` row G, `frozen_scorer_exact_factorization_20260715` §5 | `Q3` |
| frame_0 is seg-invisible (`x[:, -1, ...]`) ⇒ generate, don't store | **ALREADY BUILT** — v4d ships `f0 := a·warp(f1)+b` (`gc16` refuted this as novel) | v4d |
| openpilot camera geometry / lane polynomial / homography / `tac.lie` SE(3) | **EXPLOITED**; CLAUDE.md blesses by name | `pb3` §3 interaction matrix, v10 |
| rate denominator `37_545_489` **is exactly `sizeof(upstream/videos/0.mkv)`** — verified | fixed constant, nothing to mine | — |

**Row A — `rgb_to_yuv6` clamp-saturation as a second generic blind set: NEW, and I killed it by
derivation in five minutes.** `Y`, `U`, `V` are each `.clamp_(0, 255)` (`frame_utils.py:61-63`).
Where a clamp is active the map is locally constant ⇒ the pose gradient is *exactly* zero
regardless of content, and the decoder can detect it from its own render at zero counted bytes — a
genuinely different mechanism from `D`-geometry blindness. **But the clamps are essentially never
active.** `Y = 0.299R + 0.587G + 0.114B` is a convex combination of values in [0,255], so
`Y ∈ [0,255]` **always** — the Y clamp is unreachable. `U = (B−Y)/1.772 + 128` has range
[0.5, 255.5] and `V = (R−Y)/1.402 + 128` likewise, so each can clamp only in a corner of the RGB
cube of measure ~0 (`B=255, R=G=0` exactly, and its mirror). **Row A is EMPTY.** Recorded as a
checked-and-closed negative so nobody spends an arm on it.

**Row G — the shared-`D` correction: I audited for downstream damage and found none in the live
path.** MAIN flagged that CLAUDE.md asserted the reverse until `be9c6fd11c`, and that any
"seg-invisible therefore pose-cheap" argument resting on differing resizes is void. **Searched
`.omx/research/*.md` for claims resting on a resize *difference*; 11 hits inspected, none asserts
differing resizes — and the two live consumers had it right independently of the doc:** `sg2` §3
cites *both* `upstream/modules.py:73,109` as one operator `D`, and `bp2`'s
`blind_verify.json` carries the hard receipts `blind_hard_perturb_posenet_in_identical: true`
**and** `blind_hard_perturb_segnet_in_identical: true` — i.e. blindness to both scorers was
measured, not assumed. **The doc was wrong; the code was right.** *Scope of this negative:*
`.omx/research/*.md` only, via literal-phrase grep — not an exhaustive semantic audit of every
consumer, and not a scan of `src/`.

---

## §4 ROW F — the decode-side distilled student, priced honestly (MAIN's new row)

**All re-derived from the files, not recalled.**

```
ΔS per archive byte = 25 / 37,545,489            = 6.658590e-07
W (bytes per flip)  = (100/(196,608·600)) / that = 1.273108215332  ✓ reproduces memory exactly
current flips       = 0.4311790 · 196,608 · 600  = 50,863,944
```

**The ceiling.** The most a decode-side seg apparatus could ever be worth is the entire seg
contribution it could remove. At `d_seg → 0` exactly:

> **ceiling = 0.4311790 / 6.658590e-07 = 647,553 B = 632.4 KiB.**

| | bytes | vs ceiling |
|---|---:|---:|
| SegNet `safetensors` | 38,502,892 | **59.5×** |
| PoseNet `safetensors` | **55,835,560** | 86.2× |
| **pair (CLAUDE.md says "~73 MB" — STALE)** | **94,338,452** | 145.7× |

Shipping the full SegNet and achieving `d_seg = 0` nets **rate +25.6375, seg −0.4312 ⇒ +25.21 S**.
**The full-weights option is dead by 59×, and this is arithmetic, not assertion.**

**Break-even for a student capturing fraction `f` of the seg contribution:**

| f | max student bytes | required compression of SegNet |
|---:|---:|---:|
| 1.00 | 647,553 | 59× |
| 0.50 | 323,777 | 119× |
| 0.30 | 194,266 | **198×** |
| 0.10 | 64,755 | 595× |

**MAIN's prior ("it closes — features need the body, the body is the 73 MB") is the right prior
for the wrong target, and that is the finding in this row.** A decode-side apparatus does **not**
need to reproduce 5-class segmentation. `ddm_pc2` MEASURED that **interiors contribute ≈0
(0.058% of flips)** — every flip lives on a codim-1 separatrix — and that **Road participates in
87.8% of all 458,738 flips**, with Road↔Lane alone = 49.2% of flips. `ddm_sg2` MEASURED that
**source-margin geometry predicts edge debt to 2%**. So the target function is a
**one-hub codim-1 boundary predictor**, not a segmenter — a far smaller object than the net that
computes it, and the 198× compression required at f = 0.30 is against the wrong denominator.

**Verdict: NOT closed. Priced, with the falsifier pre-registered.** The row pays iff a
**≤194,266 B** decode-side apparatus recovers **≥30%** of the 50,863,944 flips. My own prior is
still that it fails — but MAIN is correct that the small end has never been priced, and the
separatrix reframing moves it from "obviously dead" to "cheap to falsify."

**The structural objection that must travel with this row, because it bounds the whole family:**
the ENCODER already has the real SegNet and the real GT, so anything the decoder computes with a
student the encoder computes better and ships the *answer*. **A decode-side apparatus can pay only
by AMORTIZATION** — replacing coefficient bytes with fixed apparatus bytes — so it must beat the
coefficient stream it displaces. For pose that stream is already **231 B–4,200 B** (`pb3`), so the
apparatus can *never* win there. For seg the stream is the token payload (cx1 total archive
353,808 B), which is inside the 632 KiB ceiling — **so seg is the only axis where this row is even
arithmetically alive.** Any resumption must target the seg separatrix and nothing else.

---

## §5 THE ONE MEASUREMENT I LANDED — and it refutes my own hypothesis

**Question:** is the pose tail a **reachability** tail? If the hard pairs were hard because their
error lies in a badly-reachable output direction, that would be far more actionable than "they are
just big" — it would name a per-pair *difficulty coordinate* and feed pair selection directly.

**Method ($0, scorer-free, n600, no new forward passes).** `ddm_bp2`'s landed
`reach_n600.jsonl` records `grad_all_l1 = ‖(1/3)Jᵀe‖₁` and `d_pose_base` per pair. Since
`‖e‖ = √(6·d_pose)`, the ratio

> `γ_reach = grad_all_l1 / √(6·d_pose_base) = ‖Jᵀê‖₁/3`, `ê` the **unit** error direction

is a magnitude-normalised directional-reachability index — how much gradient the *direction* of
the error produces, with its size divided out. Nobody had computed it.

**Control that the receipt is being read correctly:** recomputing `pb3`'s own
`γ_pb3 = g1/(2d)` from the same file gives `frac ≥ 1 = 0.5367`, reproducing `pb3`'s published
`0.5366666…` exactly.

**MEASURED, n600:**

| | value |
|---|---:|
| γ percentiles p1 / p50 / p99 | 0.0388 / 0.1796 / 5.5624 |
| spread p99/p1 · CV | **143×** · 1.61 |
| `r(log d_pose, log γ)` | **+0.5249** |
| top-5% d_pose pairs: mean γ | **4.079** |
| bottom-95%: mean γ | 0.750 |
| **tail / body** | **5.44× — the WRONG SIGN for my hypothesis** |

**VERDICT: REFUTED, by me, against my own hypothesis.** The hard pairs are the *more* reachable
ones. **The pose tail is a MAGNITUDE tail, not a reachability tail** — consistent with `pb3`'s
independent finding that the top 1% of pairs carry 62.1% of the achievable reduction (a
low-reachability tail would have shown the opposite). Pair selection should therefore be ranked on
`d_pose` magnitude alone; there is no second difficulty coordinate to discover here.

**`verdict_scope: measurement-level, one index, one vehicle.** γ conflates directional
reachability with the overall magnitude of `J` (a high-texture / fast-motion scene has a large `J`
everywhere), and I do **not** have a per-pair `‖J‖` in the receipt to divide out. The refutation is
therefore sound *against the actionable form of the hypothesis* — "rank pairs by a reachability
coordinate" — and does **not** establish that no `‖J‖`-normalised direction effect exists.
Measured on the `v4d` vehicle (`6e1b80e9…e764`) via `bp2`'s receipts; does not transfer to another
base without re-measurement.

---

## §6 THE RANKED TABLE

`%gap` is against the **0.6543562** cx1→PR130 gap. Sub-budgets: seg 61.4%, pose 22.1%, rate 16.6%.

| # | row | side | ceiling, %gap | falsifier | consumer | status |
|---|---|---|---:|---|---|---|
| 1 | **Row F — decode-side SEPARATRIX student** (not a segmenter): ≤194 KB apparatus vs the token stream | **SHIP-side, counted.** ≤194,266 B at f=0.30 | ≤ **61.4%** (seg gap); realistically the excess over break-even only | a ≤194,266 B student recovering <30% of 50,863,944 flips kills it | seg axis / token stream | **open, NEW, priced §4** |
| 2 | **Row E — common-denominator GENERIC-BASIS RACE**: measure, at n600, what fraction of the GT-vs-render residual each *free* basis (`D`-null, yuv6-null, `R` row space, openpilot geometry, curvelet) captures **per coefficient byte** | **ENCODE-side.** Output is a ranking; bases are free, coefficients counted | ≤ **16.6%** (rate gap) | if per-coefficient capture rates land within 2× of each other, basis choice is not the lever and `#766` should waterfill on magnitude alone | **`#766` waterfill** (memory names it a live rate mover) | **open, NEW** |
| 3 | GT1-2 — reachable feature manifold along GT | ENCODE-side | loosens a bound; **no direct byte** | must beat `sg2`'s measured ambient tightness (`R` cond 1.4975, zero free modes), not the intuition | `sg2` 14.537× | **open**, ranked low (§2) |
| 4 | Row B — `rgb_to_yuv6` 2×2 chroma null: **6 exactly-pose-null, seg-visible dims per block** = 294,912 dims on `D(f1)` | **GENERIC/FREE basis**; coefficients counted | within seg 61.4% | — | seg actuator | **FULLY PRIOR** — `Q3` MEASURED the same 294,912 (pose 5.684e-14); my derivation reproduces it |
| 5 | GT1-4 — heavy-tailed difficulty | ENCODE-side | pair selection only | — | pair selection | **already-done** + §5 negative |
| 6 | GT1-3 — margin → allocation | ENCODE-side | **≈0** (tilt 2.713×, flat) | — | `#766` | **already-done, NEGATIVE** (`sg2` §2) |
| 7 | Row G — shared-`D` wrong-premise audit | — | — | — | — | **NEW, answered: no live consumer affected** |
| 8 | Row H — unscored head rows not redundant (62–85°) | — | **0** — no actuator | — | — | **NEW datum, no lever** |
| 9 | Row I — CLAUDE.md "~73 MB" stale → **94,338,452 B** | — | 0 | — | doc | **NEW correction** |
| 10 | GT1-1 — unscored 6 pose dims | — | **0 — no freedom to gain** | — | — | **CLOSED** (§2) |
| 11 | Row A — `rgb_to_yuv6` clamp-blind set | GENERIC | **0 — set is measure-~0** | — | — | **NEW, killed by derivation** |
| 12 | Row C — rate denominator = `sizeof(0.mkv)` | — | 0 | — | — | **already-done, confirmed** |
| 13 | Row D — frame_0 seg-free ⇒ generate not store | GENERIC | 0 (banked) | — | v4d | **ALREADY BUILT** |

**`already-done 6 / newly-proposed 7 / total 13`.**

---

## §7 ROUND-1 ADVERSARIAL SELF-REVIEW

| I tried to refute | outcome |
|---|---|
| "the pose-head SVD anisotropy is my finding" | **REFUTED — PRIOR by 101 days.** `project_posenet_rank1_discovery.md` ~2026-04-24; `pi2` QA51 07-30 named cond 24.8 **and** the 26-dim head-null; `ua1` 07-31 re-derived 24.8. Claimed nowhere. This is the charter's named failure mode and it nearly landed. |
| "the pose tail is a reachability tail" — **my own hypothesis** | **REFUTED BY MY OWN MEASUREMENT, wrong sign, 5.44×.** §5. Kept as the negative rather than dropped. |
| "`rgb_to_yuv6` clamps give a second free blind set" — **my own row** | **KILLED BY DERIVATION.** `Y` is a convex combination ⇒ never clamps; `U`,`V` reach [0.5, 255.5] ⇒ clamp only on a measure-~0 corner. §3 Row A. |
| **rule-118 walk — did I propose any GT-derived decode-side prior?** | **Rows 2, 4, 11 are GENERIC (fixed-operator properties, decoder-recomputable, zero counted bytes); rows 1 is SHIP-side with its bytes priced; rows 3, 5, 6 are ENCODE-side only.** No row asks the decoder to see a GT-derived field. Row 4 is the closest call and passes only because the **basis** is generic while its **coefficients** are counted — which is the §0 law, stated so the distinction cannot be blurred later. |
| "the 632 KiB ceiling kills the student row" | **NO — over-claim, withdrawn.** It kills the **full weights** (59.5×). It does not kill a ≤194 KB separatrix predictor, and `pc2` (interiors ≈0) + `sg2` (2% edge prediction) say that is the right target. Row 1 stays **open**, not closed. |
| "GT1-2 is open, therefore rank it high" | **NO.** Open ≠ valuable. It costs a scorer slot, returns a bound not a byte, and must beat `sg2`'s already-measured ambient tightness. Ranked 3rd, with the reasons stated. |
| "the shared-`D` correction broke downstream work" | **REFUTED for the live path** — `sg2` cites both `:73,109` as one `D`; `bp2` carries `..._posenet_in_identical` **and** `..._segnet_in_identical` receipts. Doc wrong, code right. Scope: `.omx/research/*.md` grep, 11 hits, **not** an exhaustive semantic audit. |
| "`W = 1.273108215332` from memory" | **RE-DERIVED, not retyped** — `(100/(196,608·600))/(25/37,545,489)`, matches to 12 digits. The same discipline caught the stale 73 MB. |
| "my γ index needs no caveat" | **CAVEAT ADDED, travels with the number:** γ conflates direction with `‖J‖`; the receipt has no per-pair `‖J‖`. Refutes the actionable form only. §5. |

**Where my coverage is incomplete, stated rather than papered over:** (a) Row G's negative is a
literal-phrase grep over `.omx/research/*.md`, not a semantic audit and not `src/`; (b) every "not
found in prior work" here is scoped to the denominators the query surface itself reported —
`research(7425) equations(869) memory(2061) dag(915) council(292) tasks(417) docs(96)` — and my
rediscovery near-miss in §1.1 is direct evidence that recall coverage is imperfect even inside it;
**two of my own rows (B, and the pose-head SVD) turned out fully prior on a SECOND query after a
first query missed them**, so treat every open row here as "not found," never "not there"; (c) I fired no
scorer, so every row needing a forward pass is priced by inference, not measurement; (d) Row 2's
ceiling is the rate sub-gap, which assumes basis choice cannot move seg — plausible but unproven.

---

## §8 NEXT-IF-RESUMED

1. **Row 2 (basis race) first — it is the only NEW open row that needs no scorer slot.** Reduce
   the cached GT-vs-render residual against each generic basis and report capture-per-coefficient-
   byte in one common-denominator table. Pre-register the falsifier from §6 (**within 2× ⇒ basis
   choice is not the lever**). Feeds `#766`.
2. **Row 1 (separatrix student) only as a cheap falsification**, never as a build: smallest
   apparatus reaching 30% flip recovery **on the Road-hub separatrix alone** (`pc2`: Road in 87.8%
   of flips; Road↔Lane = 49.2%). Kill at >194,266 B. Do not train a segmenter.
3. **Do NOT re-open** GT1-1, GT1-3, Row A, Row C, Row D, or the pose-head anisotropy. Six of
   thirteen rows here are closed; §1.1 and §7 record exactly why, so the next arm inherits the
   closure rather than the question.
4. **Owed, low cost, not mine to commit:** fix the stale comment at
   `experiments/feedy_byteclosed_exact_row_probe.py:104` ("out=6 -> first 3" → out=12, half=6);
   correct CLAUDE.md's "~73 MB" to **94,338,452 B**.

---

*STORES CONSULTED:* `upstream/modules.py` + `upstream/frame_utils.py` +
`upstream/models/{segnet,posenet}.safetensors` (authority, source inspection);
`ddm_pb3_parametric_blind_set_20260802.md` (rank-6 reconciliation, the span-vs-coefficient law);
`reports/ddm_bp2/{reach_n600.jsonl,blind_verify.json}` + `reports/ddm_pb3/ceiling_n600.json`
(the §5 re-reduction and its control — re-derived, not recalled);
`ddm_sg2_seg_axis_actuator_20260802.md` §2/§3 (GT1-3's negative, `R` spectrum, `D` nullity);
`ddm_pc2_perclass_road_edges_20260802.md` (interiors ≈0, Road-hub 87.8% — the §4 reframing);
`ddm_gc16_upstream_score_lowering_convocation_20260731.md` §"I tried to refute" +
`ddm_ua1_frozen_scorer_weight_file_inventory_20260731.md` §5 + `ddm_pi2_posenet_inversion_20260730.md`
(the rediscovery catch); `ddm_fl1`, `ddm_pp1`, `ddm_pj2` (scanned, not load-bearing here);
`tac.canonical_equations.gap_decomposition_against_floor_20260802` (the denominator);
`docs/operating_manual_craft_handoff.md` §4 (re-derive don't recognise) and §6 (attack your own).
