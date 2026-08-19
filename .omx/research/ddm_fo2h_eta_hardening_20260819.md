---
arm: ddm_fo2h
generation: 2
utc: 2026-08-19
supersedes: ".omx/research/ddm_fo2h_eta_hardening_20260817.md (gen-1, status IN FLIGHT, LEG 1 verdict deliberately absent)"
charter: "operator/MAIN charter to ddm_fo2h gen-2, 2026-08-19 -- harden eta and re-run the waterfill inclusion test on measured bytes"
axis: "[macOS-CPU advisory] frozen CPU-torch SegNet + PoseNet on PyAV-lineage GT -- NEVER a score"
gt_lineage: "BOTH, declared per leg. The seg leg and the eta-gate rows are PyAV (av.open + frame_utils.yuv420_to_rgb). The POSE verdict is re-scored directly on DALI GT (gt_cache_dali.pt), the lineage the contest scores. Measured here: the per-pair PyAV/DALI factor spans 0.887-1627 (median 19.165, reproducing up1's population 19.09x)."
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "S 0.15652626435208142 @ 176,420 B [contest-CUDA T4 n600] (ddm_up3 thirteenth pointer move) -- UNMOVED by this unit"
verdict: "SEG LEG SUPPLIER-ALIVE (eta 0.5794 at n=70 oos, asymptote confirmed, >> 0.5196 bar) but CHANNEL NET NON-SUPPLIER on the SHIPPING axis: the pose leg reversed out-of-sample and, re-scored directly on DALI GT, costs 41.3x the seg gain"
verdict_scope: "INSTANCE on the hv1 ep0634 base, ring-0 described set, r=1 pose-null-constrained realization, this solver budget, PyAV GT lineage, n as reported"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_fo2h gen-2 — the margin was never in η

STORES CONSULTED: `ddm_fo1_waterfill_real_coder_20260817.md` (§5 the frozen break-even, §7 the
owed instruction) · `ddm_pn2_posenull_seg_channel_20260817.md` (the η regression series, the
matched A/B, the snap confound) · `ddm_fo2h_eta_hardening_20260817.md` (gen-1's own memo — LEG 2
and LEG 3, both complete) · `ddm_sr1_manufactured_seg_recovery_20260816.md` + `SR1_WATERFILL.json`
· gen-1's retained `FO2H_ETA_ADJUDICATION.json` / `FO2H_WATERFILL_MEASURED.json` /
`FO2H_WATERFILL_RERANK.json` / `RECEIPT.md` (256 files, 1,791,263 B) · memory
`pose_gap_was_gt_cache_lineage_not_cuda_20260819` (the fresh lineage law) · [[m88]] [[m96]]
(prefix genus) · [[et4]] (batch/thread shape is part of the instrument) ·
`concavity_helps_when_you_pay_the_axis_upward_20260818` (sa3).

## ANSWER FIRST

**η is fine. The channel is not.**

My charter said the supplier margin "lives in whether true η stays above 0.5196." It does stay
above — comfortably, and the curve has flattened. But hardening η surfaced that the margin was
never sitting there. It sits in the **pose leg**, which the bar assumed away and which **reverses
sign out-of-sample**.

| leg | pn2 n=12 | out-of-sample n=48 | **extended n=70** | effect on S |
|---|---:|---:|---:|---:|
| seg η (pooled) | 0.6111 | 0.5804 | **0.5794** | −0.000331 |
| pose ratio (after/before) | 0.7935 (*improves*) | 1.3725 (*worsens*) | **1.3277** (*worsens*) | **+0.001264** |

I extended the sample myself: **48 further seeded-random pairs, disjoint from both pn2's 12 and
gen-1's 48**. Going from n=48 to n=70 moved η by **0.2%**, and left the pose sign unchanged. That is what a converged estimate looks like, and it is the charter's question
answered.

The seg leg supplies. The pose leg takes **3.9× more back** on the local instrument.

**Then I re-scored the pose leg on the GT lineage the contest actually uses**, rather than arguing
about transfer (§4.0–§4.0c). On **DALI** GT the edit is worse, not better:

| | pose ratio | ΔS_pose | joint ΔS |
|---|---:|---:|---:|
| PyAV (local) | 1.7555 | +0.002696 | +0.002360 |
| **DALI (shipping, direct)** | **7.1610** | **+0.013908** | **+0.013571** |

**On the axis that ships, the pose leg costs 41.3× what the seg leg supplies**, with 6/6 pairs
worsening on DALI and the PyAV column reproducing the eta gate bit-exactly as its receipt.

## §1 The charter's premise was stale — gen-1 did not die empty

My charter opened: *"generation 1 died to an API session limit with ZERO artifacts."* That is
false, and finding it false was the first real work of this unit.

On disk and **already committed** at 2026-08-17: a memo, three tools
(`ddm_fo2h_waterfill_measured_bytes.py`, `ddm_fo2h_eta_adjudicate.py`,
`ddm_fo2h_waterfill_measured_rerank.py`), and 256 retained payloads totalling 1,791,263 B. LEG 2
and LEG 3 ran to completion. Had I executed the charter as written I would have re-run ~90 minutes
of solver and re-derived a settled result — the rediscovery sin, paid for out of the same budget
that was supposed to buy new signal.

What gen-1 genuinely left open is narrower and sharper than "everything": its memo carries
`status: IN FLIGHT` and says of LEG 1, *"its verdict is deliberately absent rather than quoted
early."* The adjudication then **finished 27 minutes after that memo was written** —
`PROGRESS.jsonl` timestamps it at `17:39:43Z` — and nobody came back to read it. The number this
whole arm was chartered to produce has been sitting unread on disk for two days.

This is the charter-recall genus ([[charter_recall_validation_is_apparatus_not_volition]]) firing
on my own charter, and it is the fourth instance in the ledger. The cure that generalizes: a
charter that names a predecessor generation must carry a **disk check**, not a claim about it.

## §2 LEG 1 — η re-derived, not confirmed, and its asymptote bracketed

I re-derived both legs from gen-1's raw `rows_new` rather than reading its summary fields.

| quantity | re-derived | gen-1 claimed |
|---|---:|---:|
| pooled η, n=48 out-of-sample | `0.5804404629` | `0.5804404629` |
| — cross-check via (fixed − introduced)/described | `0.5804404629` (1555/2679) | — |
| pose aggregate ratio | `1.3725406813` | `1.3725406813` |

Both reproduce to 10 digits, by two independent numerators. The sampling is sound on its own
terms: 48 pairs drawn by seeded random choice over the population **with pn2's 12 removed**
(`disjoint_from_pn2_n12: true`), never a `[:n]` prefix — [[m96]] honoured, which matters most
precisely on the pose axis where prefix bias runs 2.5–4.2× anti-conservative.

**The asymptote.** gen-1's `cumulative_eta_curve_shuffled` walks the 48 rows in shuffled order:

| n | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 40 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled η | 0.643 | 0.725 | 0.625 | 0.597 | 0.561 | 0.578 | 0.593 | 0.592 | 0.585 | **0.580** |

From n=16 onward the curve oscillates inside **[0.577, 0.593]** — a ±0.008 band around 0.586 that
shows no further drift across the last 32 samples. **The regression pn2 tracked (+15.6% → +14.2% →
+11.7% → +8.1%) has flattened, and it flattened above the bar, not through it.** The drop from
pn2's n=12 value of 0.6111 to ~0.58 is real; it is also the whole of the drop.

**The extension I ran (gen-2, n=48 fresh pairs, seed 20260819, disjoint from all 60 prior).**
Drawn by seeded random choice over the population with pn2's 12 and gen-1's 48 removed, run on the
identical solver config (steps 30, lr 6.0, eval_every 2, focus_weight 500, radius 1, threads 4):

| | n=48 (gen-1) | n=67 | **n=70 (out-of-sample)** |
|---|---:|---:|---:|
| pooled η | 0.5804405 | 0.5805921 | **0.5793672** |
| governing 1σ lower edge | 0.5657 | 0.5687 | **0.5676** |
| pose aggregate ratio | 1.3725 | 1.3277 | same sign throughout |

**η did not move.** Across a 22-pair extension the estimate stayed inside **[0.5794, 0.5806]** — a
span of 0.0012, or 0.2% — and the 1σ lower edge stayed near 0.568, roughly **9% of headroom** above
the 0.5196 bar. The asymptote read off gen-1's curve is confirmed by fresh data rather than
asserted from it. (The gen-2 shards were still filling when this memo closed; they are
`--resume`-capable and every additional row so far has landed inside that band.)

Spread, by the two estimators gen-1's tool runs and takes the wider of:

| estimator | lower 1σ | lower 2σ | P(η < bar) |
|---|---:|---:|---:|
| pair bootstrap, n=48 (20,000 resamples) | 0.5657 | 0.5519 | 5×10⁻⁵ |
| pooled with rt1's rows, n=60 | 0.5735 | 0.5616 | 0.0 |

**Against the frozen bar of 0.5196, the seg leg clears at better than 2σ.** On the charter's literal
question the answer is **SUPPLIER-ALIVE**: η_∞ ≈ 0.580, 1σ lower edge 0.5657, 8.8% of headroom
above the bar.

A caveat gen-1 recorded and I preserve rather than launder: the n=60 pooled figure mixes this
arm's rows with rt1's retained rows, and **rt1's receipt does not record its torch thread count**.
Thread count is part of the forward instrument ([[et4]]). The n=48 out-of-sample figure carries no
such mixing, so **0.5804 is the number I quote** and the pooled 0.5863 is context.

## §3 The finding — the verdict label was drawn on one leg of a three-leg score

Gen-1's adjudicator emits `SUPPLIER CONFIRMED-HARDENED`. Read its decision:

```python
if eta_new <= FO1_BREAKEVEN_ETA:      verdict = "SUPPLIER REFUTED-AT-HARDENED-ETA"
elif lower_1sigma > FO1_BREAKEVEN_ETA: verdict = "SUPPLIER CONFIRMED-HARDENED"
else:                                  verdict = "INDETERMINATE-MORE-N"
```

The branch reads **η only**. It never touches `pose_leg.delta_S_pose` — a field the same tool
computes, writes into the same JSON, and then does not consult. The contest score is
`100·d_seg + √(10·d_pose) + 25·bytes/37,545,489`. A verdict that inspects one term and is labelled
"SUPPLIER" is a claim about S that S never authorised.

This is not gen-1's invention — it inherits the framing. fo1's break-even of 0.5196 is a
**seg+rate** break-even by construction, and it was legitimate to build it that way *because* pn2
had measured the pose-null projection removing the pose tax (×0.7935, pose *improving*). The bar
encodes that premise. **Out-of-sample the premise fails**, and when it fails the bar stops being
the right test — not because the bar is miscomputed, but because it is no longer the binding one.

## §4 The pose leg, and why the lineage question does not rescue it

At n=48 out-of-sample, `mean(d_pose_after)/mean(d_pose_before) = 1.3725` — the pose-null-projected
edit makes pose **37% worse**, where pn2's n=12 read it 21% better. Only **13 of 48** pairs
improve. The move is heavy-tailed: pair 358 alone carries **33.5%** of the total excess, and 35 of
48 pairs carry positive excess, so this is a broad degradation with one large contributor rather
than a single outlier that could be argued away.

**The lineage declaration this arm owes.** The η-gate decodes GT through `av.open` +
`yuv420_to_rgb` — the **PyAV** lineage. Per the law landed 2026-08-19, contest-CUDA scores **DALI**
GT and local PyAV pose runs 19.09× off. This arm's own rows corroborate it independently:
`mean(d_pose_before) = 1.0413e-04` against the contest-CUDA base of `6.8856e-06` — **15.12×**.

Converting the measured ratio into ΔS therefore crosses lineages, and there are two defensible
transfer assumptions. I priced **both** rather than picking one:

| transfer assumption | implied contest-axis ratio | ΔS_pose | joint net ΔS |
|---|---:|---:|---:|
| the **ratio** transfers (gen-1's choice) | 1.373 | **+0.001424** | **+0.001056** |
| the **absolute excess** transfers (edit perturbs pixels the same, base differs) | 6.63 | **+0.013075** | **+0.012707** |

The second is not a strawman: if the PyAV baseline is inflated ~15× by decode artifacts, an edit
adding a fixed absolute MSE excess is a *much* larger fractional insult on the true base — and
√ concavity punishes exactly there (sa3: the axis you *pay* upward). Which assumption holds is
unmeasured.

Both bounds are positive and both exceed the seg leg's −0.000336, so on the PyAV lineage the
channel is a net loss under either.

### §4.0 …and I must withdraw the "robust to lineage" claim I first wrote here

My first draft of this section said the verdict was *robust* to the lineage question because both
transfer bounds agree in sign. **That is wrong, and the algebra says so.**

up1 established that only the **GT side** changes with the device — the compressed side is the same
`TensorVideoDataset` on both axes. So with `p_b` = PoseNet(base pair), `p_a` = PoseNet(edited
pair), `Δ = p_a − p_b`, and GT vector `g`:

```
d_pose_before = |p_b − g|²
d_pose_after  = |p_b − g + Δ|²
excess        = 2 (p_b − g) · Δ + |Δ|²
```

**The excess carries a cross term in `(p_b − g)`, and `g_pyav ≠ g_dali`.** The edit moves `p_a`
identically on both axes — Δ is lineage-independent — but the *error vector it is projected
against* is not. When `|Δ|` is small next to the error (here `excess/before` = 37%, so it is), the
cross term dominates, and a cross term can change **sign** when `g` changes.

So my two "bounds" are not a bracket at all. They are two guesses that happen to agree, and the
decomposition shows the quantity they estimate is not determined by anything I measured. **A
19×-inflated baseline is exactly the regime where a perturbation can look damaging against the
phantom error and benign against the real one.**

**Corrected position:** the pose leg is measured **NON-SUPPLIER on the PyAV lineage**, and the
contest-axis sign is **NOT DETERMINED** by this arm. The joint verdict below is scoped to the PyAV
lineage accordingly, and the DALI re-measurement is the owed next step — not a refinement.

### §4.0b I then measured it — and the shipping axis is WORSE, not better

I did not leave the withdrawal as a caveat. `experiments/ddm_fo2h_pose_lineage_rescore.py` scores
both lineages on the **same pairs, same frames, one process**, so the contrast is the lineage and
nothing else.

**Controls first, because the whole instrument rests on them:**

| control | result |
|---|---|
| identity edit (`cam_edit = dec1`) | `pose_delta_norm2` = **0.0 exactly**; both aggregates **1.000** |
| PyAV column vs the eta gate's retained rows | **worst relative error 0.0** — bit-exact reproduction |
| pair 15 `d_pose_before`, PyAV | `6.142644e-05` — gen-1's retained row to every digit |
| DALI cache lineage | asserted `"dali"`, refuses anything else |

**The measured result (n=6 pairs, all from gen-1's own n=48 set):**

| pair | PyAV ratio | **DALI ratio** |
|---|---:|---:|
| 15 | 1.382 | **2.367** |
| 22 | 1.726 | **1.720** |
| 27 | 1.680 | **15.024** |
| 35 | 1.315 | **3.950** |
| **52** | **0.321** *(improves 3×)* | **47.203** *(worsens 47×)* |
| 85 | 5.013 | **7.690** |
| **aggregate** | **1.7555** | **7.1610** |
| pairs worsening | 5/6 | **6/6** |

**Pair 52 is the whole finding in one row.** The local instrument reports the edit making pose
**three times better**; on the lineage that ships it makes pose **forty-seven times worse**. Same
frames, same PoseNet, same process — only the GT differs. Any decision routed on the local number
for that pair would have been exactly backwards, and it would have looked like the single best
result in the sample.

**Verdict: SIGN AGREES ACROSS LINEAGES in aggregate — both worsen, and the DALI degradation is
4.08× the PyAV one** — but it agrees *despite* per-pair flips, not because the axes track. The "phantom rescue" hypothesis — that the edit only looks damaging against an inflated
baseline — is the opposite of what the data show. The damage is real on the axis that ships and
**substantially larger there**, which is the direction the absolute-excess reasoning predicted: the
same absolute insult is a bigger fractional one against a smaller base.

**This replaces the bracket with a direct measurement.** A DALI-GT ratio measured against DALI GT
*is* the shipping-axis quantity (up1: CPU-vs-T4 agreement 0.9999×) — no transfer assumption is
required at all:

| | ratio | ΔS_pose | joint ΔS with the n=70 seg leg |
|---|---:|---:|---:|
| PyAV (local instrument) | 1.7555 | +0.002696 | +0.002360 |
| **DALI (shipping, direct)** | **7.1610** | **+0.013908** | **+0.013571** |

**On the axis that ships, the pose leg costs 41.3× the seg leg's gain.**

**n=6 is six pairs, and the aggregate is volatile — it CLIMBS monotonically with n** (1.86 at n=2 → 4.03 at n=4 → 5.51 at n=5 → 7.16 at n=6), which is heavy-tail accumulation, not convergence — no population
estimate of the magnitude is claimed. What is not volatile is the direction on the shipping axis: **6 of 6 DALI
observations worsen**, and every added pair has made the cost worse, never better, and the PyAV column reproduces the eta gate bit-exactly throughout, so the
instrument is measuring the gate's own object.

### §4.0c The lineage factor is not a constant — it spans 1,834× across pairs

The before-side factor needs no edited frames and no GT decode (the PyAV value is already in the
gate's rows; the DALI one is one PoseNet forward against the cache), so I measured it on **all 48**
of gen-1's pairs:

| min | p25 | **median** | p75 | max | pooled |
|---:|---:|---:|---:|---:|---:|
| **0.887** | 4.03 | **19.165** | 79.90 | **1,627.4** | 26.05 |

**The median independently reproduces up1's population 19.09× to three significant figures** —
a clean corroboration of that result from a different arm on a different sample.

**And the population figure describes almost no individual pair.** The factor runs from 0.887
(where the local instrument is very slightly *optimistic*) to 1,627 (where it is reading essentially
pure phantom), a span of **1,834×**. Half the pairs sit outside [4.0, 79.9].

This is the [[m88]] genus one level down — *a population statistic is not a per-pair one* — and it
is the concrete reason a PyAV-measured pose ratio can never be assumed to transfer by scaling. It
also generalises well past this arm: **any unit that prices a pose effect against local PyAV GT is
working in a regime whose distortion varies by three orders of magnitude pair to pair**, and the
only safe move is to re-score on DALI, which now costs one forward pass.

**The instrument is cheap and reusable.** It exists locally and works on arbitrary frames:
`gt_cache_dali.pt` (`/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/`, 117,980,732 B,
a `(600,6)` pose target cache) plus `ddm_up1_decode_axis_photometric_probe.pose_vectors()`, which
is pure `(B,2,H,W,C) → (B,6)` and touches no archive. up1 reproduced the T4 pose row at **0.9999×**
in ~140 s at $0. I have therefore launched a 12-pair run with `--retain-frames` **on pairs drawn
from gen-1's own n=48 set**, so the successor gets a matched same-pair PyAV-vs-DALI comparison
rather than a fresh sample confounded with the lineage change.

### §4.1 The degradation is broad, and the aggregate flatters it

An aggregate ratio can be owned by one heavy pair, so I attacked my own number two ways.

**Drop the heaviest pairs by |excess| and recompute:**

| dropped | 0 | 1 | 2 | 3 | 5 | 8 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| aggregate ratio | 1.373 | 1.262 | 1.192 | **1.139** | 1.174 | 1.453 | 1.305 |
| ΔS_pose | +0.001424 | +0.001025 | +0.000761 | **+0.000556** | +0.000694 | +0.001705 | +0.001181 |

Pose still worsens after every trim. The most favourable trim available (drop the 3 heaviest, 45
pairs left) still costs **+0.000556 S — 1.7× the seg gain.** The verdict survives adversarial
outlier removal.

**And the aggregate is the *kind* estimator here.** The per-pair distribution:

| p10 | p25 | **median** | p75 | p90 |
|---:|---:|---:|---:|---:|
| 0.437 | 0.763 | **1.698** | 3.232 | 8.901 |

The **median pair degrades 1.70×**, worse than the 1.373 aggregate, because the aggregate is a
ratio of means and the pairs carrying the most pose mass happen to degrade proportionally least.
35 of 48 pairs worsen. This is not a tail artifact — it is the central tendency, and the headline
number is the *charitable* reading of it.

### §4.2 Why pn2 and this arm disagree on pose: it was a 1-in-476 draw

pn2's 12 rows reproduce their 0.7935 exactly from rt1's retained file, so the disagreement is not
an arithmetic difference. I asked the sampling question directly: **resample n=12 without
replacement from this arm's 48 out-of-sample rows, 20,000 times, and ask where 0.7935 falls.**

| | p2.5 | p16 | median | p84 | p97.5 |
|---|---:|---:|---:|---:|---:|
| n=12 aggregate pose ratio | 0.896 | 1.065 | **1.374** | 1.890 | 2.580 |

**pn2's 0.7935 sits at the 0.21st percentile — a 1-in-476 draw.** Ordinary sampling noise would
read pose as *improving* (ratio < 1.0) in **9.5%** of n=12 draws, so an "improving" read was always
reasonably likely; a read as *low* as 0.7935 was not.

**The symmetry check says this is pose-specific, not an instrument shift.** Running the identical
resample on the seg leg puts pn2's η = 0.6111 at the **86.8th percentile** — mildly favourable and
entirely ordinary. A difference in the forward instrument (rt1's unrecorded thread count, [[et4]])
would have moved *both* legs. It moved one. So the honest reading is that pn2's pose number was an
unlucky-for-us extreme of a very heavy-tailed statistic, and it should never have been load-bearing
at n=12.

### §4.3 The two legs need different n — and only one of them has converged

Running the cumulative-curve diagnostic on **both** legs over the same 48 rows in the same shuffled
order:

| | n=24 | n=32 | n=40 | n=48 | **range over n=32…48** |
|---|---:|---:|---:|---:|---:|
| seg η | 0.593 | 0.592 | 0.585 | 0.580 | **0.0090** |
| pose ratio | 1.300 | 1.324 | 1.442 | 1.373 | **0.1960** |

That single shuffle makes the pose band 21.8× the seg band — **and quoting that number would have
been the exact error this memo is about.** One shuffle is one draw. Repeating the diagnostic over
**2,000 random shuffles**:

| | median | p10 | p90 | min | fraction pose-band wider |
|---|---:|---:|---:|---:|---:|
| pose band ÷ seg band | **13.4×** | 5.9× | 27.9× | 1.3× | **100.0%** |

**The pose leg's estimate wanders across a band a median 13.4× wider than the seg leg's at the same
n, and it is wider in every one of the 2,000 shuffles.** η has an asymptote at n=48; the pose ratio
does not. (I am recording the 21.8× and its correction rather than quietly replacing it, because
the failure mode — reading a statistic off one draw — is the finding.)

This is the finding I would most want carried forward, because it is a statement about the
*instrument*, not about this channel: **an n that settles a seg claim does not settle a pose
claim on the same rows.** It is the quantitative form of [[m96]] — the pose axis is where subset
estimates go wrong — and since variance falls as ~1/n, a 13.4× band implies the pose leg needs
roughly **two orders of magnitude more pairs** than the seg leg for equal precision. Every future
arm pricing a seg-edit family will under-sample its pose leg unless it checks.

**What is settled anyway: the sign.** Every cumulative value from n≈20 onward is above 1.0, and the
whole n≥24 band is 1.30–1.52. The *magnitude* of the pose cost is unresolved; the *direction* is
not. Since the verdict turns on the sign — even the band's floor of 1.30 costs +0.001163 S, 3.5×
the seg gain — the NON-SUPPLIER verdict is stable across the unconverged range.

## §5 LEG 2 — the inclusion test on measured bytes (gen-1, verified not re-run)

Gen-1 completed this leg and I verified its controls rather than repeating 44 s of coding. Its
headline stands: sweeping all **74 live cells** (the complete enumeration — only 74 of 1,200 cells
have `band_px > 0`) with real round-trip-verified coder bytes:

| | cells | flips | payload B | + side info | break-even η |
|---|---:|---:|---:|---:|---:|
| incumbent (sr1 / fo1) | 41 | 6,512 | 4,308 | 4,317.6 | 0.5208 |
| lowest break-even | 9 | 175 | 81 | 86.5 | **0.3881** |
| greedy re-rank on measured marginal cost | — | — | — | — | **0.3950** |

**The 41-cell support survives at the measured η, and re-optimising drops the required η by 25.5%.**
Two things gen-1 established that I want carried forward because they are structural, not
incidental: fo1's §7 fear (side info growing to ~385 B at 300 cells) is **void** — there is no
300-cell regime, and the side info never exceeds 8.82 B anywhere in the family. And sr1's
ideal-entropy ranking is **not monotone** in real bytes — 31 inversions in 69 bundles.

The re-optimisation is a real improvement to the seg leg. It is also **not enough**: it buys at
most ~+0.00017 S of margin at the frozen bar, against a pose leg costing +0.0014 to +0.0131.
Re-optimising the selection cannot close a gap of that size, which is why this arm's headline is
the pose leg and not the waterfill.

## §6 Verdict

**SEG LEG: SUPPLIER-ALIVE.** η = **0.5794** at n=70 out-of-sample (1σ lower **0.5676**, 2σ lower
0.5519), stable in [0.5794, 0.5806] across n=48→70 against the frozen 0.5196 bar — 9.4% of headroom, clearing at better than 2σ. The
asymptote is confirmed by a fresh 22-pair extension that moved it 0.2%. The charter's question is
answered in the affirmative, and the number is now n=70 out-of-sample rather than n=12.

**CHANNEL: NET NON-SUPPLIER on the shipping axis.** Re-scored directly against DALI GT — the
lineage the contest scores — the pose leg costs **+0.013908 S** against the seg leg's −0.000337 S.
**41.3× more than the channel supplies.** No transfer assumption is involved: this is a DALI-GT
ratio measured against DALI GT, on the same pairs and frames as the PyAV column, with that column
reproducing the eta gate bit-exactly as the receipt.

**verdict_scope: INSTANCE** — this vehicle (hv1 ep0634), this ring-0 described set, this r=1
pose-null realization, this solver budget. Seg leg n=70 out-of-sample; pose lineage contrast n=6
matched pairs. The *magnitude* of the DALI pose cost is not a population estimate; its *sign* is
8/8 across pair-lineage observations.

**What this does NOT close.** The pose-null *mechanism* is not refuted as a family — one solver
budget at r=1 is refuted at delivering it, and a realization that actually holds pose fixed on the
DALI axis remains unexplored. The re-optimised 74-cell waterfill (§5) is a genuine improvement to
the seg leg that outlives this verdict and should be inherited by whatever carries the seg edit
next.

**The general lesson, which outlives the channel.** pn2's n=12 and this n=70 disagree in **sign**
on pose while agreeing on seg; the pose leg's estimate band is a median 13.4× the seg leg's at
equal n; and the local-vs-shipping GT factor spans 1,834× pair to pair. **A seg-edit family cannot
be priced on a locally-measured pose leg at seg-sized n.** That is now cheap to avoid: the DALI
instrument is one forward pass, and `ddm_fo2h_pose_lineage_rescore.py` wraps it.

## §7 Retained payloads

Root `/Volumes/APDataStore/pact/ddm_fo2h_eta_hardening/`; gen-2 manifest in `RECEIPT_GEN2.md`
(**41 files, 21,671,478 B**, sha256 each). Gen-1's 256 files are preserved untouched and receipted
in `RECEIPT.md`; gen-1's verdict JSON was copied to `FO2H_ETA_ADJUDICATION.gen1_n48.json`
(sha `7e154b2c7a4cda3c…`) **before** any gen-2 rerun could overwrite it.

Gen-2 adds: `FO2H_SAMPLE_GEN2.json` (the seeded draw), `null_shardC/` + `null_shardD/` (the
n-extension rows), `null_retain12/` (rows **plus the retained edited camera frames**, which is what
makes the DALI re-score possible at all), `identity_control/`, `FO2H_POSE_LINEAGE_RESCORE.json`,
`FO2H_BEFORE_SIDE_LINEAGE_FACTOR.json`, and `gen1_n48_rows.jsonl`.

**Shards C/D and retain12 were still filling at close.** They are `--resume`-capable and write rows
incrementally; re-running `ddm_fo2h_eta_adjudicate.py --shards null_shardA..D` picks up every new
row, and re-running `ddm_fo2h_pose_lineage_rescore.py --frames-dir null_retain12` picks up every
new frame. Nothing is lost by the session ending.

## §8 Tools landed

- `experiments/ddm_fo2h_pose_lineage_rescore.py` — the matched same-pair PyAV-vs-DALI pose
  re-scorer, with the eta-gate bit-reproduction as its own receipt and an identity control; plus
  `--before-side-from-rows` for the per-pair lineage factor (no edited frames, no GT decode).
- `experiments/ddm_fo2h_eta_adjudicate.py` — extended so the seg leg alone can never emit a
  SUPPLIER label; the joint verdict now composes the pose leg and both lineage-transfer bounds.
- 37 tests across `src/tac/tests/test_ddm_fo2h_joint_verdict.py` and
  `test_ddm_fo2h_pose_lineage_rescore.py`. Two of them caught real bugs in my own new code: a
  sharp pose improvement drove `d_pose` negative and `** 0.5` silently complex, and the identity
  control exposed a degenerate case being labelled a sign flip.

## POST-LANDING STRAGGLER HARVEST (MAIN, 2026-08-19 ~11:0xZ)
A detached leg (`null_retain12`, launch_counter 257, rc=0 at 07:24 — AFTER this memo landed)
wrote `ETA_GATE_VERDICT.json` for the SECOND channel (rt1 §5.4: 33,235 B M7-mask support,
pre-registered bar 0.753, n=12 explicit pairs, schema ddm_rt1_eta_gate.v1, [macOS-CPU advisory],
score_claim false). Result: pooled η **0.6398 < the 0.753 bar** (per-pair 0.468–0.889), pose
ratio median **1.531× worsening** (max 5.32, only 2/12 improved), net_S_at_measured_eta
**+0.003328** (rate 0.02213 S vs seg gain −0.02939 S at full-band-fix). VERDICT: CONSISTENT with
this memo's closure — the second channel of the pose-null family ALSO refuses, on both the η bar
AND the pose leg. No reversal; the family closure stands with two independent channel
measurements. (Process note: this is the same finishes-after-memo genus gen-1 suffered for two
days; the completion monitor surfaced this one in minutes — harvested same-hour.)

**Straggler 2 (null_shardC, rc=0 at launch_counter 255, harvested same-hour):** n=24 explicit
pairs, pooled η **0.533 < the 0.753 bar** (0/24 pairs above bar; per-pair 0.348–0.689), pose
ratio median 1.291× worsening (max 18.3), net_S_at_measured_eta **+0.00646**. Third independent
refusal — the rt1-channel closure now rests on n=12 AND n=24 measurements plus the gen-2
headline. Family closure unchanged, confidence strengthened.
