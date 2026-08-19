---
arm: ddm_fo2h
generation: 2
utc: 2026-08-19
supersedes: ".omx/research/ddm_fo2h_eta_hardening_20260817.md (gen-1, status IN FLIGHT, LEG 1 verdict deliberately absent)"
charter: "operator/MAIN charter to ddm_fo2h gen-2, 2026-08-19 -- harden eta and re-run the waterfill inclusion test on measured bytes"
axis: "[macOS-CPU advisory] frozen CPU-torch SegNet + PoseNet on PyAV-lineage GT -- NEVER a score"
gt_lineage: "PyAV (av.open + frame_utils.yuv420_to_rgb). NOT DALI. up1 measured local PyAV pose at 19.09x the contest-CUDA value; this arm's own rows read 15.12x, consistent."
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "S 0.15652626435208142 @ 176,420 B [contest-CUDA T4 n600] (ddm_up3 thirteenth pointer move) -- UNMOVED by this unit"
verdict: "SEG LEG SUPPLIER-ALIVE (eta asymptote ~0.58 >> 0.5196 bar) but CHANNEL NET NON-SUPPLIER: the pose leg reversed out-of-sample and costs 2.9x-35x the seg gain"
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

| leg | pn2 n=12 | out-of-sample n=48 | effect on S |
|---|---:|---:|---:|
| seg η (pooled) | 0.6111 | **0.5804** | −0.000336 |
| pose ratio (after/before) | 0.7935 (*improves*) | **1.3725** (*worsens*) | **+0.001424** |

The seg leg supplies. The pose leg takes **3.9× more back**. Net at the pooled η:
**+0.001056 S — a net loss.** Under the alternative transfer assumption the pose leg costs
**+0.013075 S** and the loss is 35× the seg gain.

**Scope that honestly (§4.0): this pose result is on the PyAV GT lineage, and it does not
automatically transfer to the contest axis.** I first wrote that the two transfer bounds made the
verdict lineage-robust; the algebra says otherwise, and I withdraw it. The channel is
NON-SUPPLIER **as measured**; the contest-axis sign is undetermined and a DALI re-measurement is
owed. I have launched the retained-frames run that makes it a matched, same-pair comparison.

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

**The measured result (n=2 pairs, both from gen-1's own n=48 set):**

| | PyAV (local) | **DALI (shipping)** |
|---|---:|---:|
| aggregate pose ratio | 1.4024 | **1.8645** |
| pairs worsening | 2/2 | **2/2** |

**Verdict: SIGN AGREES ACROSS LINEAGES — both worsen, and the DALI degradation is 1.33× the PyAV
one.** The "phantom rescue" hypothesis — that the edit only looks damaging against an inflated
baseline — is **not** what the data show at this n. The damage is real on the axis that ships, and
larger there, which is the direction the absolute-excess reasoning predicted: the same absolute
insult is a bigger fractional one against a smaller base.

**n=2 is two pairs.** This is a matched contrast with a perfect control, not a population estimate;
it cannot carry a verdict on its own and I am not asking it to. What it does is remove the
*rescue* from the live hypotheses and put the burden back on anyone who wants to argue the channel
survives on the contest axis.

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

**SEG LEG: SUPPLIER-ALIVE.** η_∞ = 0.580 (1σ lower 0.5657, 2σ lower 0.5519) against the frozen
0.5196 bar; the curve is flat across the last 32 out-of-sample pairs. The charter's question is
answered in the affirmative and the number is now n=48 out-of-sample, not n=12.

**CHANNEL: NET NON-SUPPLIER**, +0.001056 S (ratio-transfer) to +0.012707 S (absolute-transfer),
against a −0.000336 S seg gain. **verdict_scope: INSTANCE** — this vehicle, this described set,
this r=1 pose-null realization, this solver budget, PyAV GT lineage, n=48 out-of-sample.

**What this does NOT close.** The pose-null *mechanism* is not refuted as a family; one solver
budget at r=1 is refuted at delivering it. pn2's n=12 and this n=48 disagree in **sign** on pose,
which is itself the finding that the pose axis needs a live-population measurement before any
channel in this family is priced again.

## §7 Retained payloads

Root `/Volumes/APDataStore/pact/ddm_fo2h_eta_hardening/`. Gen-1's 256 files preserved untouched;
gen-1's verdict JSON copied to `FO2H_ETA_ADJUDICATION.gen1_n48.json`
(sha `7e154b2c7a4cda3c…`) **before** any gen-2 rerun could overwrite it. Gen-2 adds
`FO2H_SAMPLE_GEN2.json` (the seeded draw), `null_shardC/`, `null_shardD/`.

*(§2 n-extension, §8 and the receipt update land when shards C/D complete — this file is a durable
checkpoint written incrementally, per the gen-1 lesson.)*
