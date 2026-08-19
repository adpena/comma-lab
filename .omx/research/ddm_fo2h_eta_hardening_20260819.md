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
**+0.001056 S — a net loss.** Under the alternative lineage-transfer assumption the pose leg costs
**+0.013075 S** and the loss is 35× the seg gain. **Both bounds are positive**, so the joint
verdict does not depend on resolving the lineage question.

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

**It does not matter for this verdict.** Both bounds are positive and both exceed the seg leg's
−0.000336. The channel is a net loss under either. The lineage question changes the *magnitude* of
the loss by 12×; it does not change the sign. That is the useful shape of this result — the
verdict is robust to the one thing I could not measure locally.

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
