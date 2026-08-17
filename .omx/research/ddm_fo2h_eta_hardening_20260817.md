---
arm: ddm_fo2h
utc: 2026-08-17
charter: "operator/MAIN charter to ddm_fo2h (the eta-hardening follow-on from FO-1), 2026-08-17"
axis: "[macOS-CPU advisory] frozen CPU-torch SegNet + PoseNet -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "INSTANCE on the hv1 ep0634 vehicle at the measured n; formulation where named"
tokens: "[no-triality] [p0-ledger-ok]"
status: "IN FLIGHT -- LEG 2 and LEG 3 are COMPLETE and final; LEG 1's n=48 solve is still filling. This file is a durable checkpoint, not the final verdict."
---

# ddm_fo2h — hardening the one number the supplier margin rests on

STORES CONSULTED: `ddm_fo1_waterfill_real_coder_20260817.md` (§5 break-even, §7 the owed
instruction this arm executes) · `ddm_pn2_posenull_seg_channel_20260817.md` (§0.3 the regression
series, §3 the matched A/B, §4 the snap confound, §5 joint arithmetic) ·
`ddm_sr1_manufactured_seg_recovery_20260816.md` + `SR1_WATERFILL.json` · rt1's retained
`eta_gate_null` rows and `argmax_base` / `flip_mask_vs_gt` / `free_band_mask` /
`flip_target_class` · canonical equation `seed_ensemble_falsifier_band_v1` (read at source) ·
memories [[m88]] [[m96]] (prefix genus) · [[et4]] (thread count is part of the instrument).

## STATUS

**LEG 2 and LEG 3 are complete and final.** LEG 1's n=48 out-of-sample solve was still filling
when this checkpoint was written; its verdict is deliberately absent rather than quoted early.

## LEG 2 — the bar was set by an unoptimized selection (COMPLETE)

fo1 closed its memo with one instruction: *"Price [the cell-set side info] inside the
waterfill's inclusion test, not after it."* This leg does that, and re-scores every candidate
selection with **measured, round-trip-verified coder bytes** instead of sr1's ideal per-cell
entropy.

**Controls first — three independent proofs this is the same object fo1 and sr1 measured:**

| control | result |
|---|---|
| reconstructed `cell_band_px` / `cell_flip_px` vs sr1 retained | **byte-identical** (sums 2,551,464 / 34,666) |
| sr1's exact 41-cell selection through my fast path | **M8 = 4,123 B, T2 = 185 B — fo1's numbers exactly** |
| the value-ratio ranking places sr1's 41 cells as a **prefix** | true (so the prefix family is the right one) |
| round-trip decode vs truth, at the incumbent | same sha256 `8bca66ec89eb830a…` — **fo1's own reported sha** |
| determinism repeat, 56 levels in common across two runs | **0 sha mismatches** |
| whole-band endpoint (m=74) vs rt1's describe-everything | 33,496 B / break-even 0.7590 vs rt1's 33,235 B / 0.7531 — agrees within 0.8% on a different coder pair |

**The result.** Only **74 of the 1,200 cells are live**, so sweeping every prefix m = 1…74 is the
*complete* enumeration of the family, not a sample. Coding all 74 costs 44 s.

| | cells | flips | payload B | + side info | break-even η |
|---|---:|---:|---:|---:|---:|
| incumbent (sr1 / fo1) | 41 | 6,512 | 4,308 | 4,317.6 | **0.5208** |
| lowest break-even | 9 | 175 | 81 | 86.5 | **0.3881** |

**The channel supplies at any η above 0.3881 once the selection is re-optimized, against 0.5208
for the incumbent — a 25.5% reduction in the η the channel needs.** That is the leg's headline,
because η is exactly the quantity LEG 1 is hardening.

Best selection at each η, all on real bytes:

| η | best m | net ΔS | incumbent-41 net ΔS | improvement |
|---:|---:|---:|---:|---:|
| 0.5196 (fo1's frozen bar) | 28 | **−0.000159** | +0.000007 | +0.000166 |
| 0.5651 (pn2 unprojected) | 34 | −0.000305 | −0.000245 | +0.000061 |
| 0.6111 (pn2 projected) | 44 | −0.000511 | −0.000499 | +0.000012 |
| 0.6235 (sr1 selection η) | 46 | −0.000640 | −0.000567 | +0.000073 |
| 1.0000 | 67 | −0.007625 | −0.002645 | +0.004979 |

Read the first row carefully: **at exactly the frozen bar the incumbent is a non-supplier by
construction (+0.000007 S), while a 28-cell selection supplies −0.000159 S.** Re-optimizing does
not merely improve the margin there, it changes the sign.

**fo1's §7 warning is structurally void, and that is a finding, not a dismissal.** fo1 feared the
cell-set side info would grow to ~385 B at 300 cells and eat a 756 B margin. There is no
300-cell regime: only 74 cells are live, and cell liveness (`band_px > 0`) is a deterministic
function of the decoded label field over the whole clip, so the receiver derives the live set
free and the encoder names an m-subset of **74**, not of 1,200. Priced as an exact combinatorial
rank, the side info is **8.74 B at the incumbent and never exceeds 8.82 B anywhere in the
family** — 0.2% of the payload. (The 1,200-universe price is carried alongside and peaks at
49.56 B, so the cheaper figure is never taken on trust.)

**The honest limit, measured not asserted.** sr1's ranking is by *ideal* per-cell entropy. In
real coder bytes the marginal cost along that order is **not monotone: 31 inversions in 69
single-cell bundles.** The biggest bundle (cell 34, the Lane-vs-Road horizon cell) costs
0.6798 B/flip and pays whenever η > 0.534, while cells 31 and 33 cost 1.20–1.23 B/flip and only
pay above η ≈ 0.95. Prefixes of this ranking therefore **do not exhaust the family** — a greedy
re-ranking on measured marginal cost could beat every row above. That is an owed follow-on this
arm did not run, and it is why the verdict_scope below says *formulation*.

## LEG 3 — the snap confound at full n (COMPLETE; independently confirms MAIN's closure)

pn2 left FO-2 filling at n=7. It is now 12/12. The three-way decomposition on the complete
sample:

| arm | support | projection | pooled η | d_pose aggregate |
|---|---|---|---:|---:|
| `free` | pixel | none | 0.5651 | ×4.609 |
| `free --snap-support true` | 2×2 blocks | none | **0.5365** | ×7.243 |
| `null` | 2×2 blocks | **pose-null** | 0.6111 | ×0.793 |

- snap alone: **−0.0286 η** (pn2 read −0.0029 at n=7)
- projection alone, at matched support: **+0.0746 η**
- the projection carries **162.1%** of the observed gain; the snap carries −62.1%
- on pose the snap alone is **1.57× worse**; the projection alone is **9.13× better**

**The confound runs against the claim harder at full n than pn2's n=7 read showed.** Restricting
my computation to pn2's 7 pairs reproduces their reported figures exactly (0.5588 / 0.5559 /
0.6382; snap −0.0029, projection +0.0824, share 103.7%), which is the receipt that I am
computing the identical statistic.

MAIN landed this same closure independently at `99ecccf621` with the same four numbers. **The
distinct thing this arm adds is the provenance receipt underneath it.** Arm B's retained verdict
does not record a `snap_support_flag` at all — the field postdates the run — so nothing on disk
showed that arm B was actually the snapped arm rather than a duplicate of arm A. I re-ran pair 33
under `--mode free --snap-support true` and it reproduces arm B's row exactly (`flips_after` 20,
`fixed` 46, `introduced` 17, η 0.5918367346938775, `d_pose_after` to 16 digits) and does **not**
reproduce arm A. The closure rests on a fact that is now measured rather than assumed.

## Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_fo2h_eta_hardening/`; full sha manifest in `RECEIPT.md`.
Every one of the 74 candidate levels retains its mask payload, target payload and selected-cell
set — not only the winner's — plus the decoded round-trip beside its truth field for the
incumbent and the lowest-break-even selection, so the round trip is provable from disk without
re-running the coder.

## Tools landed

- `experiments/ddm_fo2h_waterfill_measured_bytes.py` — LEG 2. Imports fo1's `code_mask` /
  `code_target` verbatim; reimplements no coding.
- `experiments/ddm_fo2h_eta_adjudicate.py` — LEG 1 adjudication against the frozen bar.
- `experiments/ddm_rt1_eta_gate_pose_constrained.py` — additive `--exclude-pairs`, `--pairs`,
  `--resume`; legacy draw verified byte-identical to the pre-change form, and `threads` added to
  the receipt because thread count is part of the forward instrument.
