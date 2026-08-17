---
arm: ddm_sf1
title: "The -2,874 B does not exist and what remains is worth 0 B. The sum is undecodable (two mutually exclusive receiver formats), so the honest ceiling is -2,051 B, and its MEASURED pose price on the correct GT axis is +0.062 S -- 6.5x the whole remaining gap, the wrong way. Re-pricing every mp2 row against the authority-tracking GT makes the refusal 1.45x STRONGER, not weaker. The map, over an 18-group partition of all 576 FiLM rows: raw damage spans 192.9x but per unit perturbation energy only 5.65x, so energy is the controlling variable. The one free lever found and REALISED: the shipped selector ranks rows WITHIN each tensor while blocks.1 carries 152x the row energy of blocks 2-3, and ranking globally at byte parity cuts d_pose 18.0-22.3x across two seeded subsets -- taking the best cell from 3,942x over break-even to 184x over. Mechanism: keep87 removes 0.00078% of the semantic weight energy and moves d_pose 76.8x."
utc: 2026-08-17
parent: ".omx/research/ddm_ra2crr_priced_pose_null_and_pool_census_20260816.md"
fire_order: "ra2crr NEXT_IF_RESUMED row 1b (this unit also closes ra2crr row 5, unowned until now)"
axis: "[macOS-CPU advisory; authority-tracking DALI GT, MEASURED by pi2 at 1.00081x vs contest-CUDA] -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "see section 7; every claim carries its own scope"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_sf1 — the semantic/FiLM row-group pose map

STORES CONSULTED, read before any code was written: the parent
`ddm_ra2crr_priced_pose_null_and_pool_census_20260816.md` (§5b, NEXT_IF_RESUMED rows 1b/4/5) ·
`ddm_mp2_mixed_precision_receiver_close_20260815.md` ·
`ddm_1058_composition_campaign_close_20260816.md` ·
`ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md` ·
`ddm_a1s_foa_band_masked_pose_verdict_20260816.md` ·
`ddm_b2e_edit_replay_admission_verdict_20260816.md` · all six rows of
`.omx/research/falsified_premise_registry.jsonl` · `docs/operating_manual_craft_handoff.md` ·
CLAUDE.md (NO-FAKE, ALWAYS KEEP THE PAYLOAD, the verdict_scope ladder, n600-or-not-evidence,
"MPS is NEVER authority") · memories `prefix_bias_sign_inverts_between_seg_and_pose_20260803`
([[m96]]) · `cross-regime-constant-transfer-genus-finishing-stage` ·
`measured_object_vs_named_object_20260816` ·
`the_denominator_and_the_falsifier_can_both_be_vacuous_20260816`.
Source read at source, not inherited: `upstream/modules.py` · `cpr1/inflate.py` ·
`runtime/f26_inflate.py` · `experiments/ddm_mp2_semantic_receiver.py` ·
`experiments/ddm_sm3_semantic_representation.py`.

## OPTIMAL FORM

**Reference form** of a weight-space pose-sensitivity map: perturb the real weights with the real
actuator, decode through the real renderer, score with the frozen contest scorer against a
trustworthy GT, over the population, and price the bytes in the real archive format.

| delta from that reference | class | why |
|---|---|---|
| n = 120 seeded RANDOM pairs for the map groups (seed 20260817), not n600 | **SCOPE** | legal; the n600 anchors are the three re-priced mp2 archives (§2), and the n120-vs-n600 calibration is measured in §6 |
| 32-row groups, not 576 single rows | **SCOPE** | legal; finer granularity is explicitly NOT closed (§9) |
| — | **MECHANISM** | **none reduced.** The actuator is the exact prune (`_decode_row_prune` reconstructs a dropped row as `torch.zeros`, so zeroing IS pruning); the render is the shipped renderer at batch = 1, proven bit-identical at the zero perturbation; the scorer is the frozen CPU PoseNet with `preprocess_input` verbatim; the GT is the authority-tracking cache; the byte credit is brotli-q11 on the real SM3R packet, and the archive-delta == section-delta identity is MEASURED, not assumed |

No TOY-BRACKET declaration is owed: no mechanism was reduced. Provenance pins: archive sha
`80d9c8c6…` @ 182,759 B · tokens 117,964,800 B · GT cache sha `a91d9825…` · renderer and receiver
runtime from the custody-pinned `hv1_base_control` generation.

## ANSWER FIRST

**The honest byte number the -2,874 B becomes is 0 B, and it fails at two independent stages.**

1. **The sum is undecodable.** MEASURED at source: the mixed-q3/q4 credit rides an `SD1M` packet
   (per-tensor bit depth, no row mask); the FiLM-row credit rides an `SM3R` packet (row mask,
   every depth HARDCODED to 4 in `_decode_row_prune`). `unpack_variant_semantic_or_none` dispatches
   on ONE magic and no combined format exists, so no shipped receiver decodes a candidate carrying
   both. The -2,874 B was 823 + 2,051 — a sum mp2's own governing constraint already forbade as
   double-counting (`mp2:115`), and which I find is not merely non-additive but unreachable.
   **The honest ceiling from the banked candidates is -2,051 B = 14.23% of the gap, not 19.94%.**
2. **What remains is priced NEGATIVE.** Re-measured against the authority-tracking GT, the best of
   mp2's three scored candidates nets **+0.062227 S**. The gap is 0.009597 S. This family does not
   miss the bar — it moves the score **6.5x the whole gap in the wrong direction.**

**Re-pricing on the correct GT axis makes the refusal STRONGER by ~1.45x.** I expected the opposite
and recorded that before measuring. mp2's rows were drawn on a PyAV GT lineage whose base `d_pose`
is **21.417x** the authority value (1.4747e-04 vs 6.885576e-06 — the two-lineage instrument bug
`pi2` diagnosed). `sqrt` is concave, so the same drift costs MORE score off a smaller base:

| candidate | ΔB | ΔS_pose [PyAV, as mp2 and #1058 published it] | **ΔS_pose [authority]** | ratio |
|---|---:|---:|---:|---:|
| FiLM keep87 | −130 | +0.044268 | **+0.064411** | 1.455x |
| FiLM keep75 | −471 | +0.041573 | **+0.062541** | 1.504x |
| mixed q3/q4 | −823 | +0.047110 | **+0.066619** | 1.414x |

**The mechanism, and it is the finding.** keep87's 75 pruned rows remove **0.00078%** of the
semantic renderer's total weight energy (4.418990e-02 out of 5.649227e+03) and move `d_pose` by
**76.8x**. The pose load in these weights is wildly disproportionate to their byte value.

**The map: damage tracks ENERGY.** Across an 18-group partition covering all 576 FiLM rows, the
weight-perturbation energy spans **855.6x**, the raw `Δd_pose` spans **192.9x**, and `Δd_pose` per
unit weight energy spans only **5.65x** (per unit output-frame energy, **2.96x**). Energy is the
controlling variable; row identity is a ~5x correction on top of it. This transfers `a1s` FO-A's
camera-plane law — "pose damage tracks energy, not location" — into weight space, where it had
never been tested. By my own pre-registered bars (STRUCTURED ≥ 10x, ENERGY_LIKE ≤ 2x) both
normalisations land **INDETERMINATE**, and I do not move the bars.

**The one lever the shipped selector misses.** `pack_prune_candidate` ranks rows by norm **WITHIN
each tensor** and takes 25 from each. MEASURED: `blocks.1.film.weight` holds row-norm energy
**1.680936e+00** against **1.106068e-02** and **1.188705e-02** for blocks 2 and 3 — **152x more**.
Since the SM3R packet spends 6 B per RETAINED row, the byte credit depends only on HOW MANY rows
are dropped, so ranking globally is free. At the keep87 count a global ranking picks **0 rows from
blocks.1** (37 from blocks.2, 38 from blocks.3) and drops **24.95x less energy** for −125 B instead
of −130 B. **REALISED through the full render and scorer: `Δd_pose` falls from 5.6863e-04 to
2.5498e-05 — a 22.30x reduction at byte parity (17.99x on an independent second subset,
so the honest band is 18.0-22.3x)**, against a prediction of 2.09e-05 registered
before the measurement (realised/predicted = 1.22). That is a real, free, previously unfound
improvement to the selector: it takes the best cell from 3,942x over break-even to **184x over**.
It does not rescue the family.

**Pointer UNMOVED: hv1 ep0634, S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600].**
No dispatch, no Modal, $0. Every number is `[macOS-CPU advisory]`.

## §1 What my charter got wrong, and how I found out

My charter said row 1b's map "HAS NEVER BEEN MEASURED" and priced it at ~20% of the gap. Both
halves are stale — and one is stale the same way `ra2crr` reported about its OWN charter, one arm
earlier. Recording it plainly, because a relayed stale premise reaches the next arm:

| charter premise | status | where it fails |
|---|---|---|
| "worth about -2,874 B" | **REFUTED** | an undecodable sum of two exclusive formats (§2) |
| "roughly 20% of the whole gap" | **CORRECTED** | 14.23%, off the reachable −2,051 B |
| "held, awaiting this map" | **SUPERSEDED** | `#1058` closed the family at FAMILY scope on 2026-08-16 — the day before my charter — on three measured n600 rows |
| ra2crr §5b "the semantic renderer paints both [frames]" | **REFUTED** | it paints frame_1 only (§3) |
| `#1058` "dose-response is monotone and hopeless" | **HALF REFUTED** | hopeless yes; monotone no (§5) |

The measurement was still worth running. `#1058`'s closure rests on a mechanism sentence that is
partly wrong, its three rows are three MAGNITUDES rather than a partition, and the question "WHICH
rows, not how many" had never been asked. This arm converts a three-sample closure into a
partition-wide one on the corrected axis, and finds a free selector improvement on the way.

## §2 The −2,874 B, re-derived from the banked archives

Every generation stores its single member `p` without deflation, so **an archive delta equals its
semantic-section delta exactly**. MEASURED across all nine banked generations: ZIP overhead is a
constant 100 B and `Δarchive == Δsemantic.br` on every row. That control is what licenses pricing
an arbitrary row mask by brotli-ing one section instead of rebuilding a whole archive.

| generation | archive | semantic.br | Δ archive | Δ semantic.br | format |
|---|---:|---:|---:|---:|---|
| hv1 base | 182,759 | 34,763 | 0 | 0 | WANS1 |
| mixed q3/q4 | 181,936 | 33,940 | −823 | −823 | **SD1M** |
| FiLM keep87 | 182,629 | 34,633 | −130 | −130 | **SM3R** |
| FiLM keep75 | 182,288 | 34,292 | −471 | −471 | SM3R |
| FiLM keep62 | 182,011 | 34,015 | −748 | −748 | SM3R |
| FiLM keep50 | 181,694 | 33,698 | −1,065 | −1,065 | SM3R |
| FiLM keep37 | 181,235 | 33,239 | −1,524 | −1,524 | SM3R |
| **FiLM keep25** | **180,708** | **32,712** | **−2,051** | **−2,051** | SM3R |
| keep75∖keep87 | 182,734 | 34,738 | −25 | −25 | SM3R |

`SD1M` and `SM3R` are alternatives, not layers. **−2,051 B is the ceiling** — on a family priced
negative.

**A second measured fact that explains the poor byte yield: the format switch costs +232 B before a
single row is dropped.** An SM3R packet with NO rows pruned brotli-q11s to **34,995 B** against the
shipped WANS1 section's 34,763 B. keep87's headline −130 B is really **−362 B of pruning minus a
232 B format toll**; keep25's −2,051 B is −2,283 B minus the same toll. Every FiLM-prune candidate
pays it first. (Consequence for readers of my receipts: `delta_bytes_vs_base_packet` is
packet-relative and generous by exactly 232 B; the archive-relative credit adds 232.)

Break-even on the authority base `d_pose` = 6.885576310691407e-06:

| credit | rate credit | allowed `d_pose` | allowed `Δd_pose` | ratio of base |
|---:|---:|---:|---:|---:|
| −130 B | 8.656e-05 S | 7.0300e-06 | 1.4441e-07 | 1.0210x |
| −471 B | 3.136e-04 S | 7.4159e-06 | 5.3031e-07 | 1.0770x |
| −823 B | 5.480e-04 S | 7.8251e-06 | 9.3949e-07 | 1.1364x |
| **−2,051 B** | **1.3657e-03 S** | **9.3386e-06** | **2.4530e-06** | **1.3562x** |

Measured misses on the authority axis: keep87 **3,613x** over, keep75 **933x**, mixed q3/q4
**590x**. Extrapolating keep25's unscored row at the family's own damage level puts it **201x**
over. The miss shrinks with size and never approaches 1x.

Because `d_pose = |r + δ|²` and the authority residual `r` is tiny, the damage is essentially pure
drift energy: keep87's measured drift rms is 2.284e-02, and `2.284e-02² + 6.886e-06 = 5.286e-04`,
which IS the measured `d_pose` to four figures. So the bar restates as a drift bar: **a candidate
buying −2,051 B may move the pose vector by at most rms 1.566e-03; this family moves it 2.221e-02,
14.2x too far — 201x in energy.**

## §3 Controls — every one passed, and one refutes a relayed mechanism

| control | required | measured | verdict |
|---|---|---|---|
| zero-perturbation re-render vs the shipped raw | bit-identical | **12/12 frames bit-identical** (seeded random, seed 8170001) | **PASS** |
| the null group's `Δd_pose` | exactly 0 | **+0.000000e+00** | **PASS** |
| independent n600 base `d_pose` vs ra2crr's | agree | 6.885576310691407e-06 vs 6.885595058208011e-06 — **2.72e-06 relative** | **PASS** |
| `frame_0` invariance under a semantic edit, n600 x 3 candidates | identical | **600/600 pairs, max abs diff 0**, all three | **PASS** |
| my SM3R weight reconstruction vs the shipped packer's own `expected` | equal | **max abs diff 0.0** on the real hv1 weights | **PASS** |
| my SM3R byte pricing vs mp2's shipped archive | equal | keep87 packet brotli **34,633 B == the shipped 34,633 B** | **PASS** |
| GT cache sha vs the `pi2` pin | `a91d9825…` | matched | **PASS** |
| archive sha / bytes vs the frontier pin | `80d9c8c6…` / 182,759 | matched | **PASS** |

**The `frame_0` control refutes `ra2crr` §5b at n600.** That memo explains mp2's pose movement with
"`d_pose` reads both rendered frames; the semantic renderer paints both". At source,
`cpr1/inflate.py::render_video` writes `output[2*i + 1]` from the semantic renderer and
`output[2*i]` from the carrier — **the semantic renderer paints frame_1 only** — and the paragraph
is internally inconsistent, since it also says the carrier renders frame_0 and both cannot own it.
Measured: `frame_0` is bit-identical across all 600 pairs for all three semantic candidates. The
CONCLUSION survives — `PoseNet.preprocess_input` keeps both frames, so a frame_1-only edit moves
input channels 6-11 and does move pose — but the stated mechanism does not. Tenth instance of
`measured_object_vs_named_object_20260816`, and the exact mirror of registry row 4
(`frame0_is_the_pose_carrier_only_20260817`): same axis, opposite error. Registered as row 7.

## §4 The map — 18 groups, all 576 FiLM rows, seeded random n = 120

`Δd_pose` against the same-subset base 6.870975e-06. Groups are `b{block}_{scale|shift}_{norm
tercile}`, 32 rows each, disjoint, exhaustive.

| group | Δd_pose | per unit weight energy | per unit output energy |
|---|---:|---:|---:|
| `b2_scale_lo` | 1.0261e-05 | 1.0708e-02 | 5.0346e-13 |
| `b2_shift_lo` | 1.0345e-05 | 1.1779e-02 | 4.4762e-13 |
| `b3_shift_lo` | 1.1057e-05 | 1.1737e-02 | 4.6929e-13 |
| `b3_scale_lo` | 1.2520e-05 | 1.3506e-02 | 5.3679e-13 |
| `b2_shift_mid` | 1.4139e-05 | 9.5744e-03 | 5.7439e-13 |
| `b3_scale_mid` | 1.4317e-05 | 8.4515e-03 | 6.2520e-13 |
| `b2_scale_mid` | 1.6805e-05 | 9.8874e-03 | 5.7452e-13 |
| `b2_scale_hi` | 1.8791e-05 | 6.2325e-03 | 6.1950e-13 |
| `b3_scale_hi` | 2.1060e-05 | 6.6696e-03 | 5.9953e-13 |
| `b3_shift_mid` | 2.6632e-05 | 1.4876e-02 | 8.3439e-13 |
| `b2_shift_hi` | 2.7134e-05 | 8.9468e-03 | 7.3604e-13 |
| `b3_shift_hi` | 2.9134e-05 | 8.6299e-03 | 7.3550e-13 |
| **`b1_scale_lo`** | **4.3438e-04** | 5.6263e-03 | 5.0806e-13 |
| `b1_shift_mid` | 5.7675e-04 | 3.2581e-03 | 1.1181e-12 |
| `b1_shift_lo` | 6.6678e-04 | 7.2295e-03 | 1.1178e-12 |
| `b1_scale_mid` | 1.3772e-03 | 7.0211e-03 | 1.3253e-12 |
| `b1_shift_hi` | 1.4875e-03 | 3.8444e-03 | 9.7376e-13 |
| `b1_scale_hi` | 1.9789e-03 | 2.6336e-03 | 9.3150e-13 |

**Three structures, all measured:**

1. **Depth dominates, and the split is clean.** The 12 quietest groups are ALL from blocks 2 and 3
   (1.03e-05 … 2.91e-05); the 6 loudest are ALL from block 1 (4.34e-04 … 1.98e-03). **No overlap**;
   the quietest block-1 group is **14.9x** the loudest block-2/3 group. This is the depth structure
   the per-tensor energy census predicts, realised.
2. **Efficiency runs AGAINST the magnitude pruner.** Mean `Δd_pose` per unit weight energy by norm
   tercile: **hi 6.16e-03 · mid 8.84e-03 · lo 1.010e-02**. The SMALLEST-norm rows are **1.64x more
   expensive per unit of energy removed** than the largest. Magnitude pruning is still directionally
   right on ABSOLUTE damage (small rows carry less energy), but it is buying the least efficient
   energy in the tensor.
3. **The FiLM scale/shift distinction barely matters:** 7.86e-03 vs 8.87e-03 per unit energy —
   **1.13x**. Whatever carries the pose here, it is not the multiplicative-vs-additive role.

**Selector alternatives at the identical 75-row cardinality** (archive-relative Δ bytes, i.e. after
the 232 B format toll):

| selection | Δ bytes | Δd_pose | vs mp2's choice |
|---|---:|---:|---:|
| **`sel_mp2_keep87_lowest_norm`** (the shipped magnitude pick) | **−130** | **5.6863e-04** | 1.00x |
| `sel_random_seed101` | −162 | 6.1981e-04 | 1.09x worse |
| `sel_random_seed404` | −48 | 7.1292e-04 | 1.25x worse |
| `sel_random_seed202` | −170 | 7.5286e-04 | 1.32x worse |
| `sel_highest_norm_anticontrol` (prune the LARGEST rows) | −153 | 1.4061e-03 | 2.47x worse |
| `sel_random_seed303` | −140 | 1.5450e-03 | 2.72x worse |
| `sel_random_seed505` | −179 | 1.6726e-03 | 2.94x worse |
| **`glob_lowest_norm_keep87count`** (rank globally, same 75 rows' worth of bytes) | **−125** | **2.5498e-05** | **22.30x BETTER** (17.99x at the second seed, §6b) |

Against a random-pick mean of 1.0606e-03 the shipped magnitude selector buys a **1.87x** damage
reduction and beats all five random draws, and the anti-control (prune the LARGEST rows) is 2.47x
worse than it — so the selector is doing real work. It is simply doing it at the wrong SCOPE: the
same byte spend, ranked across tensors instead of within them, is **22.30x** better again.

At the deeper prune count the same lever holds with a smaller factor, as the energy census
predicts: `glob_lowest_norm_keep25count` (432 rows, −1,901 B against keep25's −2,051 B) realises
`Δd_pose` = 6.6671e-04 on a 5.25x energy advantage — **295x over break-even**, versus the
incumbent-shaped extrapolation's 201x-at-2,051 B. The global pick buys less at depth because it
must eventually reach into block 1 anyway.

## §5 The dose-response is NOT monotone, and that is why this map was worth running

`#1058` closed the family citing a "monotone and hopeless" dose-response. Hopeless is right;
monotone is not, on `#1058`'s own table and on my re-priced one:

| candidate | rows pruned per tensor | `d_pose` [PyAV] | `d_pose` [authority] |
|---|---:|---:|---:|
| keep75∖keep87 marginal | 23 | 5.5551e-04 | not scored |
| keep87 | 25 | 6.8390e-04 | **5.286610e-04** |
| keep75 | 48 | 6.3959e-04 | **5.018133e-04** |

Pruning **23 more rows per tensor LOWERS** `d_pose` by 5.1% on the authority axis (6.5% on PyAV).
mp2 saw this and filed it as a live hypothesis; `#1058` overwrote it with "monotone". A closure
ground that is factually wrong should be corrected at source even when the verdict survives — and
it does survive, by three orders of magnitude.

## §6 Calibration — predicted vs realised (ra2crr NEXT_IF_RESUMED row 5, previously unowned; CLOSED here)

MAIN's relay is right that this vehicle has already killed one FD-fitted pose model (`pk4`: every
rung LOPO-positive in the modeled space, negative-or-zero in reality; `pk3`: 23/23 in-sample became
0/23 leave-one-out). So the first thing to state plainly:

**This map is not a fitted linearization. Every group is REALISED** — the rows are zeroed in the
weights, `frame_1` is re-rendered through the shipped renderer, and the frozen CPU PoseNet is run
on the real pair. No coefficient is fitted to anything, so there is no in-sample fit to collapse out
of sample. `pk4`'s failure mode does not reach this object.

What IS model-shaped here, and the realised check for each:

| model-shaped step | how it was REALISED | result |
|---|---|---|
| "my SM3R weight reconstruction IS mp2's candidate" | compared against `pack_prune_candidate`'s own `expected` map, on the real hv1 weights | **max abs diff 0.0** — exact |
| "my byte pricing IS the shipped archive's" | brotli-q11 of my keep87 packet vs the shipped `semantic.br` | **34,633 B == 34,633 B** — exact |
| "n = 120 stands in for n600" | `sel_mp2_keep87_lowest_norm` at n120 vs the SAME candidate re-priced at n600 (§2) | 5.6863e-04 vs 5.2178e-04 — **agree within 9.0%** |
| "zeroing rows in the base weights == decoding the SM3R candidate" | the q4 re-encode's weight-space delta energy is **2.650843e-07** against the prune's **4.419016e-02** — **0.0006%** of the perturbation | the two objects are the same to six parts in a million of energy; the 9.0% above is therefore subset sampling, not reconstruction error |
| "zeroing rows in the base weights == decoding the real SM3R candidate" | `sel_mp2_keep87_lowest_norm` (zero rows only) vs `attr_sm3r_q4_plus_keep87_rows` (full q4 re-encode + the same rows), same subset | 5.6863e-04 vs 5.6995e-04 — **agree within 0.23%** |
| "the q4 re-encode is free" | realised as `attr_sm3r_q4_reencode_only` | `Δd_pose` = **5.9191e-08** = **0.0104%** of the candidate's damage — free in practice as well as in energy |
| "a 24.95x energy advantage buys ~25x less pose" | **PRE-REGISTERED before measuring** (`run/pre_registered_prediction_global_selection.json`, predicted `Δd_pose` = 2.09e-05), realised as `glob_lowest_norm_keep87count` | realised **2.5498e-05** — **realised/predicted = 1.22**, inside the pre-registered 3x band. **ENERGY_LAW_HOLDS.** |
| "group damages add" — the one linearization-shaped claim in the unit | sum of the 18 disjoint groups (**6.7336e-03**) vs `attr_zero_all_film_rows` realised (**4.7270e-03**) | **realised/predicted = 0.702** — the parts OVER-predict the whole by **1.42x**. Damage is sub-additive. |

**The two calibration numbers a successor should carry, and their signs.**

1. **An additive group model OVERSTATES a large prune by ~1.42x.** That is the *opposite* sign to
   `pk4`/`jc1`, whose linearizations UNDERSTATED damage and thereby manufactured false positives.
   A model that overstates produces false NEGATIVES, so the honest reading is that this unit's
   partition-sum, used as a screen, is conservative — it would refuse things that are cheaper than
   it thinks, never admit things that are dearer.
2. **The energy law is a within-perturbation-TYPE law, not a universal one.** Across the 18 row
   groups (all row zeroings) `Δd_pose` per unit energy spans 5.65x. But the q4 re-encode —
   quantization noise spread over all 66,339 parameters instead of concentrated in 600 values —
   scores **0.223** per unit energy against ~1e-02 for the row groups, i.e. **~22x more damaging
   per unit of energy it applies.** Its absolute energy is so small that it does not matter here,
   but anyone porting "damage tracks energy" to a different actuator must re-measure the constant.
   This is exactly the cross-regime constant transfer the corpus keeps paying for.

### §6b Subset-sampling variability — MEASURED on a second seeded subset

The instrument's own repeat floor is exactly zero (deterministic render, deterministic forward,
fixed thread pin and batch shape), so the variability worth reporting is SUBSET sampling. Five
groups were re-measured on an independent seeded random subset, **seed 20260818**:

| group | seed 20260817 | seed 20260818 | ratio |
|---|---:|---:|---:|
| `null_zero_perturbation_control` | 0.0000e+00 | **0.0000e+00** | control passes at both seeds |
| `b2_scale_lo` | 1.0261e-05 | 9.1229e-06 | 0.889 |
| `sel_mp2_keep87_lowest_norm` | 5.6863e-04 | 5.7492e-04 | **1.011** |
| `glob_lowest_norm_keep87count` | 2.5498e-05 | 3.1959e-05 | 1.253 |
| `b1_scale_hi` | 1.9789e-03 | 2.5931e-03 | 1.310 |

**Individual absolutes move by −11% to +31%; the 75-row selections, which average over all three
tensors, move by 1.1% and 25%.** The one number this could threaten is the global re-ranking gain,
so it is reported as a range rather than a point: **17.99x (seed 20260818) to 22.30x (seed
20260817).** Nothing in this memo turns on a factor smaller than that band — the smallest
load-bearing factor is the 5.65x per-unit-energy spread, and the decisive ones are 184x, 3,942x
and 21.4x.

**What this variability does NOT threaten:** the byte findings (exact file measurements), the
authority re-pricing (n600), the frame_0 invariance (n600, exact), the format-exclusivity finding
(source), or the family verdict (a 184x miss cannot be closed by a 31% sampling swing).

## §7 Verdict

**1. The charter's literal question — the honest byte number: `0 B`.**
`verdict_scope: INSTANCE` on the banked mp2 candidate set. This is exact rather than inferential:
their archive bytes and their pose are both measured, and the −2,874 B composition is refused by
the shipped receiver at source.

**2. Is there a pose-quiet FiLM row subset? NO, at 32-row granularity.**
Perturbation energy spans 855.6x across the partition while damage per unit energy spans only
5.65x (2.96x per unit output energy), so energy — not row identity — is the controlling variable.
By my pre-registered bars both normalisations land **INDETERMINATE** (between 2x and 10x); I do not
move the bars, and I note that my own prediction (30x weight-energy, 1.6x output-energy) was wrong
in both directions.
`verdict_scope: FORMULATION` — 32-row groups of the three SM3R-eligible FiLM weights, hv1 ep0634,
n = 120 seeded random. **NOT closed:** single-row granularity, tensors outside `PRUNE_NAMES`, and
anything involving joint re-descent.

**3. The free lever the shipped selector misses — MEASURED, not predicted.**
`pack_prune_candidate` ranks within each tensor; `blocks.1.film.weight` carries 152x the row energy
of blocks 2 and 3; a global ranking at the same cardinality picks 0 rows from block 1 and realises
a **22.30x** lower `Δd_pose` (5.6863e-04 → 2.5498e-05) for 5 B less credit (−125 vs −130 B);
**17.99x on an independent second subset, so the load-bearing band is 18.0-22.3x** (§6b). That
moves the best cell in the unit from **3,942x to 184x** over break-even.
`verdict_scope: INSTANCE` on the two mp2 prune counts. **This improves the selector and does not
rescue the family.**

**4. `#1058`'s FAMILY closure SURVIVES**, on a corrected axis (1.45x stronger) and a corrected
mechanism (non-monotone; energy-tracking, not dose-tracking). This unit replaces three magnitude
samples with a partition-wide measurement over all 576 rows, plus a byte-matched re-selection that
extracts every factor the energy structure has to give. Do not reopen without joint re-descent.

**What would reopen it:** a realised `Δd_pose ≤ 2.4530e-06` at the −2,051 B credit. The best
realised cell in this unit is 2.5498e-05 at −125 B, which is **184x** short; at the deeper count
`glob_lowest_norm_keep25count` realises 6.6671e-04 at −1,901 B, **295x** short. Re-selection was
the only free structural lever available and it has now been spent. Nothing inside this actuator
class closes the remaining two orders of magnitude.

## §8 Retained payloads (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_sf1_semantic_film_pose_map_20260817/`. VertigoDataTier holds
893 MiB and is read-only — read from, never written to.

| artifact | what it is |
|---|---|
| `SF1_CONTROLS.json` | every control in §3, with each input's sha and byte count |
| `SF1_BYTES.json` | the nine-generation byte census, the format-exclusivity finding, the break-even table |
| `SF1_REPRICE.json` | the authority re-pricing of mp2's three n600 rows |
| `SF1_MAP.json` / `SF1_VERDICT.json` | the map summary and the adjudicated branch |
| `retained/sf1_base_pose6.float64.npy` | (600, 6) n600 base pose, independent forwards |
| `retained/sf1_authority_gt_pose6.float64.npy` | (600, 6) authority GT pose |
| `retained/sf1_film_row_norms.float64.npy` | (3, 192) the row-norm census the selector ranks on |
| `retained/sf1_pose6_<candidate>.float64.npy` | (600, 6) generated pose for each re-priced mp2 archive |
| `groups/<id>/pose6.float64.npy` | (120, 6) per-group generated pose |
| `groups/<id>/output_energy.float64.npy` | (120,) per-pair frame_1 perturbation energy |
| `groups/<id>/perturbed_film.float32.npy` | (3, 192, 8) the perturbed field itself |
| `groups/<id>/frame1_stack.u8` | the 120 rendered camera frames the pose was measured on |
| `run/pre_registered_prediction.json` · `run/pre_registered_prediction_global_selection.json` | both predictions, written before their measurements |
| `run/{controls,reprice,map,chain,seed2}/` | launch manifests (git HEAD, argv, env, thread pin) and full logs |

**84 group files, 5.80 GiB of frame stacks and per-pair arrays retained**, plus every scalar
receipt. Every JSON records the sha256 and byte count of the payload it describes. No measurement
in this unit produced a length without keeping the bytes.

## §9 What I did NOT establish

- **No exact-eval row, no contest score.** Everything is `[macOS-CPU advisory]` against an
  authority-TRACKING GT (`pi2`: 1.00081x), not a contest measurement. $0 spent, no Modal.
- **No `d_seg` leg for the map groups.** mp2 measured `Δd_seg` at +6.4e-07 … +1.14e-06 for its
  three candidates (ΔS_seg ~ +1e-04), two orders below the pose leg, so it cannot change a sign
  here. It is not measured for my groups and I claim nothing about it.
- **No per-ROW map.** The partition is 32-row groups. A pose-quiet subset finer than 32 rows is NOT
  excluded by this measurement.
- **No n600 for the map groups.** They are n = 120, seeded random, **seed 20260817**, never a
  prefix ([[m96]]: pose prefixes read 2.54–4.21x harder and would fake a NO-GO). The n600 anchors
  are the three re-priced mp2 archives in §2, and the n120↔n600 agreement is 9.0% (§6).
- **No claim about joint re-descent.** Everything here perturbs finished weights. Nothing measured
  here transfers to a renderer trained with the pose term in the loop.
- **ra2crr row 4 (the 22,032 B pool drift) is untouched.** My object is the 34,763 B semantic
  section, not the 22,161 B carrier, so I never cite the pool figure and did not re-derive it. That
  row stays open for whoever prices against the carrier next.
- **The second-seed repeat covers 5 of the 31 groups, not all of them** (§6b). The other 26 have a
  single subset each; their individual absolutes carry an unmeasured error of the same order
  (−11% to +31% is the observed range).
- **No shippable global-re-ranked candidate.** The 22.30x lever is measured on the weights; a real
  archive would need the SM3R header to carry a per-tensor keep count (it carries one global
  `keep_percent`), which is a receiver change this unit did not make.

**My pre-registered predictions, and how they scored.** Recorded in
`run/pre_registered_prediction.json` before any group ran: weight-energy spread **30x**
(STRUCTURED), output-energy spread **1.6x** (ENERGY_LIKE). Measured: **5.65x** and **2.96x** — both
INDETERMINATE, so I was wrong in both directions and the bucket call was wrong too. I also
predicted, before re-deriving, that the −2,874 B would prove a forbidden sum — that one was right,
and understated: it is not merely forbidden, it is undecodable. And I expected the authority-axis
re-pricing to WEAKEN the refusal; it strengthened it 1.45x.

## NEXT_IF_RESUMED

| # | row | owner | fire-condition |
|---|---|---|---|
| 1 | **Nothing further on post-hoc semantic weight edits.** `#1058`'s FAMILY closure survives on a corrected axis and a corrected mechanism; this unit adds a partition-wide measurement over all 576 rows. | — | do not reopen without joint re-descent |
| 2 | **Second-seed repeat: DONE and folded in** (§6b). Sampling error on individual absolutes is −11% to +31%; the global re-ranking gain reads 18.0-22.3x across the two subsets. No verdict moved. | — | closed |
| 3 | **Fix the ranking SCOPE in `pack_prune_candidate`.** It ranks rows within each tensor; ranking globally is byte-neutral (−125 vs −130 B) and MEASURED 22.30x cheaper in pose. `global_lowest_norm_selection` is written and tested in `experiments/ddm_sf1_semantic_film_pose_map.py`; shipping it also needs a per-tensor keep count in the SM3R header, which today carries one global `keep_percent`. It does not rescue this family, but any future consumer of that packer inherits the defect. | unowned | any arm that reuses SM3R row pruning |
| 4 | **Re-price the remaining PyAV-axis rows.** mp2's three are done here. `ddm_b2e_edit_replay_admission_verdict_20260816` and every memo quoting base `d_pose` 1.4747e-04 sits on the 21.4x-inflated lineage; where `pose6` was retained the correction is pure arithmetic, and it makes refusals STRONGER, not weaker. | MAIN | $0 wherever pose6 is retained |
| 5 | **The 0.00078%-energy / 76.8x-`d_pose` ratio is the transferable number for the joint line.** It says which weights a joint objective must protect: block-1 FiLM carries 152x the row energy of blocks 2-3 and every one of the six loudest groups. The 18-group map plus the per-tensor energy census is the retained asset. | joint line (#982 trained receiver · js8) | when the joint line needs a freeze/allow mask |
| 6 | ra2crr row 4 (the 22,032 B carrier pool drift) remains open — this unit never priced against the carrier. | unowned | before the pool is priced again |

**Own-vehicle frontier: hv1 ep0634, S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4 n600]` —
UNMOVED by this unit.**
