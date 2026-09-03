# ddm_ar1 — the $0 price of the footprint (AA) render on the BORN field

Arm `ddm_ar1` · charter `.omx/research/charters/ddm_ar1_aa_render_price_on_born_field_20260903.md` (785522afa)
Axis **`[macOS-CPU advisory; frozen scorer; not contest authority]`** · `score_claim=false` · `promotion_eligible=false`
Cost **$0** (local CPU, 4 threads, `nice 10`). Instrument `experiments/ddm_ar1_aa_render_price.py`
(c9bf822fb → cc0b010e8 → 33471e60b), tests `tests/test_ddm_ar1_aa_render_price.py` (49 passing).

---

## HEADLINE

**The pre-registered falsifier FIRED on all three clauses, and the sign is reversed.** On the sealed QBR1
control checkpoint at step 5000, the footprint (AA) render at ss=2 does not lower the born field's d_seg — it
RAISES it by **1.314×** (0.002857049 → 0.003753185, DALI authority, n600, **32/32 trained pairs worse, not one
exception**), and raises d_pose by **+1.563e-3**, which is **156× the falsifier's +1e-5 bound**. The composite
cost is **ΔS +0.161213** on the burn's own Horvitz–Thompson estimator = **+242,113 equivalent bytes at
6.658589531221714e-7 S/B** — more than twice the entire 106,643-byte archive. ss=3 is worse still
(ratio 0.6614, ΔS_HT +0.261938). The lever is not merely absent on this vehicle; it is strongly negative.

MEASURED, not inferred: the damage is footprint AVERAGING, not the 0.25-px lattice misregistration this arm
also found and repaired. Re-rendering all 600 pairs on the registration-corrected lattice recovers **1.6%** of
the d_seg damage and makes ΔS **worse** (+0.172670 HT); the broken sites are 84.6% mid-frame and **0.000** in
both outermost row bands, the exact anti-shape of the drift profile (Pearson(drift, broken) = **−0.834**).

---

## 1. The pre-registered prediction, read out clause by clause

Charter, verbatim:

> Prediction: ss=2 box render lowers born d_seg by **≥1.5×** on a seeded random n32 with d_pose change ≤ +1e-5.
> **Falsifier:** ratio < 1.10 OR d_pose rises > 1e-5 OR the B/H/W split shows the gain is all in one class with
> harm elsewhere.

| Clause | Bound | MEASURED (trained n32, DALI, ss=2) | Verdict |
|---|---|---|---|
| d_seg ratio (base/AA) | predict ≥ 1.50; falsify < 1.10 | **0.7612** (median over pairs 0.7707) | **FIRES** |
| d_pose rise | falsify > +1e-5 | **+1.563040188392085e-3** (156× the bound) | **FIRES** |
| B/H/W concentration | falsify if gain is one class with harm elsewhere | gain is **Undrivable alone** (+164,360); every other class is harmed (−989,640 combined) | **FIRES**, mirrored |

The third clause fires in mirror image: the charter anticipated a one-class *gain* with collateral harm; what
happened is a one-class gain with harm everywhere else AND a net loss overall. Same shape, opposite sign.

### The prediction's premise had already expired (MEASURED, re-derived before reading the falsifier)

The charter's argument — "the born field (d_seg ≈ 0.0130) sits **2.37× above** the point-sampled achievable
bound 0.0054940456814236115, so the mechanism is in range" — was true of the vr1-era read. It is not true of
the checkpoint this arm measured. The QBR1 control cell's own milestone ladder (`RESULT.json`, its axis
`[macOS-MPS n32 stratified advisory]`):

| step | d_seg_hat | d_pose_hat | S_hat | archive_bytes_exact |
|---:|---|---|---|---:|
| 0 | 0.002518335978190104 | 0.0005757456120606528 | 0.39876797285867277 | 106,714 |
| 1000 | 0.003051122029622396 | 0.0008233354187810106 | 0.46687521208987615 | 106,667 |
| 2000 | 0.0032170613606770835 | 0.000864393511532432 | 0.48567677825279465 | 106,626 |
| 3000 | 0.003139241536458333 | 0.0008181846911522883 | 0.47538291701253005 | 106,637 |
| 4000 | 0.0029336293538411457 | 0.0006051119375803525 | 0.4421903707337701 | 106,687 |
| 5000 | 0.002758916219075521 | 0.0006122744215585018 | 0.42514878445269977 | 106,643 |

At step 5000 the born field sits at d_seg 0.002758916 — **1.99× BELOW** the 0.005494 point-sampled bound, not
2.37× above it. The "in range" argument rested on a number the burn had already moved past by 4.7×. This is
`[[binding-instruction-numbers-expire-and-nobody-rederives-them]]` at charter scope. I re-derived it before
reading the falsifier out; the falsifier fires anyway, but on different grounds than the charter anticipated.

Note also, as a standing fact for the burn (not this arm's business to adjudicate): the control cell's d_seg
at step 5000 is **worse** than at step 0 (0.002759 vs 0.002518).

---

## 2. CALIBRATION GATE — **PASS**, decomposed into three legs

The charter's gate: the ss=1 read must reproduce the checkpoint's own recorded d_seg/d_pose within 2%. The
burn's milestone is an **MPS** number and this arm is CPU-only, so a single pass/fail would conflate
arithmetic, scorer axis and render axis. Three legs, all on the burn's own PyAV target arrays and its own
Horvitz–Thompson weights, against recorded `d_seg_hat 0.002758916219075521` / `d_pose_hat 0.0006122744215585018`:

| leg | what it isolates | d_seg_hat | gap | d_pose_hat | gap |
|---|---|---|---:|---|---:|
| 1 | arithmetic on the burn's own retained arrays | 0.002758916219075521 | **+0.0000%** | 0.0006122744215585018 | **+0.0000%** |
| 2 | CPU scorers on the burn's own retained camera uint8 bytes | 0.0027587890625 | −0.0046% | 0.0006122594454092925 | −0.0024% |
| 3 | CPU render at ss=1 → CPU roundtrip → CPU scorers | 0.0027605692545572915 | **+0.0599%** | 0.0006131150456335778 | **+0.1373%** |

**Verdict PASS** at 33× and 15× inside the 2% bar. Leg 1 exact proves the per-pair arithmetic is the burn's own
(`_retain_eval_outputs`). Legs 2 and 3 together bound the MPS↔CPU axis gap **on this vehicle at this operating
point** at 0.06% (d_seg) / 0.14% (d_pose) — far below the historical 2×/23× drift CLAUDE.md warns of. That is a
useful bound for the burn's readers; it is **not** a licence to treat MPS as authority, and this arm does not.
Artifact: `CALIBRATION.json`. It also confirms the `stage_01_end.pt` EMA shadow is the milestone's weights.

---

## 3. Scope correction: the burn trains 32 pairs, not 600 (VERIFIED AT SOURCE)

The sealed checkpoint's `config_identity.pair_ids` **equals `qbt.SELECTION_IDS`** — the burn optimises its 32
stratified pairs and no others. QBF1's `boundary_latents`/`interior_latents` are per-pair `(600, ·)`
parameters, so the other 568 pairs received **no d_seg/d_pose gradient at all**; their latents moved only by
AdamW's decoupled weight decay (`torch.optim.AdamW(model.parameters(), lr=2e-4)`, default `weight_decay=0.01`
→ ≈1% shrink over 5000 steps; MEASURED max latent delta 0.0103/0.0077 untrained vs 0.0201/0.0285 trained).

Consequence for this arm's charter: the charter specified a **seeded random n32**, of which **31 of 32 pairs
are outside the burn's selection** (only pair 173 overlaps). That sample measures the inherited-latent
population, not the born field. I therefore ran the **full n600** — a scope INCREASE, per the n600-scale
non-negotiable — and report the two populations separately. They are different objects and are never averaged:

| population | ss=1 d_seg (DALI) | ss=1 d_pose (DALI) | what it is |
|---|---|---|---|
| burn selection n32 | 0.002857049 | 7.389459e-04 | **the born field** — the only trained object |
| untrained n568 | 0.062399573 | 4.515785e-01 | inherited latents + shared params; 21.8× worse d_seg |
| charter seeded n32 | 0.058523019 | 3.894590e-01 | 31/32 drawn from the untrained population |

---

## 4. A registration defect in the AA operator itself (MEASURED, repaired, then falsified as the mechanism)

`aa_sdf_observation_render.build_supersampled_coords(h, w, ss)` is `build_render_coords(h*ss, w*ss)` =
`linspace(-1, 1, n*ss)` — endpoint-inclusive, sharing the coarse grid's endpoints. Its module docstring claims
this "integrates each coarse pixel's footprint". **It does not, exactly.** MEASURED: the ss-block means of
`linspace(-1,1,n*ss)` drift from `linspace(-1,1,n)` by

| ss | max drift (coarse pixels) | at | at frame centre |
|---:|---:|---|---:|
| 2 | **0.2497** | both frame edges, inward | ~0 (6.5e-4) |
| 3 | **0.3328** | both frame edges, inward | ~0 |
| 4 | **0.3743** | both frame edges, inward | ~0 |

so the module's AA image is a slightly **contracted** copy of the field (converging to 0.5 px as ss→∞). Any AA
delta measured on that lattice therefore mixes footprint averaging with a sub-pixel registration shift.

I implemented the corrected lattice (`--lattice footprint_centred`): with coarse pitch `p = 2/(n-1)` the
sub-cell centres are `-1 + i·p + p·(k + 0.5 - ss/2)/ss`, which flattens to a uniform grid over
`[-1 - p/2 + p/(2ss), 1 + p/2 - p/(2ss)]` whose ss-blocks equal the coarse samples **exactly** (tested to 1e-12
at n ∈ {8, 384, 512}, ss ∈ {1..5}). The field is evaluated through the trainer's own `_base_features` with only
the two grid-laying `linspace` calls re-spanned.

**But registration is NOT the mechanism.** The retained argmax already decides it, at zero extra cost. Row
octile profiles (edge→edge, uniform = 0.125), ss=2:

| profile | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DRIFT (what registration predicts) | 0.219 | 0.156 | 0.094 | 0.031 | 0.031 | 0.094 | 0.156 | 0.219 |
| BROKEN, trained n32 (MEASURED) | **0.000** | **0.000** | **0.000** | 0.407 | 0.439 | 0.129 | 0.025 | **0.000** |
| FIXED, trained n32 | 0.000 | 0.000 | 0.000 | 0.344 | 0.472 | 0.166 | 0.018 | 0.000 |
| BROKEN, all 600 | 0.003 | 0.001 | 0.002 | 0.075 | 0.337 | 0.372 | 0.092 | 0.119 |

**Pearson(drift, broken) = −0.8342** on the trained pairs (−0.4623 over all 600). Outer-two-band share: drift
0.437, broken **0.000**. Mid-two-band share: drift 0.063, broken **0.846**. The damage lives exactly where the
drift is zero and vanishes exactly where the drift is maximal. The registration error lands on flat regions
(sky above, ego-hood below) where a quarter-pixel shift changes no argmax; the blur lands on the mid-frame
horizon band where the boundaries actually are. **The AA damage is footprint averaging.**

### The corrected lattice, measured directly (n600 ss=2) — inference replaced by measurement

I did not stop at the profile argument. I re-rendered all 600 pairs at ss=2 on the registration-corrected
lattice and re-scored them:

| lattice | trained n32 d_seg (mean) | ratio (base/AA) | pairs worse | Δd_pose | **ΔS (HT)** |
|---|---|---:|---:|---:|---:|
| `module_endpoint` | 0.003753185 | 0.7612 | 32 / 32 | +1.563040e-03 | **+0.161213** |
| `footprint_centred` | 0.003692468 | 0.7738 | 31 / 32 | +1.963750e-03 | **+0.172670** |

Removing the misregistration entirely recovers **1.6% of the d_seg damage** (0.003753185 → 0.003692468 against
a 0.002857049 baseline) and makes **ΔS worse, not better** (+0.172670 vs +0.161213), because d_pose degrades
further. 31 of 32 pairs are still worse. The B/H/W totals barely move (net −813,246 vs −825,280; Lane −8,201 vs
−8,578; MyCar −296,112 vs −278,999). The damage profile is unchanged — broken octiles
`0.000 0.000 0.000 0.440 0.434 0.116 0.011 0.000`, Pearson(drift, broken) = **−0.8292**.

**Registration accounts for at most 1.6% of the d_seg cost and none of the ΔS cost. The AA penalty on the born
field is footprint averaging.** The registration defect is real and worth fixing in the module regardless — it
is simply not what is costing S here.

---

## 5. The measurement (n600, module lattice, DALI authority)

Per-pair receipts: `per_pair_rows.jsonl`, 1800 rows (600 pairs × ss ∈ {1,2,3}); every rendered argmax retained.

### The born field — burn selection n32 (the load-bearing block)

| ss | render grid | d_seg (mean) | d_seg (HT) | d_pose (mean) | d_pose (HT) | S_hat (HT) |
|---:|---|---|---|---|---|---|
| 1 | 384×512 | 0.002857049 | **0.002806091** | 7.389459e-04 | **6.781343e-04** | **0.433967** |
| 2 | 768×1024 → box 2 | 0.003753185 | 0.003722254 | 2.301986e-03 | 2.308757e-03 | 0.595181 |
| 3 | 1152×1536 → box 3 | 0.004319668 | 0.004338074 | 3.645625e-03 | 3.651498e-03 | 0.695906 |

| comparison | d_seg ratio (base/AA) | median over pairs | pairs AA-worse | ΔS (mean) | ΔS (HT) |
|---|---:|---:|---:|---:|---:|
| ss1 → ss2 | **0.7612** | 0.7707 | **32 / 32** | +0.155375 | **+0.161213** |
| ss1 → ss3 | **0.6614** | 0.6844 | **32 / 32** | +0.251235 | **+0.261938** |

Unanimity matters: 32/32 and 32/32. This is not a mean dragged by outliers.

### The untrained population, for contrast (n568)

| comparison | d_seg ratio | median over pairs | pairs AA-worse | ΔS (mean) |
|---|---:|---:|---:|---:|
| ss1 → ss2 | 0.8948 | 0.9975 | 292 / 568 | +0.881317 |
| ss1 → ss3 | 0.8463 | 0.9873 | 319 / 568 | +1.326612 |

The untrained field is hurt **less** (ratio 0.895 vs 0.761) and only about half its pairs move at all (median
ratio 0.9975 ≈ no change). **The better-fitted the field, the more AA costs it.** That is the mechanism stated
as a dose-response: training sharpened the render at the sample points the trainer uses, and footprint
averaging destroys exactly that sharpening. A field that was never fitted has less to lose.

The charter's seeded n32 (31/32 untrained) reads ratio 0.9113 with 15/32 worse — it would have produced a much
weaker, and population-wrong, verdict.

### GT lineage

Both frames read through `assert_gt_lineage`. DALI (`gt_cache_dali.pt`, sha `a91d98252fe377c5…`, 117,980,732 B)
is the authority; `gt_n600.npz` (PyAV, sha `cf8d83605d2198ef…`) is reported only as the burn's continuity frame.
**MEASURED 20,671 argmax sites differ** between them — matching the charter's stated figure exactly. On the
trained n32 the lineage costs +1.7% on d_seg (DALI HT 0.002806091 vs the burn's PyAV 0.002758916). Every number
in §5 is DALI; the calibration in §2 is deliberately PyAV because it reproduces the burn's own read.

---

## 6. B/H/W split per class (ss=2, all 600 pairs, DALI, attributed by target class)

| class | target sites | base wrong | AA wrong | fixed | broken | **net** |
|---|---:|---:|---:|---:|---:|---:|
| Road | 27,407,371 | 3,795,882 | 4,472,473 | 264,196 | 940,787 | **−676,591** |
| Lane | 690,753 | 600,534 | 609,112 | 4,290 | 12,868 | **−8,578** |
| Undrivable | 58,413,069 | 841,693 | 677,333 | 202,424 | 38,064 | **+164,360** |
| Movable | 1,460,386 | 1,216,458 | 1,241,930 | 10,751 | 36,223 | **−25,472** |
| MyCar | 29,993,221 | 531,777 | 810,776 | 12,433 | 291,432 | **−278,999** |
| **TOTAL** | | | | **494,094** | **1,319,374** | **−825,280** |

At ss=3 the same shape deepens: fixed 572,169, broken 1,846,894, net −1,274,725; Undrivable +209,690 against
−1,484,415 everywhere else.

**AA breaks 2.67 sites for every one it fixes.** The single gaining class is Undrivable (sky/background, 49.5%
of area, the easiest class). **Lane — the class the published law's headline claims footprint integration
recovers ("recovers finest-scale lanes that point-sampling erases") — goes the wrong way: net −8,578, with
broken 3.0× fixed.** On the born field, AA does not recover lane structure; it erodes it.

---

## 7. ΔS and the exchange rate

AA is a decode-time render change on an **unchanged archive**, so the rate term does not move and ΔS is pure
distortion: `ΔS = 100·Δd_seg + [√(10·d_pose_AA) − √(10·d_pose_base)]`.

Exchange rate: **25 / 37,545,489 = 6.658589531221714e-7 S/B** (verified to 1e-22; it is exactly the rate-term
slope, and a test pins it).

At the burn's archive `B = 106,643` (rate 0.07100919633780772; `B_hat` 121,860 is its HT projection — ΔS is
byte-independent, so the choice affects only the absolute S level):

| population | comparison | ΔS | equivalent bytes |
|---|---|---:|---:|
| **born field (HT, n32)** | ss1 → ss2 | **+0.161213** | **+242,113 B** |
| **born field (HT, n32)** | ss1 → ss3 | **+0.261938** | **+393,384 B** |
| born field (mean, n32) | ss1 → ss2 | +0.155375 | +233,345 B |
| untrained n568 | ss1 → ss2 | +0.881317 | +1,323,579 B |

**+242,113 equivalent bytes is 2.27× the entire 106,643-byte archive**, and 1.75× the 137,986-byte
complete-archive cap. Adopting the module's AA render on this field would cost more S than deleting the
archive twice over. There is no byte budget at which this trade is worth making.

Against the live sub-0.12 demand (−42,016 B at held distortion): AA moves the wrong way by **5.76×** the entire
remaining byte demand.

---

## 8. Wall seconds per pair per mode (n600, 4 CPU threads, `nice 10`, uncontended)

| ss | render s/pair | total s/pair | multiple of ss=1 | theoretical (ss²) |
|---:|---:|---:|---:|---:|
| 1 | 0.244 | **1.04** | 1.00× | 1× |
| 2 | 1.036 | **1.85** | 1.78× | 4× |
| 3 | 2.167 | **2.97** | 2.86× | 9× |

The render itself scales close to ss² (0.244 → 1.036 → 2.167 is 4.2× and 8.9×); the fixed scorer forward
(~0.70 s/pair) dilutes it to 1.78×/2.86× end-to-end. Peak RSS 9.0 GiB at ss=3. Whole n600 × 3 modes: 3536 s.
The cost is affordable — the lever is simply negative, so the cost is beside the point.

---

## verdict_scope

**FORMULATION.** Scoped to: the POST-HOC render swap (footprint integration applied at eval time to a field
whose training loop point-sampled), on the QBR1 control `stage_01_end.pt` (step 5000, sha
`f5a152c37a5a0d27…`), the module-endpoint lattice at ss ∈ {2, 3} and the registration-corrected lattice at
ss = 2, frozen CPU-torch scorers, DALI authority, n600 with the trained and untrained populations reported
separately and never averaged together.

**What this does NOT close.** It does not kill the AA family. Specifically it does not test — and cannot speak
to — **AA in the training loop**: a field trained with the footprint render in `forward` optimises against a
different observation operator and has no reason to inherit this result. That is a train/test mismatch verdict,
not a mechanism verdict, and it lands squarely on the campaign's standing pattern that post-hoc swaps die where
joint descent crosses.

**What it does close.** The published law's stated inequality `d_seg(AA) ≤ d_seg(point)` is **false on a
point-trained learned field**, by 1.314× at ss=2 with 32/32 unanimity.

---

## GESTALT-DELTA

The registered law `aa_sdf_observation_footprint_render_dseg_v1` carries an honest anchor —
`inputs.signal = "real_frame_achievable_through_R (confound-free upper bound)"` — but its
`domain_of_validity` block names only `{R, aa_mode, n_pairs_range, render_grid, scorer}`. **The one dimension
that decides the sign is missing from the domain: what is being rendered.** A reader consulting the domain, or
the `latex_form` inequality `d_seg(AA) ≤ d_seg(point)`, trips no guard while transferring a real-frame result
onto a learned field, which is exactly what the charter did. This is the split-scope failure written at
registry level: the anchor is scoped, the domain is not, and the domain is what routing reads.

The finding itself sharpens the campaign's cross: **an observation operator is part of the object, not a free
post-hoc choice.** The born field is not an approximation of the video that happens to be sampled at 384×512;
it is a solution optimised *for* that sampling. Its d_seg 0.002806 already beats the 0.005494 real-frame
point-sampled "achievable bound" by 1.96× — a learned field can beat a bound derived from real frames because
it was never trying to be the real frames. The same reasoning that makes it beat the bound makes it fragile to
changing the operator underneath it. Dose-response confirms it: the trained field loses 1.314×, the unfitted
field only 1.118×.

Two smaller deltas fold in. (a) On this vehicle at this operating point the MPS↔CPU axis gap is 0.06% d_seg /
0.14% d_pose — the burn's MPS milestones are near-CPU-faithful, which does not promote MPS but does bound how
much the burn's own readings can be doubted. (b) `build_supersampled_coords` has a real 0.25-px registration
defect that its docstring denies; here it is harmless because it lands on flat regions, but any future arm
using that helper near a frame edge inherits it.

---

## NEXT_IF_RESUMED

1. **The only live version of this lever is AA IN THE TRAINING LOOP.** Change `QBFLOWTorch.forward` to render
   at `ss*grid` and box-downsample before `roundtrip_to_camera_uint8_ste`, train from the same warm start, and
   compare byte-closed against the point-sampled control. Cost is measured: ss=2 costs 1.78× wall per pair
   end-to-end (render alone 4.2×), so a matched 5000-step cell is ≈1.78× the control's 10,604 s ≈ 5.2 h on the
   same hardware. **Pre-register the falsifier**: AA-trained d_seg must beat the point-trained control at equal
   steps and equal bytes, or the family closes at FORMULATION+1.
2. **Narrow the registry domain before anything else routes on it** (owed to MAIN; this arm is `[no-triality]`
   and did not write it): add a `signal` key to `aa_sdf_observation_footprint_render_dseg_v1`'s
   `domain_of_validity` with `real_frame_achievable` IN-DOMAIN and `point_trained_learned_field` EXCLUDED,
   citing this memo's n600 reversal. The `latex_form` inequality needs the same qualifier.
3. **Fix `build_supersampled_coords`** to the footprint-centred span (the derivation and a 1e-12 test are in
   `experiments/ddm_ar1_aa_render_price.py::footprint_centred_span`), or document the 0.25-px contraction in
   the module docstring. Any consumer sampling near a frame edge is currently misregistered.
4. **Do not re-run this arm's question on the charter's seeded n32.** 31 of its 32 pairs are outside the burn's
   trained selection; the correct born-field population is `qbt.SELECTION_IDS`, and the full n600 (banked here)
   subsumes both.
5. **Arm-id collision to resolve:** `ddm_ar1` is already taken by `experiments/ddm_ar1_pose_target_structure_probe.py`
   (34694fcd5, "archetype codec PRICED SPEC"). Filenames do not collide; the ledger id does.

---

## Custody

Store `/Volumes/APDataStore/pact/ddm_ar1_aa_render_price/` (ALWAYS KEEP THE PAYLOAD — every rendered argmax
retained, camera uint8 for 4 pairs per mode, every per-pair row fsync'd as produced so a crash loses nothing):

| artifact | what |
|---|---|
| `per_pair_rows.jsonl` | **2,400** per-pair receipts: d_seg/d_pose on both GT frames, pose6, wall render/score seconds, argmax sha256. The 1,800 module rows predate the `lattice` field and carry no such key; readers treat a missing key as `module_endpoint`, which is what they are by construction. The 600 centred rows carry `lattice: footprint_centred` explicitly. |
| `argmax/pair_XXXX_ssN[_centred].npy` | **2,400** rendered SegNet argmax arrays (600 × 3 module + 600 centred) — every render this arm produced |
| `camera/pair_XXXX_ssN[_centred].npy` | 16 camera uint8 pairs (4 pairs × 3 module modes + 4 × centred) |
| `CALIBRATION.json` | the three-leg gate |
| `AGGREGATE.json` / `AGGREGATE_centred.json` | subsets, deltas, B/H/W, spatial profiles |
| `MANIFEST.json` / `MANIFEST_centred.json` | git head, platform, torch version, checkpoint + source file facts, GT lineage. The module manifest predates the `lattice` field and records it as `null`; its run is `module_endpoint`. Module run 1,800 rows / 3,536 s / 9.0 GiB peak; centred run 600 rows / 1,228 s / 4.9 GiB peak. |
| `measure_run/`, `measure_centred_run/`, `calibration_run/` | governed-launch manifests and logs (`tools/launch_detached_process.py`, `--nice 10 --nice-best-effort`, derived resource budgets) |

Total retained payload **1.0 GB**. Instrument commits: c9bf822fb (instrument + 3-leg gate), cc0b010e8
(lattice defect + repair), 33471e60b (spatial discriminator). 49 tests pass; ruff clean; the instrument file
carries two review-tracker passes per landing.

Inputs (read-only): checkpoint
`/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/seed_20260902/control_native100/stage_01_fairform_finish/checkpoints/stage_01_end.pt`
(1,607,325 B, sha `f5a152c37a5a0d27c2ad4ae6af1c59b0d8237f4735a4a29cf7c5d0ff4355ca91`, `completed_steps` 5000,
EMA decay 0.9990793899844618 / 5000 updates); milestone
`…/control_native100/milestones/step_005000`. `experiments/ddm_qbt1_qbflow_trainer.py` sha16
`6eda9c202b3aee00`; `src/tac/boundary_math/aa_sdf_observation_render.py` sha16 `a9842371c483c617` — both match
the charter's pins.

Nothing was written under the burn's `runs/`. The live treatment cell was untouched and stayed healthy
throughout. No Metal, no MPS, no Modal, no contest evaluation, no `upstream/` edit, no
`submissions/semantic_joint_ctxmix/` edit. No `/tmp` path appears in any retained artifact.

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `aa_sdf_observation_footprint_render_dseg_v1` — `tac.canonical_equations (registered via tools/register_aa_sdf_observation_render_equation.py)` (`tac.canonical_equations`). **Relation:** REFINES — the refinement this memo asked for was landed FOR it at commit d3212bed1.

AR1 §332 named the change and said plainly it did not write it. The law's `domain_of_validity` now carries it: the `signal` dimension, with `real_frame_achievable_through_R` IN-DOMAIN and `point_trained_learned_field` EXCLUDED, `last_domain_refinement_rationale` naming this memo. This line closes the loop by putting the reference token in the memo; it asserts nothing new.
