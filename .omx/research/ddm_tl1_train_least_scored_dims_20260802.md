# ddm_tl1 — train-least: the scorer is already near-minimal; the oversized object is the actuator

**UTC** 2026-08-03 · **arm** `ddm_tl1_train_least_scored_dims` · **axis** `[macOS-CPU advisory]` /
`$0 static-weight + cached-artifact reductions`. No scorer forward or backward was fired; the n600
scorer slot stayed held. `score_claim=false`, `promotion_eligible=false`. **Pointer UNMOVED.**

**Operator directive under test (2026-08-02, verbatim):** *"we want to train the least amount possible
and that within SegNet and PoseNet, there are layers and dimensions we can break it down to and only
train against those which are absolutely necessary."*

**Denominator for every ΔS:** `tac.canonical_equations.gap_decomposition_against_floor_20260802` —
gap to the PR130 demonstrated floor **0.7262358** (seg 0.4015 · pose 0.2120 · rate 0.1127); 1% of gap
= **10,907 B**; PR130 floor archive **191,052 B**. Live base `dc1_fold` **S = 0.8983775** @ 360,309 B.

**Frozen custody re-hashed for this memo:** `upstream/models/segnet.safetensors` sha256
`68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` (matches SPEC_v10 [E17]);
`upstream/models/posenet.safetensors` sha256
`0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`.

---

## The answer first

1. **The directive is CORRECT for pose and FALSE for seg, and both halves are now measured.**
   Pose really is a six-dimensional problem (`rank(J) ≤ 6`, `ddm_pb3`). Seg is **not** a
   low-dimensional problem: the famous rank-4 head quotient is a per-*site* output gauge that
   **does not propagate backward one layer**. RE-DERIVED here, $0, from the frozen weights: a single
   16-channel decoder feature location is read by 9 output sites through 9 different 3×3 taps; the
   stacked centered map is **rank 16 of 16, condition number 10.48**. **100% of the penultimate
   channel space is scored, and well-conditioned.** §2.

2. **We already train close to minimally on the scorer graph, and the discard is architecturally
   non-recoverable.** The scored-active seg set is **0.42% of pixels** (MEASURED n600, §3) — a 238×
   spatial sparsity — and the measured ideal speedup from exploiting it is **1.00×**, because the
   exact receptive field closes to the full frame at halo 685 and **23 squeeze-excite blocks**
   (count RE-DERIVED from the weights here) make every output globally dependent regardless. Deleting
   the *entire* SegNet backward while keeping the exact forward is a measured **1.2993×** ceiling.
   The head — the one place the rank argument bites exactly — is **725 of 9,610,645 SegNet
   parameters (0.0075%)** and 1.17% of measured fwd+bwd time. §2, §4.

3. **The oversized object is the actuator's DOF, not the scorer's layers.** `ddm_bp2` pointed
   692,712 coordinates at a rank-6 target and measured `cos(e, Jδ) ≈ −0.03…−0.06`: **>99.6% of its
   reach landed in pose dimensions that were already correct.** Reparametrized to the 6+1 scalars the
   rank actually justifies, the price collapses **95.5×** (`ddm_pb3`). *That* is where "train the
   least" pays, and it is a description-length statement, not a wall-clock one. §5.

4. **One concrete unlock, priced: the standing #456 NO-GO is 1 pixel in 117,964,800.** A measured
   **2.956×** exact-forward transfer was refused because 15/600 pairs mismatched by argmax digest. At
   n600 exactly **one** source pixel of 117,964,800 sits below the pair-78 tie margin (2.384e-7), and
   **34 of 600 frames** hold ≥1 pixel below 1e-5 — bracketing the observed 15/600. A fail-closed
   per-pair tie guard at τ = 1e-5 keeps **2.53× and stays argmax-bit-identical by construction**. §6.

5. **New campaign law, MEASURED n600, free:** the source-margin density is **constant at
   ρ = 0.0282 ± 0.0003 (0.7% variation) across t ∈ [0, 0.2]** — an interval that contains both our
   operating point (t\* = 0.1527) and the PR130 floor (t\* = 0.0105). **d_seg is exactly linear in
   the effective logit-gap reach over the entire remaining seg distance. There is no wall.** §3.

---

**Reproducer (every number below, one command, $0, no scorer pass):**
`.venv/bin/python tools/ddm_tl1_scored_dims_reductions.py [--json] [--weights-only]`

---

## 1. Method and what was and was not fired

Everything below is one of: (a) re-derived from the frozen safetensors (no forward pass), (b) reduced
from the cached `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (`margins`, shape
`(600, 384, 512)`, float32 — the source-side top1−top2 logit gap at n600), or (c) quoted from a landed
receipt with its axis and closure carried. **No SegNet or PoseNet forward/backward was executed by
this arm.** The n600 scorer slot was not requested and not touched.

Two authority corrections applied to my own brief before using its numbers, both from
`codex_findings_throughput_nogo_naive_rescope_audit_20260714_codex.md`:

- **"the frozen scorer is ~95% of training wall-clock" has NO current-loop authority.** It came from a
  stripped seg-only B8 MLX closure (SegNet fwd+bwd 69%, INR trunk 26%, R 5%) or an older n8 full-stack
  CPU closure (98.36%), neither of which is the live schedule. The audit's stated settled authority is
  the **MEASURED n96 1-thread verdict** timing. I use that and label the rest by its closure.
- **Task `#495` is a phantom.** The audit searched and found no canonical task or source object
  (its row 40). The number my brief attributes to it is real but belongs elsewhere: it is
  `pose_fraction_of_verdict = 0.226` in `.omx/research/frozen_scorer_verdict_wallclock_n96_20260714.json`
  — a **forward-only verdict** share, not a training-step share. Per `m89`, ids `#484 #486 #487 #495
  #449 #141 #539 #581 #583` are all ABSENT from `.omx/state/canonical_task_status.jsonl` (148 unique
  ids); only `#455` and `#456` resolve there, both `completed`. Cited by content below.

## 2. T1 — the necessity map

### 2.1 The seg head: rank-4 is real, exact, and worth 1.25×

RE-DERIVED, $0, from the frozen weights (independent of SPEC_v10 [E17], which it reproduces):

| quantity | measured (float64) |
|---|---|
| head | `Conv2d(16 → 5, k=3)`, i.e. 5 affine logits over each 144-D `16×3×3` patch |
| centered singular values | `3.1283763256, 2.1542713873, 2.0247078699, 1.7962638357, 3.7304e-16` |
| rank-4 reconstruction max abs error | `4.996004e-16` (float64); `5.960464e-08` in float32 |
| σ1/σ4 | `1.741602` |
| head parameters | **725 of 9,610,645 (0.0075%)**; learnable total 9,543,831 |
| head share of measured SegNet fwd+bwd | **1.17%** (block profile, §4) |

σ5 ≈ 0 is exactly the argmax gauge: adding `c·1` to the logit vector cannot change `argmax`. So the
scored content of the head output is genuinely 4-dimensional, exactly. **Its value is 5→4 = a factor
1.25 on an object that is 1.17% of the slice ⇒ 0.23% of SegNet time.** Not a lever.

*Precision note (reproduction detail, not a discrepancy):* [E17] reports float64 singular values
alongside a float32 reconstruction error. Both halves reproduce here — float64 gives σ5 =
`3.7304048124e-16` and recon error `4.996e-16`; casting the head to float32 first gives recon error
`5.960464e-08`, [E17]'s figure exactly. Nothing about the rank changes.

### 2.2 The reduction does NOT propagate backward — the decisive new negative

**Claim under test:** "everything upstream matters only through the rank-4 projection ⇒ ≤10 scored
dimensions per site ⇒ we can stop training most of the trunk."

**MEASURED refutation.** The head is a 3×3 conv, so one 16-channel decoder feature *location* is read
by **9** output sites, each through a *different* 3×3 tap. Stack the 9 centered per-tap maps
`Wc[:, :, dy, dx]` (each 5×16) into a 45×16 matrix and take its rank:

```
per-site   : rank(Ac) = 4 of 144 patch dims  ->   2.78% of the patch space is scored
under 3x3 overlap: rank(M) = 16 of 16        -> 100.00% of the channel space is scored
singular values of M: 2.7269 1.9484 1.7600 1.6850 0.9812 0.8958 0.7511 0.6920
                      0.6251 0.6192 0.5264 0.5157 0.4214 0.3508 0.3015 0.2602
condition number sigma1/sigma16 = 10.4806
```

Rank 16 of 16 at condition number 10.5 is not "nearly degenerate" — it is a healthy full-rank map.
**There is no invisible subspace of the penultimate representation to stop training.** The 140/144
per-site invisibility is destroyed by overlap at the very first layer you back up through.

**Verdict scope:** exact, for the frozen `segmentation_head.0` 3×3 conv and the 16-channel decoder
output that feeds it. It does not by itself prove that no *deeper* layer has a low-rank scored
subspace — but it removes the only place the rank-4 argument could have entered, and it removes it at
the layer where the argument was strongest.

### 2.3 Spatial necessity: closed, and closed twice over

From `cheapen_real_95_tilehalo_fp16_20260713.md` (n600, exact pair IDs 0…599), with the second leg
RE-DERIVED here:

- Exact integer receptive-field recurrence through the U-Net gives max reach **685 left / 654 right**;
  that support **clips to the entire 384×512 frame for every selected output**.
- **All 23 EfficientNet MBConv blocks contain squeeze-excite.** I re-counted this from the frozen
  weights directly (23 distinct `encoder.model.blocks.{i}.{j}.se.*` modules; enumerated in the run
  log for this memo). Each spatial mean makes the forward **and its VJP** globally dependent even if
  the local receptive field were small.
- Exact source area after closure: **1.0**. Exact ideal speedup ceiling: **1.00×**.

**Consequence.** Localizing exact scorer compute to the boundary annulus is not "hard" — it is
**structurally impossible for this oracle**, twice independently. Any proposal of that shape must be
labeled an approximation with a trust radius, never an exact saving.

### 2.4 Backward necessity

| question | measured answer | source |
|---|---|---|
| can the backward be truncated / early-exited exactly? | **No.** Final logits depend on every encoder and decoder block; freezing weights does not remove input-gradient propagation | necessity memo §2.5 |
| what is deleting the *whole* backward worth? | **1.2993× ceiling** on the scorer slice, before rest-of-loop | necessity memo §2.4, paired-sample median |
| is the seg backward already minimally seeded? | **Yes.** The relaxed loss is a scalar ⇒ one VJP. There is no second output dimension to drop | source, `make_loss_fn` |
| is the pose backward already minimally seeded? | **Yes.** `d_pose = ‖e‖²/6` over 6 outputs ⇒ one VJP seeded by `e`. You cannot do fewer than one | `upstream/modules.py::PoseNet.compute_distortion`; `ddm_pb3` |

### 2.5 Pose: the six dimensions are real, and they are already free

`compute_distortion` uses `out[...][..., : h.out // 2]` — the head emits 12, the first **6** are
scored. So half the pose head output is genuinely unscored. Priced: `hydra.final_layer.pose` is
`Linear(32 → 12)` = 396 parameters of PoseNet's **13,943,652**. Dropping the unscored half saves
**192 parameters = 0.0014% of PoseNet**. The parameter mass is `vision` 62.22% / `summarizer` 22.59% /
`hydra` 15.20%, all of it upstream of and shared by the 6 scored outputs.

**So on pose too, "only train against the necessary dimensions" buys nothing in the scorer graph.**
`rank(J) ≤ 6` is a statement about what a *perturbation* can achieve, not about what the network costs.

## 3. T2 — the honest decomposition, and the discard fraction

### 3.1 Where the scored mass actually is (MEASURED n600, $0, cached GT)

`gt_n600.npz::margins` is the source-side top1−top2 logit gap on all 600 pairs, 117,964,800 pixels.
A witness pixel can flip only if the logit-gap perturbation exceeds that pixel's source margin, so the
margin CDF *is* the flip-capability gate.

| margin < t | pixels (of 117,964,800) | % of pixels | ≈ px/frame |
|---:|---:|---:|---:|
| 2.384e-7 (pair-78 tie) | **1** | 0.000001% | 0.002 |
| 1e-5 | 36 | 0.000031% | 0.06 |
| 1e-4 | 335 | 0.000284% | 0.6 |
| 1e-3 | 3,353 | 0.002842% | 5.6 |
| 1e-2 | 33,212 | 0.028154% | 55.4 |
| 0.1 | 333,078 | 0.282354% | 555 |
| **0.153053 (our operating point)** | **508,616** | **0.431159%** | **848** |
| 0.25 | 827,002 | 0.701058% | 1,378 |
| 2.0 | 5,701,511 | 4.833231% | 9,502 |

Independent corroboration: the landed `bulk_boundary.px_share = 0.04736597696940104` (n600, a
differently-defined annulus) lands within 2% of my `margin < 2.0` share of 4.8332%.

**Inverting the CDF at our realized d_seg** (equal-count reading — see the caveat below):

| d_seg | equal-count source-margin threshold t\* |
|---|---:|
| 0.0043116 — live base `dc1_fold`, DERIVED from the gap decomposition | **0.153053** |
| 0.0038892 — burn ep399, MEASURED | 0.137906 |
| 0.0002966 — PR130 demonstrated floor | **0.010518** |

**⇒ The seg scored-active set is 0.42% of pixels.** Everything scored happens in the thinnest 1/238th
of the frame.

*Caveat carried with the number:* the equal-count inversion assumes the flip set is exactly the
lowest-margin set. Since every flipped pixel must satisfy `|Δ| > margin`, t\* is a **lower bound** on
the effective reach; the true reach is ≥ t\*. It is a well-defined, reproducible summary statistic,
not a claim that the flips *are* those pixels.

### 3.2 The new law: the margin density is flat where the whole seg gap lives

| t interval | ρ(t) = dP/dt |
|---|---:|
| [0.000, 0.005] | 0.028242 |
| [0.010, 0.020] | 0.028166 |
| [0.040, 0.060] | 0.028278 |
| [0.080, 0.100] | 0.028124 |
| [0.125, 0.150] | 0.028040 |
| [0.175, 0.200] | 0.027863 |
| [0.400, 0.500] | 0.027007 |
| [0.750, 1.000] | 0.025170 |

`ρ = 0.0282 ± 0.0003` — a **0.7% variation across [0, 0.2]**, and [0, 0.2] contains *both* our
operating point (0.1527) and the PR130 floor (0.0105). Decade-scale check: `N(<1e-4) : N(<1e-3) :
N(<1e-2) : N(<0.1)` = 335 : 3,353 : 33,212 : 333,078, ratios **10.009 / 9.905 / 10.029**.

> **LAW (MEASURED n600, source-side):** `d_seg ≈ 0.0282 · r`, where `r` is the effective source-margin
> reach in frozen-SegNet logit units, valid to 0.7% over the entire interval separating us from the
> PR130 floor.

Two consequences that change how we should plan:

- **There is no wall.** No threshold, no cliff, no diminishing return between here and the floor.
  Every 1% reduction in reach buys exactly 1% of d_seg, linearly, for 14.5× in a row.
- **The seg gap is one scalar.** `r: 0.1527 → 0.0105` is a **14.54×** reduction in the mean logit-gap
  perturbation the render delivers. (The 14.54 is arithmetically just the d_seg ratio — ρ cancels.
  The non-trivial content is that ρ is *constant*, so the map d_seg ↔ r is a single-constant linear
  bijection over the whole distance, and that both endpoints are ordinary interior points of a flat
  density rather than positions on a shoulder.)

### 3.3 The wall-clock decomposition, by closure, with authority attached

| closure | measurement | authority |
|---|---|---|
| **verdict (forward only), n96, 1-thread CPU-torch** | combined 59.615 s, 0.621 s/pair; **SegNet 0.774 / PoseNet 0.226**; n600 = 372.6 s DERIVED by linear projection | the settled authority-verdict timing (`frozen_scorer_verdict_wallclock_n96_20260714.json`) |
| **SegNet slice, 1 pair, CPU, hook-instrumented** | forward **76.97%** / input-backward **23.03%**; fwd 1,443.996 ms, bwd 443.975 ms, paired total 1,887.971 ms | `experiments/results/segnet_block_profile_20260712T151901Z/profile.json` |
| **stripped seg-only MLX B8 step** | SegNet fwd+bwd 69% · INR trunk 26% · R 5% | necessity memo §2.3; a stripped closure, **not** the live V9 schedule |
| **older full-stack torch-CPU n8 step** | scorer fwd+bwd 98.36%; backward 68.16%, SegNet fwd 27.37%, PoseNet fwd 2.83% | `scorer_step_profile_20260612.md`; superseded as a current-loop claim |

Per-block, on the CPU slice: encoder blocks 5+6 = **37.66%** of measured total; block 5 alone 24.24%;
the segmentation head **1.17%**; encoder block 1 is the *memory* outlier (175.23 MiB) not the time one.

**Note the two profiles disagree on fwd-vs-bwd direction (77/23 vs 32/68).** They are different
closures and different instrumentation (hooks add overhead; the n8 backward is an estimate). I do not
average them. Where a decision depends on the split, the 1.2993× backward-deletion ceiling from the
*paired* CPU measurement is the one with a stated paired-sample method, and it is the conservative one.

### 3.4 The discard fraction, stated honestly

| axis | scored fraction | discard | recoverable as compute? |
|---|---:|---:|---|
| seg, spatial | 0.42% of pixels | **99.58%** | **No** — halo 685 + 23 global SE ⇒ exact area 1.0, ideal ceiling **1.00×** (MEASURED) |
| seg, head output gauge | 4 of 5 logits | 20% | Yes but worthless — 0.23% of the SegNet slice |
| seg, penultimate channels | 16 of 16 | **0%** | **Nothing to recover** (§2.2, rank 16/16, cond 10.5) |
| pose, head output | 6 of 12 | 50% | Yes but worthless — 192 of 13,943,652 params |
| pose, perturbation directions | 6 of 692,712 (`bp2`) | **>99.6%** measured wasted | **Yes — and this is the real one.** §5 |

## 4. T3 — what we can stop computing, ranked, with the bit-identity constraint applied

Hard constraint honored throughout: **a cheaper forward that changes one argmax pixel is not a saving,
it is a corrupted measurement.** A loss-scalar match is explicitly *not* an acceptable control (§7).

| # | move | measured value | risk to the scored quantity | status |
|---:|---|---|---|---|
| 1 | **Tie-guarded 1-thread exact forward** (§6) | **2.53× exact** (from a measured 2.956× raw) | zero *by construction* if the guard threshold is measured to bound the cross-arm logit deviation — that measurement is OWED | **BUILDABLE NOW; the standing NO-GO is 1 px in 118M** |
| 2 | **Duplicate same-frame f1 SegNet call elimination** | up to one full SegNet forward per step when a surgical raw-margin lever is active; exact CSE, changes no function | zero — compute raw logits once, add the class offset for the base path | recorded 2026-07-12 as "the highest-confidence residual exact optimization"; **verify whether it ever landed before re-proposing** |
| 3 | **Skip PoseNet in seg-only verdict phases** | **22.6%** of verdict wall-clock (MEASURED n96) | zero to d_seg; **non-zero to us** — you go blind on the axis that has historically diverged (`m85`: seg-only base ⇒ pose ρ149/6.36, carrier-dependent) | conditional; requires a matched-base control ≥32 pairs before composing |
| 4 | Last-frame-only; static GT/label/margin caching; shared surgical-margin forward | already exact and already exploited | none | **DONE** — this is why the honest headline is "close to minimal" |
| 5 | Spatial tiling / annulus-restricted exact forward | **1.00×** | — | **STRUCTURALLY REFUSED** (§2.3) |
| 6 | Channel pruning at the penultimate | none available | — | **REFUTED** here (§2.2) |
| 7 | Truncated / early-exit backward | not exact | corrupts the metric | **REFUSED** |
| 8 | Deleting the backward entirely (cache/student) | **1.2993×** ceiling on the slice | large — a forward-accurate student is measured non-equivalent in descent (0.3087 / 0.2327 vs 0.0058) | low priority: the ceiling does not justify the fidelity risk |
| 9 | Activation checkpointing | throughput-**negative** (250.2 ms replay to release 337.9 MiB) | — | only if it unlocks a measured larger batch |

**The ranking's own headline:** rows 4–9 say the scorer graph is done. Rows 1–3 are worth roughly
2.5× on the verdict/forward path and nothing structural. **If "train least" means "make the frozen
scorer cheaper," the honest answer is that we are within ~2.5× of the floor and the remaining factor
is a thread-count guard, not a mathematical insight.**

## 5. Where "train the least" actually pays: the actuator, not the scorer

The directive's premise — that the scored space is tiny — is *true*, and its correct consequence is a
statement about the **actuator's degrees of freedom**, which is a description-length statement:

- `ddm_bp2` optimized **692,712** blind-set coordinates against a **rank-6** target. Measured
  `cos(e, Jδ) = −0.027 … −0.059`, stable across 4.5 decades of `k`. **>99.6% of its reach landed in
  pose dimensions that were already correct.** Its median per-pair gain was 0.198%; its headline
  65.9% mean was a tail artifact (top 1% of pairs = 62.1% of the total reduction).
- `ddm_pb3` reparametrized to **6 coefficients + 1 density = 7 scalars/pair**, DOF-justified by
  `rank(J) ≤ 6` plus the ∞-norm box scale. Payload **401,285 B → 4,200 B**, ΔS_rate **+0.26720 →
  +0.00280**, break-even from "reduce d_pose 99.89%" to **"reduce mean d_pose 2.01%"** — **95.5×**.
  (Its §5 realized-capture measurement is still OWED; the price collapse is not.)

**That is the operator's directive, correctly localized.** The layers are not the oversized thing.
The *coordinates we point the gradient at, and then ship*, are. This is the same statement as the
crossing the campaign already uses: the frozen head's cell geometry is generic ⇒ free in `inflate.py`;
don't train what the head determines and don't ship what the receiver regenerates.

The seg analogue of pb3's `cos` is **not** yet measured and is the single highest-value follow-on this
arm identifies: the fraction of the training gradient's mass that sits on pixels which cannot flip.
§3.1 gives the target set (0.42% of pixels); what is missing is the measured mass share of `∂L/∂x`
outside it. That is one backward pass at n600 against the cached margins — cheap, and it would price
the seg actuator the way pb3 priced the pose one.

## 6. The #456 unlock, priced

`cheaper_exact_forward_transfer_95kill_20260713.md` (task #456, `completed`) measured a **process-static
one-thread eager NCHW frozen-SegNet forward** at **2.9562855478×** (Torch 2.12.1) and **2.9970427×**
(Torch 2.12.0), n600, 600/600 timing wins, p = 2.41e-181. It was refused because **15/600** per-pair
argmax digests mismatched: pairs `[35, 78, 88, 120, 131, 132, 140, 196, 214, 242, 327, 395, 468, 495,
578]`. Pair 78's top-two margin at `(y=275, x=356)`, classes 0/1, was **2.384185791015625e-7** — fp32
epsilon. Flip pixel counts were never recorded ("UNKNOWN"), so the blocker was never sized.

**Sizing it, MEASURED n600 from cached GT margins:**

| τ | pixels < τ (of 117,964,800) | **frames with ≥1** | frame fraction | net exact speedup with a fail-closed per-pair fallback |
|---:|---:|---:|---:|---:|
| 1e-6 | 3 | 3 | 0.0050 | **2.913×** |
| 3e-6 | 12 | 12 | 0.0200 | **2.791×** |
| **1e-5** | **36** | **34** | **0.0567** | **2.532×** |
| 3e-5 | 101 | 88 | 0.1467 | 2.062× |
| 1e-4 | 335 | 250 | 0.4167 | 1.325× |

`S(τ) = T_ref / (T_fast + f(τ)·T_ref)` with the receipt's `T_ref = 893.005052` ms/pair,
`T_fast = 302.06995825` ms/pair. The observed 15/600 = 0.025 mismatch rate sits between
`f(3e-6) = 0.020` and `f(1e-5) = 0.0567`, so **the GT-margin distribution reproduces the observed
blocker rate to within a factor of ~2 from a completely independent artifact.**

**The proposal.** Run the fast arm on every pair. Flag any pair holding a site with realized
`|top1 − top2| < τ`. Recompute exactly those pairs with the reference arm. At τ = 1e-5 that is **5.7%
of pairs** and the result is **argmax-bit-identical by construction** — because any unflagged site has
margin > τ, and if `max|Δlogit_fast − Δlogit_ref| ≤ τ` then its argmax cannot differ.

**Pre-registered falsifier, and the thing that must be measured before this is believed:** the guard
is sound **iff** `τ` upper-bounds the actual cross-arm logit deviation. The #456 receipt never
recorded logit deltas. The owed measurement is one n600 two-arm pass recording
`max |logit_fast − logit_ref|` and the realized margin at every disagreeing site. If that max exceeds
τ, raise τ and re-read the table; if it exceeds 1e-4 the lever is worth <1.33× and should be dropped.
**Do not ship this on the GT-margin proxy** — GT margins are the source side; the realized-render
margins are a different distribution (this arm used GT because it is cached and free, and labels the
substitution here rather than hiding it).

**Scope kept from the source:** #456's negative closes only the tested eager-NCHW static-thread
formulation on two fingerprinted local builds. The tie-guarded form is a *new* formulation, untested.

## 7. T4 — the standing bit-identity hazard, and what my own proposal owes it

Two prior findings define the control that any "cheaper path" must clear, and both are load-bearing
for §6:

- **`ddm_mi1` (#855):** the default MLX conv adapter flips **76 argmax pixels on real frames,
  systematically.** Corroborating context from `docs/mlx_contest_scorer_port_guide.md`: the owned MLX
  port deviates on `243 / 19.66M` pixels (1.24e-5), "**all boundary near-ties**" — the same population
  §3.1 measures, at the same order of magnitude (my `margin < 1e-4` share is 2.8e-6; `< 1e-3` is
  2.8e-5).
- **`ddm_bp1` (#903):** an upsample-VJP scatter × Adam `sign(g)` divergence in which **the loss scalar
  was identical while 40 of 41 arrays diverged.** ⇒ **a loss-scalar match is not a bit-identity
  control.**

**Applied to my own §6 proposal:** the control is the per-pair ordered argmax SHA-256 vector that the
#456 receipt already produces (8 independent children per build), plus the logit-delta bound above.
Not a loss match, not a d_seg match, not "the digests agree on the pairs we checked." The guard's
correctness argument is a *margin inequality*, so the measurement that validates it must be a *margin
and logit-delta* measurement.

**And the same hazard bounds my §3 numbers:** the margin distribution near zero is exactly the regime
where kernel/reduction-order differences change the answer. My CDF is computed on the *cached*
`gt_n600` margins produced by one path; a different path would move the sub-1e-5 tail. Everything in
§3.1 above `1e-3` (≥ 3,353 pixels) is robust to that; the `1e-7…1e-5` rows are the ones where the
producer identity matters, and they are used only for *ordering-of-magnitude* pricing in §6, never as
an exactness claim.

## 7.5 Round-1 adversarial self-review — four controls, all executed

Attacking my own conclusions before handing them over. Every control below was **run**, not reasoned.

**C1 — can my rank probe return the negative?** (`m50`: a probe that cannot return the negative is not
a probe.) Given a *single* centre tap instead of all nine, the identical code returns **rank 4 of 16**.
Given all nine taps it returns **16 of 16**. The probe is capable of the low-rank answer and did not
give it. §2.2 stands.

**C2 — is `gt_n600.npz::margins` actually the top1−top2 argmax gap?** My entire §3 rests on this and I
did not produce the array. Structural test on pair 0: a low-margin pixel must sit on a class boundary
in `lstars`.

| | P(4-neighbour class boundary) |
|---|---:|
| all pixels | 0.023198 |
| **given margin < 0.1** (n = 566) | **0.991166** |
| given margin > 4 | **0.000000** |

99.12% vs a 2.32% base rate, and *exactly zero* in the high-margin bulk. The array is the argmax gap.
CONFIRMED. Also `min = 0.0`, `max = 16.2358`, no negatives.

**C3 — exact CDF, no interpolation.** `P(margin < 0.153053) = 508,616 px = 0.431159%` (848 px/frame);
`P(< 0.010518) = 34,989 px = 0.029661%` (58 px/frame). The §3.1 table row is now exact.
*Non-check flagged:* `P(margin < t*_burn) = 0.388920%` coincides with the burn's measured
`d_seg = 0.0038892` to five figures — but that is **circular**, because t\* was defined by inverting
the CDF at that d_seg. It is not corroboration and is not offered as such.

**C4 — the tie-guard arithmetic.** `T_ref/T_fast = 893.005052/302.06995825 = 2.9562855478`, matching
the #456 receipt's `2.9562855478032297` to 10 digits, so the inputs to the §6 table are the receipt's
own. Every `S(τ)` row re-derived from the measured `f(τ)`.

**What the review did NOT close.** (i) §3's sub-1e-5 tail depends on the producer identity of the
cached margins (§7). (ii) The §6 guard is unvalidated until the logit-delta bound is measured — I have
sized the blocker, not cleared it. (iii) §2.2 refutes low-rank at the head; it does not survey deeper
layers. (iv) Both t\* values are lower bounds on reach (§3.1 caveat), so §3.2's `r` values inherit that.

## 8. Verdict, scope, and what is owed

**VERDICT: the operator's concern is answered, and the answer splits.** `verdict_scope: the frozen
`tu-efficientnet_b2` SMP U-Net SegNet and FastViT-T12 PoseNet at their canonical 384×512 / YUV6
inputs, on the frozen weight SHAs above; static-weight algebra and n600 cached-margin reductions; no
new scorer execution.` Pointer **UNMOVED**.

**MEASURED and durable:**

1. Seg has **no** low-dimensional necessary sub-network. Rank-4 is a per-site output gauge worth 1.25×
   on 1.17% of the slice; it is destroyed one layer up by 3×3 overlap (**rank 16/16, cond 10.48**).
2. The seg scored-active set is **0.42% of pixels** (n600) and the **99.58% discard is architecturally
   non-recoverable** — halo 685 + **23** SE blocks (re-counted from weights) ⇒ exact area 1.0, ideal
   ceiling **1.00×**.
3. **ρ = 0.0282 ± 0.0003 across [0, 0.2]**: `d_seg ≈ 0.0282 · r`, exactly linear over the entire
   remaining seg gap. **No wall between us and the PR130 floor; the gap is a 14.54× reduction in one
   scalar.**
4. Pose's `rank(J) ≤ 6` buys nothing in the scorer graph (192 of 13.9M params) and **everything in the
   actuator** (bp2 >99.6% waste → pb3 95.5× price collapse).
5. The **#456 blocker is 1 pixel in 117,964,800**; a τ = 1e-5 fail-closed per-pair tie guard keeps
   **2.53× exact**, and the GT-margin distribution independently reproduces the observed 15/600 rate.
6. Two brief-level provenance corrections: the "~95% of wall-clock" figure has **no current-loop
   authority**, and **task #495 is a phantom** (its number is `pose_fraction_of_verdict = 0.226`, a
   forward-only verdict share).

**OWED, in priority order:**

- **A.** The seg analogue of pb3's `cos`: measured mass share of `∂L/∂x` outside the 0.42% flip-capable
  set, n600, one backward against the cached margins. This is the measurement that would price the seg
  actuator's DOF the way pb3 priced pose's. **Highest value of anything named here.**
- **B.** The §6 logit-delta bound (`max |logit_fast − logit_ref|` at n600) — the single number that
  converts a standing NO-GO into a 2.53× exact lever.
- **C.** Confirm whether the duplicate-f1 CSE (§4 row 2) ever landed. It is exact, free, and was
  identified 3 weeks ago.
- **D.** `ddm_pb3` §5 (realized capture `η ≥ 0.0047`) remains OWED and un-softened; nothing here
  changes that.

**NEXT-IF-RESUMED:** start at **A**. Everything needed is on disk: `gt_n600.npz::margins` for the
target set, and the trainer's existing backward for the gradient field. Do **not** re-derive §2, §3 or
§6 — run `tools/ddm_tl1_scored_dims_reductions.py` instead; it reproduces every number in them in one
command with no scorer pass. If A returns a large discard fraction, the seg actuator has the same
disease bp2 had and the cure is the same shape: reparametrize to the DOF the geometry justifies, do
not add capacity.

**Triality.** *DAG:* this memo (a necessity map + two authority corrections + one priced unlock).
*DSL:* no new lever — §4 rows 1–3 are changes to existing exact surfaces and must acquire typed,
resume-safe control before execution; nothing here compiles a trainer flag. *Equations:* the §3.2
density law is a candidate canonical equation (`d_seg ≈ ρ·r`, ρ = 0.0282 on [0, 0.2], one MEASURED
n600 anchor); **not minted here** — a single-anchor law from one cached artifact should acquire a
second, independent anchor (the owed item **A** would supply it) before registration.

---

*STORES CONSULTED:* `upstream/modules.py` + `upstream/models/{segnet,posenet}.safetensors` (re-hashed,
the §2 algebra is computed from them directly); `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`
(the §3/§6 reductions); `.omx/research/ddm_pb3_parametric_blind_set_20260802.md` and
`ddm_bp2_blind_set_pose_actuator_20260802.md` (the pose half); `frozen_segnet_necessity_optimality_alternatives_20260712.md`
(the block profile, the 1.2993× backward ceiling, the duplicate-f1 finding);
`cheapen_real_95_tilehalo_fp16_20260713.md` (halo/SE closure, annulus px_share);
`cheaper_exact_forward_transfer_95kill_20260713.md` (#456, the 2.956× and the 15 pairs);
`frozen_scorer_verdict_wallclock_n96_20260714.json` (the settled verdict timing);
`codex_findings_throughput_nogo_naive_rescope_audit_20260714_codex.md` (the two authority corrections);
`scorer_step_profile_20260612.md` (the superseded n8 closure, quoted as such);
`SPEC_v10_integer_plane_vehicle_20260719.md` [E17]/[E20];
`ddm_pc2_perclass_road_edges_20260802.md` (the edge/interior framing);
`docs/mlx_contest_scorer_port_guide.md` (the near-tie deviation population);
`tac.canonical_equations.gap_decomposition_against_floor_20260802` (the denominator);
`.omx/state/canonical_task_status.jsonl` (the id-absence check, per `m89`).
