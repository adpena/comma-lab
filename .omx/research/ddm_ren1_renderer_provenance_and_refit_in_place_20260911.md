# ddm_ren1 — the shipped renderer's training provenance, and the refit-in-place RE-PRICED: the pose wall is a capacity, not a ratio

`[no-triality] [p0-ledger-ok]` · research_only=true · score_claim=false · promotion_eligible=false ·
pointer_moved=false

<!-- # FORMALIZATION_PENDING: the two numbers this arm proposes as laws -- the payable post-re-solve coupling ceiling k_payable = 100/(5/sqrt(10*d_pose_base)), which is an exact re-derivation of the score's own arithmetic at a named operating point rather than a new equation, and the carrier-absorption CAPACITY finding, which is a three-arm n600 measurement on one object -- are registered as canonical equations only when a second arm re-derives k_payable at a different pointer and a second object re-measures the capacity, per the binding-numbers-expire discipline; n=1 on the capacity, and this memo says so. The renderer provenance table is a custody record, not a model. -->

Base: move 48, S 0.13638261682704697 @ 179,111 B `[contest-CUDA T4 n600]`, archive
`d830edd371641e1968765ae6be27120a6c55a9b1ca4b3747158e603a43ef149c`; components d_seg 0.00010345,
d_pose 4.59e-06; admit bar −2e-5.

---

## ANSWER FIRST

**The pointer did not move and this arm shipped no candidate.** Half 1 is complete, **the gate
PASSES on both legs**, and Half 2 did not fire for reasons stated as a decision in §4.6 rather than
as a result.

**Leg 1 — the renderer is stale in every sense the charter asked about, with receipts.** It is not
ours: a banked PR130 third-party artifact, trained **seg-only** (no PoseNet, no RGB), on **GT-oracle
tokens** rather than any coded field, with **no EMA**, against a **contested GT lineage**, through a
**uniform int4 quantizer that is not the mixed-{3,4}-plus-99 %-row-prune one it ships in**. The field
has moved eight times since; the weights have never moved at all.

**Leg 2 — and here the arm falsified its own draft.** I first refused the refit on `ddm_pr1`'s
post-re-solve coupling `k_post = 13.82` against a payable ceiling of 0.1355 — a 102× gap. Then I
attacked that, because `k` belongs to a DIRECTION and pr1 measured one candidate. The attack
succeeded, at n600, on three arms of the canonical unforked solver:

> **The carrier's pose absorption is a CAPACITY, not a ratio.** A 0.5-LSB render change — the
> amplitude at which a refit first moves the shipped bytes — costs **7.4×** (noise) to **142.9×**
> (smooth) the pointer's whole `d_pose` before re-solve. After the terminal carrier re-solve it lands
> within **2.5 %** and **7.7 %** of the shipped level, for **+2 B** and **+4 B**. `k_post` measures
> **0.0078** and **0.0190** — **17.4× and 7.1× INSIDE** the payable ceiling, where pr1's 13.82 is
> 102× outside. In score units the re-solve removes **140.6×** and **291.4×** of the pose leg.

So **a renderer refit is not refused by pose.** Its target is a **1.0 %–2.7 % `d_seg` cut**
(spectrum-dependent), plus an unmeasured member re-encode. **The binding term is SEG** — exactly
where `ddm_pr1` said it was — and seg is not closed: `ddm_rw1`'s 241 evaluations found 0.0 % at
FORMULATION scope without the depth table or the prune mask in the loop, and `ddm_ft1`'s +31.23 %
was measured on an object that is not the one that ships (§3A).

**Two results the charter did not anticipate:**

* **Feeding the renderer the TRUE partition makes `d_seg` 2.825× WORSE** (2.9210e-04 vs 1.0339e-04).
  The shipped token field is not an approximation of GT the renderer copes with; it is an optimized
  **pre-image** that already absorbs this renderer's bias. That render floor nonetheless **clears
  `ddm_obx2`'s 4.0e-04 reactivation gate** and is 7.9× below the QBF1 object's — handed forward.
* **`ddm_obx2`'s smooth-vs-noise law INVERTS on this object.** obx2 measured noise 9.40× worse than
  smooth and exported *"its render error must be SMOOTH"*; here smooth is **19.38× worse than noise**
  at matched RMS. Composed, the two objects disagree by 182×. Neither object's spectrum law may be
  quoted on the other.

**Three corrections owed upward**, all to numbers this arm's own charter carries:

1. **`k_post = 13.82` is a SATURATED-regime figure, not the coupling.** The charter quotes 170–220
   (pre-re-solve); pr1 refined it to 13.82; §3B shows both belong to amplitudes above the carrier's
   capacity. Quoting any of the three without the regime is the stale-headline genus
   [[corrections_land_in_bodies_headlines_keep_the_stale_number_20260805]].
2. **`17e0fd0b…` is 36,130 B, not 29,862 B.** 29,862 B is the brotli-compressed *member*, sha
   `786950a5…`. Both shas are right; the byte count belongs to the member. Verified by decoding the
   shipped archive (§1.1).
3. **`ddm_rf1`'s "the renderer axis cannot reach 0.12 even if the renderer ceases to exist" has
   expired at move 48** — re-derived, deleting the member at zero added distortion now gives
   **0.11649** (§6).

---

## 1. HALF 1.1 — the provenance table

Every row carries a receipt. Rows marked **unknown** were searched and not found; they are not
inferred.

### 1.1 What ships, measured off the move-48 archive

I decoded `/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_frame_even/candidate_runtime/archive.zip`
with the receiver's own splitter and un-coders. The chain and its sizes:

| stage | object | bytes | sha256 (16) |
|---|---|---:|---|
| RX1M member | `semantic` section (brotli) | **29,862** | `786950a5…` |
| after brotli + CK2 un-interleave | `SM1S` rider | **31,451** | `9c0b579c…` |
| after `sm1_semantic_mixer.restore_semantic` | **SM3R v1 mode 6 body** | **36,130** | **`17e0fd0b…`** |

RX1M header measured: `(b'RX1M', 1, 2, 0, 250, 11629, 29862, 18450)`. SM1S saves **4,679 B** over the
raw SM3R packet with 24 counted int8 shared-mixer weights.

| property | value | receipt |
|---|---|---|
| architecture | `SemanticTokenRenderer(width=96)`: `Embedding(5,96)` token · `Embedding(600,8)` frame · `Conv2d(100→96,1×1)` coord_mix (4 analytic coord channels `x,y,x²,y²`) · 4× `TokenBlock` = depthwise `Conv2d(96,96,3,groups=96)` at dilations **(1,1,2,4)** → pointwise `Conv2d(96,96,1)` → `GroupNorm(12,96)` → FiLM `Linear(8→192)` → GELU → residual · `head Conv2d(96→3,3×3)` → `sigmoid()·255` | `cpr1/inflate.py:74-130`, read out of the promoted tree |
| parameter count | **66,339** (I summed the shipped `state_dict`) | this arm; matches `ddm_ft1_…_20260903.md:67` *"66,339 params, 15,363 zero (23.16%)"* |
| conditioning | the ONLY per-pair signal is `frame_embed` — **8 floats per pair**, broadcast to all four FiLM layers. No temporal context, no phase channels. | same |
| quantizer, **as shipped** | per-tensor mixed depth, read out of the packet: **3 bit** on `frame_embed.weight` and `blocks.0.film.weight`; **4 bit** on the other 14 rank≥2 tensors. Scale = **per-axis max-absolute / (2^(b−1)−1)**, stored fp16, floored at `5.96e-08`; scale axis is `shape[-1]` for `*embed.weight`, else `shape[0]`. Rank<2 tensors (all biases, all GroupNorm) ship as raw fp16. | decoded here; encoder `experiments/ddm_sm3_semantic_representation.py:197-212` |
| row pruning | `keep_percent = 1` ⇒ `blocks.{1,2,3}.film.weight` keep **2 of 192 rows**; 190 rows are exact zeros in each. I confirmed nonzero FiLM rows = **{blocks.0: 192, blocks.1: 2, blocks.2: 2, blocks.3: 2}**. | decoded here |
| nonzero parameters | **50,976 / 66,339 (76.8 %)**; 0.474 member-bytes per parameter | this arm |
| **who trained it** | **no arm of ours.** Banked third-party PR130 intake artifact. *"`reproduce.sh` consumes banked TRAINED artifacts as inputs"* | `.omx/research/ddm_pr130_reproduce_20260809/PR130_REPRODUCED_HERE.md` |
| terminal checkpoint | `semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt`, 282,352 B, sha `3948ccfc…` (parent stage-07 `1549607d…`) | `…/RR2_SEMANTIC_LEG_AUDIT.md:52-53` |
| **losses** | **seg only. No PoseNet anywhere in the trainer. No RGB distillation (0/18,000 steps).** A phase curriculum, not a weighted sum: stage-07 CE 6,000 / softplus-margin 4,200 / expected-flip 1,800 of 12,000; stage-08 **expected-flip 6,000/6,000** at τ≈0.05 | `RR2…:69,73,74,75,174` |
| **field version trained against** | **none — GT-ORACLE tokens.** Conditioning is `cache["seg"]`, the GT argmax plane, not a coded field. The shipped field differs from DALI GT at **9,179** sites and from PyAV GT at **28,133**. | `src/tac/pr130_lift/lifted/train_semantic_quantized.py:210-217`; `ddm_ft1_…_20260903.md` §2 |
| GT lineage | **CONTESTED.** The checkpoints' own recorded metric names `/workspace/semantic-pose/gt_cache_600.pt`, which is **PyAV-like** (differs from the AV ruler at 1/117,964,800 sites); the repro e2e graph declares the DALI `official_targets.pt`. The original invocation is not recoverable. DALI-vs-PyAV differ at **20,671** argmax sites. | `SG2_FINDINGS.md:25`; `FX4_GT_LINEAGE.md:62-68,41` |
| training data | n600, full field, batch 2, seeded permutation per epoch | `RR2…:66` |
| epochs | **not recorded**; DERIVED 40 (stage-07) + 20 (stage-08) from 18,000 steps × batch 2 ÷ 600 | this arm, from `RR2…:66,70` |
| checkpoint selection | best **d_seg only**, evaluated every 250 steps (49 + 25 records). *"States between evaluations cannot win."* | `RR2…:77` |
| **EMA** | **NONE.** *"No EMA construction, update, shadow selection, or save site exists in the complete QAT trainer… The shipped stage-08 object is not an EMA shadow."* | `RR2…:79`, finding RR2-F4 |
| resumability | none: one `torch.save` after the loop; no optimizer/scheduler/RNG state persisted | `RR2…:237-243` |
| optimizer | AdamW, wd 0, cosine to 1 % floor; stage-07 lr 2e-5 / 12,000 steps, stage-08 lr 2e-7 / 6,000 steps; grad-norm clip 2; QAT on 18,000/18,000 steps; no AMP | `RR2…:70,71,76` |
| **quantizer, as TRAINED ≠ as SHIPPED** | trained through **uniform int4, limit 7, no row prune**; shipped is mixed {3,4} + `MODE_ROW_PRUNE` keep 1. *"The trained object and the realized object differed by `2.32e-3`"*, in **5/38 tensors and 9,414 elements** (`frame_embed` + all four `blocks.*.film`). | `ddm_rw1_…_20260909.md` §1; `ddm_wd4_warm_lineage_width_20260821.md` |
| the shipped depth table | **never a score optimum.** `blocks.0.film.weight` was put in the 3-bit set by a BYTES argument; SD1 had measured that cell at **`+4.92e-05` S, a loss.** | `ddm_ntb2_…_20260911.md:270,554-557` |
| producer commit / argv / date | **unknown**, searched `.omx/research/**`, SG2/FX4/RR2 receipts, `git log --all -S"3948ccfc"` and `--grep`. The only date-shaped values are the trainer seeds 20260715 / 20260716. | |
| ever retrained on the current field `a92e7d90…`? | **NO — clean negative**, independently re-verified: 30 `a92e7d90` hits in `.omx/{research,state}` are all tail/prior/carrier/token arms; moves 28→48 carry no renderer-weight lane; `git log -S"3948ccfc"` shows no retrain. Two arms *attempted* renderer training on post-move-30 fields (`ddm_rw1`, moves 33/35) and one pre-move-31 (`ddm_ft1`); **both accepted zero.** | `ddm_hpr1_staleness_audit_20260911.md` §3c + this arm's re-search |

**Read of the table.** Leg 1 of the gate is not marginal, it is emphatic. The object that renders
every scored frame was fit by someone else, to a different input distribution, against a possibly
different ground truth, under a different quantizer, with no EMA and no resume, and selected by the
better of 74 spot evaluations of a single metric. Nothing about it was chosen for the object we ship.

---

## 2. HALF 1.2 — the successor-check table

| arm | object | what it CLOSED | scope | what it left OPEN |
|---|---|---|---|---|
| `ddm_rf1` (08-24) | `film_amortized_flat_w96`, 179,290 B | *"REFUSED at 2.7749× the matched base… 97.59 % of it is pose."* d_seg rose 1.2384×, d_pose 94.97×. Coupling anchor **166.81**. | **INSTANCE** (one un-retrained structural change) | everything else; it is not a training verdict |
| `ddm_ft1` (09-03) | the shipped body `17e0fd0b…` | seg-only aligned fine-tune. *"step 600 d_seg ROSE to 0.000267529, +31.23 %"*; terminal +6.54 %; best exact-seg never left step 0. Coupling **217.30**. | **FORMULATION** (seg-only loss, no pose term) | *"the **joint** formulation stays open"* |
| `ddm_pr1` (09-04) | a renderer-change candidate, n600 | the terminal carrier re-solve **recovers 16.42×** (598/600 pairs improve), **k_post = 13.82** (k_pre 228.45), re-solve costs **+125 B**, candidate still **41.5× over the payable bar**; *"the reflected step raises d_seg 21.55× MORE than the forward step"* — **the axis closes on SEG, not on pose** | **FORMULATION**, three anchors | a direction with a genuinely lower k_post |
| `ddm_rw1` (09-09) | shipped weights, moves 33/35 | **five routes, one cause**: 3,000-step joint MPS run, random-direction control, exact-field gradient both ways, discrete realized code search (150 evals, **0.0 %**), per-row fp16 scale search (91+ evals, **0.0 %**). Law: *"one int4 code step in `head`/`blocks.3` already moves 240–455 argmax cells, so the fold-back is limited by the grid's resolution, not by the search."* | **FORMULATION** (AdamW on a continuous latent over a piecewise-constant forward) | the guided discrete realized search |
| `ddm_ntb2` (09-11) | shipped renderer precision | *"The renderer's 29,862 B are not purchasable."* Damage-per-byte **102× to 1,864×** the score's exchange rate; 3-bit closed on every layer; *"this section sits AT its distortion-rate knee."* | **FAMILY-at-one-formulation** | **"A quantization-aware refit is NOT closed."** Also: per-row mixed depth; `MODE_ROW_PRUNE` on pw/dw; growing the HPAC prior |
| `ddm_sd1` (08-09) | the **191,052 B PR130 ancestor**, not the shipped body | uniform q3/q5 negative; the shipped q4 allocation is **not** the semantic-leg optimum inside the research format (four-q3 wins 0.000424 S at n600) | **INSTANCE/FORMULATION** | *"Bit-depth family — OPEN"*; **`pose_status: NOT_MEASURED`** |
| `ddm_fe1` (09-08) | per-pair `frame_embed` pre-distortion | *"the mechanism this charter tested contributes NOTHING that survives."* The wall is the **CONTAINER**: a fixed ~+70 B break fee at the shipped shape, **+1.3 B at N=1 / +28.0 B at N=72** with a container search. Pose was measured **not** to be the killer. | **FORMULATION** | the container-search law it left behind (reused in §4) |
| `ddm_md1`–`md4` (09-04/05) | the **QBF1 born vehicle**, not this renderer | 62.0–63.0 % of terminal d_seg is **persistent** across schedule, data order and start; *"the born route needs a mechanism that changes which sites are reachable, not a schedule"*; step-0 probe predicts unreachability 9 in 10 | **FORMULATION**, other object | nothing about the shipped renderer; it supplies the step-0 DISCIPLINE this arm honoured |
| `ddm_ls1`/`ls2` (09-11) | the **tail**, current field | scorer-free receiver-probability work; ls2 nets **385.55 byte-equivalents, 1.49 % of the 25,899 B demand**; no fire | — | not renderer arms; they contribute the field sha and the rate denominator only |

**What is NOT closed, stated plainly:** the quantization-aware refit (ntb2's own words), per-row mixed
depth inside a tensor (a receiver-format change), `MODE_ROW_PRUNE` on the pw/dw tensors, the guided
discrete realized code search (rw1's successor), and a **joint** (pose-priced) renderer loss on the
current field. This arm prices the last of those and refuses it; it does not touch the other four.

---

## 3. HALF 1.3 — the step-0 probe (n600, frozen CPU scorer)

`experiments/ddm_ren1_step0_probe.py`, one pass over **all 600 pairs**, four treatments sharing one
instrument state. Axis `[macOS-CPU advisory, jg1/up2 instrument, DALI GT lineage]`. 1,620 s.

### 3.1 The instrument passes its own positive control

| quantity | this probe (n600, CPU advisory) | the pointer's T4 row | agreement |
|---|---:|---:|---|
| `d_seg` | **0.00010338677300347** | 0.00010345 | **0.06 %** |
| `d_pose` | **0.0000045867583717** | 4.59e-06 | **0.07 %** |

It also reproduces `ddm_ntb2`'s *independently written* control on the move-44 tree — d_seg
`0.00010338677` (identical to 11 figures) and d_pose `4.58673e-06` vs `4.58676e-06` — which is the
expected result, because I verified the `semantic` member is **byte-identical** between move 44 and
move 48 (both `9c0b579c…` after brotli, 31,451 B) and pc3's move-45 carrier refit is lossless on the
decoded coefficients. Two producers, two authors, one number.

### 3.2 The four rows

| treatment | `d_seg` | vs control | `d_pose` | vs control | pose leg Δ (S) | Δ as multiples of the −2e-5 bar |
|---|---:|---:|---:|---:|---:|---:|
| **control** (shipped tokens) | 1.0338677e-04 | 1.000× | 4.586758e-06 | 1.00× | — | — |
| **gt_partition** (GT argmax as the token plane) | 2.9209561e-04 | **2.825×** | 5.304407e-03 | **1156.5×** | +0.223540 | 11,177× |
| **noise_p05** (iid, 0.5 LSB camera RMS) | 1.1790805e-04 | 1.140× | 3.381317e-05 | **7.37×** | +0.011616 | 581× |
| **smooth_p05** (24×32 bicubic, 0.5 LSB camera RMS) | 1.2196011e-04 | 1.180× | 6.553723e-04 | **142.88×** | +0.074183 | 3,709× |

No-op detector, per chunk: the two perturbations move **31.77 %** (noise) and **31.47 %** (smooth) of
camera samples, so they are matched in magnitude AND in how much they actually change — neither is a
rounding artefact. `gt_partition` changes **4.32 %** of camera samples relative to the control render.

### 3.3 What the render floor says — and the charter framing it corrects

**Rendering the TRUE partition is 2.825× WORSE than rendering the shipped field.** That inverts the
intuition the charter carries. The shipped token field is not a lossy approximation of GT that the
renderer copes with; it is an optimized **pre-image** that already absorbs this renderer's bias.
Eight moves of joint-admitted token editing put it there, and the composition `(θ*, T*)` is 2.8×
better on seg than `(θ*, GT)`.

So for this vehicle the render floor at a correct partition is **d_seg 2.9210e-04**, and it is an
*upper* bound on the renderer's intrinsic error rather than a floor beneath the shipped row. Read
against `ddm_obx2`'s pre-registered reactivation gate of **4.0e-04** it **PASSES**, and it is
**7.9× below** the QBF1 born object's 2.3100e-03. Handed forward: by obx2's own criterion 2, the
edge-local implicit correction lattice is reactivatable on *this* object's render floor — obx2
refused its own object for a floor 5.77× over the ceiling; this one is under it.

### 3.4 obx2's smooth-vs-noise law INVERTS on this object — measured, both directions

At matched camera-plane RMS and matched changed-sample fraction:

> **smooth error costs 19.38× MORE Pose than noise** on the pointer's object
> (6.553723e-04 / 3.381317e-05).

`ddm_obx2` measured the opposite on the QBF1 born vehicle: *"Independent noise does 9.40× the Pose
damage at matched magnitude"*, and exported it as a successor constraint — *"its render error must be
SMOOTH."* **On this object that constraint is backwards**, by a factor of 19.4 in the other
direction; composed, the two objects disagree by 182×.

Two candidate mechanisms, offered as HYPOTHESES and not as findings:

1. **Amplitude.** obx2 measured at ~4.5 LSB RMS; this probe is at 0.5 LSB, where the receiver's
   `clamp(0,255).round()` is the dominant nonlinearity — a smooth half-LSB field flips whole
   contiguous regions by exactly ±1, while an iid one flips ~32 % of samples incoherently.
2. **The object.** This vehicle's frame 0 is **not a rendered scene** — it is a pure 12-coefficient
   smooth carrier field (`cpr1/inflate.py:336-349`). PoseNet reads the pair, so a *smooth* change to
   frame 1 looks exactly like a change of carrier state, which is the one thing PoseNet is built to
   report. Noise lies outside that 12-dimensional smooth span.

Hypothesis 2 makes a falsifiable prediction the re-solve arms in §3.5 test directly: if smooth error
is "carrier-shaped," the carrier should be able to *undo* it.

**Either way, one thing is settled and portable: neither object's spectrum law may be quoted on the
other.** obx2 was right to say it measured on its own vehicle; this is the receipt that it does not
travel.

---

## 3A. MAIN's init-drop diff, run on the renderer — CONFIRMED, and it re-reads ft1's negative

MAIN's finding (2026-09-11, measured on the HPAC prior by `ddm_dpi1`): *an EMA "init" cut from a
checkpoint silently drops non-weight state*, and `load_state_dict(strict=False)` then turns every
dropped key into a **default** that nobody records as a delta. MAIN asked me to diff the renderer's
warm-start init against its source and against the shipped deployed state before any refit. I did.

**Object:** `/Volumes/VertigoDataTier/pact/ddm_ft1_shipped_renderer_aligned_finetune/retained/init_shipped_semantic_renderer.pt`,
278,693 B, sha `460aa1c5dc365de1…` — the only renderer warm-start init on either tier, written by
`experiments/ddm_ft1_identity_gate_and_caches.py::write_init_checkpoint`.

| key the deployed object needs | in the init? | what the trainer gets instead |
|---|---|---|
| the 38 weight tensors | **YES** — I verified `init.state_dict == deployed.state_dict` exactly | — |
| **per-tensor bit-depth table** (16 entries; `frame_embed.weight` **3**, `blocks.0.film.weight` **3**, the other 14 **4**) | **NO** | the scalar `quant_bits: 4` — a **uniform** grid |
| **`keep_percent = 1`** and the FiLM **row-prune mask** (surviving rows measured: `blocks.1` [11, 13], `blocks.2` [34, 119], `blocks.3` [30, 189] — 2 of 192 each) | **NO** | **no prune at all**: 190 rows per tensor that will be zeroed at pack time are trained as if they ship |
| the per-axis fp16 **scale rule** and its `5.96e-08` floor | **NO** | re-derived by whatever the trainer's own fake-quant does |
| binding to the CURRENT archive | **NO** — `provenance.archive_sha256` is `cbb8d928…`, not move 48's `d830edd3…` | the weights happen to still match (the member never moved), so it is stale in BINDING, not in value — and nothing in the checkpoint would catch it if the member ever did move |

Top-level keys the init actually carries: `{schema, state_dict, architecture_config, quant_bits,
provenance}`. Three named non-weight objects are dropped, and each becomes a silent default.

**This is not a hypothetical: it is the measured cause of `ddm_ft1`'s +31.23 %.** `ddm_rw1` had
already quantified the consequence without naming it as a dropped input — *"ft1's trainer applied a
**uniform int4 fake-quant with no row prune**, while the deployed SM3R section uses per-tensor depths
`{3, 4}` and keeps **2 of 192 rows** in each of `blocks.{1,2,3}.film.weight`. The trained object and
the realized object differed by `2.32e-3` max-abs."*

**So ft1 did not refit the shipped renderer. It refit a different object** — uniform-4-bit,
unpruned — and the +31.23 % excursion is that object's, not this one's. As a prior negative against
the chartered Half 2, ft1 is therefore **weaker than the charter treats it**: its scope is
"seg-only fine-tune of a *dequantized-and-re-uniform-quantized* variant," not "refit of the deployed
renderer."

**The cure is already built and was default-OFF** — the orphaned-signal class exactly.
`src/tac/pr130_lift/train_semantic_quantized_resumable.py` carries `--weight-qat-q3q4`
(*"F2: train through the exact mp2 mixed q3/q4 weight grid instead of uniform `--bits`"*) and
`--film-row-dropout` / `--film-row-dropout-protect-top` (F3, using the packer's own descending-norm
row order). **ft1 set neither.** Any Half 2 must set both, warm-start from a **restored** init that
carries the depth table and `keep_percent`, and record which state it ran from.

One precision, so the fix is not overstated: the row-prune *mask* is chosen by the encoder at pack
time from the descending row norm, so it is re-derivable rather than stored. That does not make it
safe to drop — a trainer that does not know 190 of 192 rows will be zeroed spends capacity on rows
that never ship, which is the same realized-vs-trained divergence by a different route. The
**depth table** and **`keep_percent`**, by contrast, are literal arguments of
`pack_prune_mixed_candidate` and are genuinely stored inputs.

**This does not change the gate decision below**; it strengthens leg 1 (the object was never fit to
what ships) and weakens one of the prior negatives standing against leg 2.

---

## 3B. THE ADVERSARIAL TEST — can the carrier ABSORB a render change? (n600, all three arms)

I wrote §4 first, refusing the refit on `ddm_pr1`'s post-re-solve coupling `k_post = 13.82`. Then I
attacked it, because `k` is a property of a DIRECTION and pr1 measured one candidate. The attack
succeeded and §4 is now wrong. This section is the measurement; §4 is rewritten beneath it.

`experiments/ddm_ren1_resolve_absorption.py` drives the **canonical, unforked**
`up2.solve_pair_realized` — uncapped greedy descent on the REALIZED objective, every candidate
rendered through the exact receiver path and scored by the frozen CPU PoseNet — through a frame-1
shim, on **all 600 pairs**, for three arms.

### 3B.1 The control, which is what makes the rest readable

| | value |
|---|---|
| start `d_pose` | **4.586845e-06** — the step-0 probe's own control to **0.0019 %**, and the pointer's 4.59e-06 |
| final `d_pose` | **4.586160826e-06** |
| pairs improved | **2 of 600**; **2** changed coordinates; **Δ0 bytes** |

**The shipped carrier is converged**: this solver, run uncapped on the pointer's own bytes, finds a
strict local optimum on 598 of 600 pairs and buys 1.00015× on the other two. So the solver does not
manufacture gains, and any absorption below is real work against the perturbation. (The pricer also
passes its own anchor — `control_reproduces_shipped_payload: true`, 50,270 shipped Rice bits — before
returning any delta.) The 2-pair win is worth **−4.4e-07 S at 0 bytes**: real, free, and **0.022× the
admit bar**. Not a candidate; recorded so nobody re-finds it.

### 3B.2 The two spectra, at n600

Baseline for every `k_post` is the control's own re-solved 4.586160826e-06, on the same 600 pairs.

| arm | `d_pose` start | `d_pose` final | recovery | final / control | `k_pre` | **`k_post`** | Δ carrier bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 4.586845e-06 | 4.586161e-06 | 1.00× | 1.00000 | — | — | **0** |
| noise_p05 | 3.381081e-05 | **4.699366e-06** | 7.19× | **1.0247** | 2.0125 | **0.00780** | **+2** |
| smooth_p05 | 6.553703e-04 | **4.938115e-06** | **132.72×** | **1.0767** | 35.0386 | **0.01895** | **+4** |

`k_payable` at move 48 = **0.135451**. **Both arms land INSIDE it** — noise by **17.4×**, smooth by
**7.1×**. `ddm_pr1`'s 13.82 is **102× outside**.

In score units the re-solve removes **140.6×** of the noise arm's pose leg (+0.011616 → +8.263e-05 S)
and **291.4×** of the smooth arm's (+0.074183 → +2.546e-04 S).

And the re-solve is **nearly free in bytes**: 824 changed int12 coordinates cost **+2 B**, 1,209 cost
**+4 B**, priced by the carrier's own CPR1 Rice pricer against the shipped predictor. `ddm_pr1`
measured **+125 B** for its candidate.

### 3B.3 The law this replaces the coupling headline with

> **The carrier's pose absorption is a CAPACITY, not a ratio.** Inside capacity the 12-coefficient
> int12 carrier restores `d_pose` to within 2.5–7.7 % of the shipped level for ~2–4 bytes, and
> `k_post` falls an order of magnitude INSIDE the payable ceiling. Outside capacity it saturates.

Leading hypothesis for the pr1 discrepancy, offered as a hypothesis: pr1's candidate had a
pre-re-solve `d_pose` around 1.55e-02 — **~24× larger** than my smooth arm — and pr1 named the
saturation itself: *"Residue = representation limit of the int12 carrier (100 % of pairs want a GN
step beyond ±2; 9.67 % beyond the lattice)."* My perturbations sit inside the lattice's reach; pr1's
candidate did not. Two amplitudes, two regimes, one carrier.

**Honest limits on this section, stated before the conclusion is used.**

1. My treatments **damage** seg (+14 %, +18 %); a refit must **buy** it. Nothing here shows that a
   seg-BUYING renderer change has the same absorbability. It shows the pose term is not the wall it
   was believed to be at this amplitude.
2. The perturbations are a proxy for "a refit moved the render by half an LSB," not a refit.
3. `all_converged` is reported `true` by the producer, but at `--max-passes 0` that field is
   **vacuous by construction** (`not (0 and …)`). The real convergence proof is the loop's own exit
   condition — no improving lattice neighbour exists — which is what ran. Recorded rather than
   patched, because three n600 runs have this producer's sha pinned in their `INPUTS.json`.
4. The re-solve producer calls `up2.enable_posenet_gradients()` and the step-0 probe does not; I
   measured the resulting forward difference at **0.17 % mean / 2.3 % max** per pair. Every number
   above is control-relative within one producer, so it does not ride on that gap.

---

## 4. THE GATE — it PASSES, and Half 2 is now specified rather than refused

**My §4 draft refused the refit on pose. The n600 measurement in §3B falsified that draft.** What
follows is the corrected arithmetic.

### 4.1 The payable coupling ceiling, re-derived at move 48

The score is `S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489`. At the pointer's own d_pose:

* `∂S/∂d_seg = 100`
* `∂S/∂d_pose = 5/√(10·d_pose) = 5/√(4.59e-05) = **738.02**`

A renderer weight change that buys `δ` of d_seg and costs `k·δ` of d_pose after the terminal carrier
re-solve is payable only if `100 > 738.02·k`, i.e.

> **k_payable = 0.135497 at move 48.**

For a finite 25 % seg cut clearing the −2e-5 bar the ceiling is 0.1598; the marginal figure is the
tighter one and the one I use.

`ddm_pr1` measured **13.82** on this axis at n600 — **102× outside**. §3B measures **0.00780**
(noise) and **0.01895** (smooth) at n600 on the same axis and the same solver — **17.4× and 7.1×
INSIDE**. The ceiling is not the wall; the amplitude regime is.

### 4.1b What a refit must actually clear, priced end to end

Composing the measured post-re-solve pose residue with the measured carrier byte delta:

| if the refit's render change looks like… | post-re-solve pose+byte residue | as multiples of the −2e-5 bar | required `d_seg` cut |
|---|---:|---:|---:|
| **noise** at 0.5 LSB camera RMS | **+8.3963e-05 S** | **4.20×** | **1.006 %** |
| **smooth** at 0.5 LSB camera RMS | **+2.5727e-04 S** | **12.86×** | **2.682 %** |

So the refit's target is a **1.0 %–2.7 % cut in `d_seg`**, depending on the spectrum of its own
render error, **plus** whatever the semantic member's re-encode costs (§4.4). That is an ordinary
engineering target. It is not the 102× cliff the campaign has been carrying.

### 4.2 Where the wall actually is — the smallest realizable action

`ddm_rw1`: *one int4 code step* in `head` or `blocks.3` **already moves 240–455 argmax cells**. The
admit bar is `2e-5 / (100/117,964,800)` = **23.6 repaired cells**. So the smallest action the shipped
format can express is **10–19× coarser than the bar**, and it moves the render of all 600 frames at
once. There is no per-pair admission lever on a weight change — `ddm_ft1` says so and `ddm_fe1`
confirmed it by having to go to `frame_embed` to get one.

That coarseness cuts both ways and should not be read as a refusal on its own: 240–455 cells is
**10–19× the admit bar**, so a single code step that moved cells in the RIGHT direction would clear
the bar by an order of magnitude. `ddm_rw1` searched 150 such steps and found 0.0 %, which is the
real negative here — but it searched with the depth table and the prune mask absent from the loop
(§3A), on the move-33/35 fields, at FORMULATION scope.

### 4.3 ntb2's named prize, re-priced

Take `ddm_ntb2`'s own open door: recover the `+4.92e-05` S that the 3-bit `blocks.0.film` choice
already cost (SD1 measured that cell as a LOSS; it was taken on a bytes argument). That is
`delta = 4.92e-07` of d_seg, about **58 argmax cells**, a **0.48 %** cut.

Against §4.1b that prize alone does **not** clear: it is under the 1.006 % a noise-spectrum refit
must buy. It clears only if the refit's render change is smaller than 0.5 LSB RMS (the residue scales
down with it) or if it is composed with other seg gains. Stated plainly so nobody quotes the 4.92e-05
as a standalone win.

### 4.4 The term that is still unpriced — the semantic member's own re-encode

The SM3R **body** is size-invariant by construction: `pack_prune_mixed_candidate` lays the payload
out from shapes, `keep_percent` and the depth table, never from the values, so any refit exports at
exactly **36,130 B** (regression-tested upstream as
`test_sm3r_export_size_is_independent_of_the_weight_values`). But the **member** is
`brotli(CK2(SM1S(body)))` and that is value-dependent, and the 24 shared int8 mixer weights were fit
to the codes that exist today.

`ddm_fe1` measured this container's rate law, at N ≤ 200 changed codes: a flat **~+70 B**
break fee at the shipped shape, recoverable to **+1.3 B (N=1)** / **+28.0 B (N=72)** by searching
`q ∈ {9,10,11} × lgwin ∈ {16,18,20,22,24} × {ck2, plain}`, with the shipped shape pinned by BYTE
identity at `(ck2, 11, 24)`. A refit changes **all 59,376 codes**, far outside that table, so its
member size is **not predictable from fe1 and must be measured by a real encode**. It may also
shrink: a refit is free to prefer lower-entropy codes, and the mixer weights can be re-fit as counted
data without touching the receiver.

**This is the one term of the refit's price that this arm did not measure.** It is cheap to
measure — one `pack_prune_mixed_candidate` + `sm1.encode` + `brotli` over a candidate state — and it
should be measured on the FIRST checkpoint, not at the end.

### 4.5 The counter-arguments to §3B, and what survives them

Three prior measurements pointed the other way. Each is now placed rather than dismissed.

1. **`ddm_w96b`** fired an aligned renderer gate at two seeds with **pose in the loop** and landed
   *"pose 185–204× the incumbent."* Scope: the W96 witness family, n60 seeds, a different renderer
   from a different start — and, decisively for the comparison, **no terminal carrier re-solve in the
   quoted number.** It bounds a pre-re-solve regime, which §3B shows is 141–291× away from the
   post-re-solve one.
2. **`ddm_pz1`**: PoseNet and SegNet make the **identical** `interpolate` call to the same
   `segnet_model_input_size` (`upstream/modules.py:73` vs `:109`), so "seg-visible but
   pose-invisible" cannot rest on a resize difference. That still stands, and §3B does not contradict
   it: the cure here is not invisibility to PoseNet, it is **compensation on frame 0**.
3. **`ddm_rf1`**: 97.59 % of a structural renderer refusal was pose — again pre-re-solve, on an
   un-retrained FiLM amortization, at INSTANCE scope.

**What survives all three, and is now the binding term: SEG.** `ddm_rw1` searched the shipped weights
from five directions — a 3,000-step joint MPS run, a random-direction control, the exact-field
gradient in both ranking directions, a **discrete realized code search over 150 evaluations (0.0 %)**,
and a **per-row fp16 scale search over 91+ evaluations (0.0 %)** — and accepted nothing, closing at
FORMULATION scope on grid resolution: *"one int4 code step in `head`/`blocks.3` already moves 240–455
argmax cells, so the fold-back is limited by the grid's resolution, not by the search."* `ddm_pr1`
said the same thing from its own direction: *"the axis closes on SEG, not on pose."* **They were
right about the axis and the campaign attributed the refusal to the wrong term.**

### 4.6 Why Half 2 did not fire in this arm, stated as a decision and not an excuse

The gate PASSES. Half 2 did not run here for three reasons, all structural:

1. **The charter binds me to "BOTH scorers in the loop, never seg-only," and the launch-admissible
   trainer has no pose term.** `src/tac/pr130_lift/train_semantic_quantized_resumable.py` has
   `--init` warm start, canonical `tac.training.EMA`, exact-R eval-roundtrip, **`--weight-qat-q3q4`**,
   `--film-row-dropout`, stage checkpoints, `--resume-from` and a deployed pack/parse argmax-parity
   gate — everything except PoseNet in the loss. §3B is the argument that the *terminal* re-solve is
   the right place for pose on this vehicle, which would satisfy the rule's intent; but rewriting a
   binding charter's loss shape is MAIN's call, not an arm's.
2. **The amplitude bound that makes §3B apply is not built.** Nothing in the trainer constrains the
   render delta to stay inside the carrier's capacity, and that constraint is the whole content of
   the finding. A refit that wanders outside it re-enters pr1's saturated regime.
3. **Time.** The three n600 arms took 5.1 ks, 5.1 ks and 6.2 ks of the arm's clock. A QAT refit plus
   byte-close plus n600 scoring could not have been finished, and an unfinished burn produces no
   exact row.

---

## 5. VERDICT

**The pointer did not move. This arm produced no candidate and no exact row.** What it produced is a
measured correction to a prior that four arms had been reading the wrong way, and a Half 2 that is now
specified by numbers instead of guessed.

**Half 1 gate, leg 1: PASSES, emphatically.** The renderer is a banked third-party PR130 artifact,
trained seg-only on GT-oracle tokens, no EMA, contested GT lineage, through a **uniform int4**
quantizer that is not the mixed-{3,4}-plus-99 %-row-prune one it ships in — and the warm-start init
built from it **drops the depth table, `keep_percent` and the prune mask** (§3A), which is the
measured reason `ddm_ft1` refit a different object.

**Half 1 gate, leg 2: PASSES.** The measured floor and sensitivity leave room:

* the render floor at a correct partition is `d_seg` **2.9210e-04**, which clears obx2's 4.0e-04
  reactivation gate and is 7.9× below the QBF1 object's;
* the carrier restores `d_pose` to within **2.5 %** (noise) / **7.7 %** (smooth) of shipped for
  **2–4 bytes**, putting `k_post` **7–17× inside** the payable ceiling;
* the refit's target is therefore a **1.0 %–2.7 %** `d_seg` cut, not a 102× cliff.

**The binding term is SEG, and it is not closed — but it is also not established.** `ddm_rw1`'s 241
evaluations found 0.0 % at FORMULATION scope, on an AdamW-over-a-piecewise-constant-forward
formulation, on the move-33/35 fields, with the token plane held fixed and **without** the depth
table or the prune mask in the loop. `ddm_ft1`'s +31.23 % was measured on an object that is not the
one that ships. Neither is a verdict on a quantization-aware refit of the deployed object, and
`ddm_ntb2` says so in its own words: **"A quantization-aware refit is NOT closed."**

### What the next unit should run, specified

A refit-in-place, seg objective with a **terminal resolved pose** (not a pose loss), on the CURRENT
field `a92e7d90…`:

1. **Restore the init** before anything else: carry the per-tensor depth table `{frame_embed: 3,
   blocks.0.film: 3, rest: 4}` and `keep_percent = 1` as inputs, and re-bind `provenance.archive_sha256`
   to move 48's `d830edd3…`. Record the diff and state which state the run used (§3A, MAIN's law).
2. **Set the two default-off levers ft1 did not**: `--weight-qat-q3q4` and `--film-row-dropout
   --film-row-dropout-protect-top`. Without them the trained object is not the realized one.
3. **Bound the render delta** to ≤ 0.5 LSB camera RMS against the shipped frame 1, measured, and
   prefer the NOISE end of the spectrum — 19.4× cheaper on pose here, and the opposite of what obx2
   exported from its own vehicle (§3.4).
4. **Price the member's re-encode on the first checkpoint** (§4.4), not at the end.
5. **Clear the bar**: `d_seg` cut ≥ **1.006 %** (noise-spectrum render change) or ≥ **2.682 %**
   (smooth), plus the member delta, with the terminal re-solve run and its 2–4 B carried.
6. Warm start, EMA shadow, per-stage checkpoints, `--resume-from`, eval_roundtrip — all already in
   the trainer.

**The zero-pose-exposure alternatives remain ranked above it**, because they change no frame and
therefore pay none of §4.1b: `ddm_hpr1`'s rung 1 — refit the tc1 tail mixer's 35 int8 weights on the
current field, **60 B of state pricing 118,511 B of stream** — and growing the HPAC prior, the
section whose refit produced moves 47 and 48.

## 6. Corrections and receipts owed upward

1. **Neither 170–220 nor 13.82 is "the coupling" — each belongs to an amplitude regime.**
   §"Prior negatives accounted" of the ren1 charter, `ddm_rbf1`'s charter line 43, and `ddm_gs4`:26
   carry the **pre**-re-solve 170–220. `ddm_pr1` refined it to a **post**-re-solve 13.82. §3B
   measures **0.0078 / 0.0190** post-re-solve at an amplitude inside the carrier's capacity. All
   three are right about their own regime and none is the renderer's coupling. A successor that
   quotes any of them without naming the amplitude will price the wrong wall — and, on the evidence
   here, will refuse a door that is open.
2. **`17e0fd0b…` is 36,130 B, not 29,862 B.** Member `786950a5…` is the 29,862 B object. Verified
   here by decoding the shipped archive.
3. **The renderer is not ours.** Any memo that reads the `semantic` member as our trained artifact
   should say "banked PR130 intake artifact, trained by others, provenance reconstructed by
   `ddm_pr130_reproduce_20260809/RR2_SEMANTIC_LEG_AUDIT.md`."
4. **`rf1`'s "the renderer axis cannot reach 0.12 even if the renderer ceases to exist" has expired
   at move 48.** Re-derived here: deleting the 29,862 B member at zero added distortion gives
   S = 0.010345 + 0.006775 + 0.099369 = **0.11649**, which is below 0.12. The bound was true at its
   own archive size; it is a binding number that expired
   [[binding-instruction-numbers-expire-and-nobody-rederives-them]]. It remains a fantasy bound — you
   cannot delete the renderer and keep the frames — but it should not be quoted as current.

---

## 7. Custody

All writes under `/Volumes/VertigoDataTier/pact/ddm_ren1/`. The promoted move-48 tree was read only;
no sister arm's directory was touched; no Modal call, no contest evaluation, no candidate archive, no
seal. Every payload retained with its sha256:

| artifact | bytes | sha256 (16) |
|---|---:|---|
| `step0_probe/floor_and_sensitivity/INPUTS.json` | 2,060 | `be4d096a01958e84…` |
| `step0_probe/floor_and_sensitivity/PERTURBATION.json` | 301 | `3e0fad46b88c779e…` |
| `step0_probe/floor_and_sensitivity/RESULT.json` (per-pair d_seg/d_pose × 4 treatments, per-chunk frame sha + changed fractions) | 275,781 | `6d0de07fce1c76b7…` |
| `step0_probe/floor_and_sensitivity/planes.npz` (full SegNet argmax plane + pose vectors, all 4 treatments) | 2,765,766 | `a6e3c138c3332b3e…` |
| `step0_probe/floor_and_sensitivity/checkpoints/LATEST.{json,npz}` (resumable stage state) | 272,552 / 471,976,534 | `059f9b0b80b0a600…` / `f03aac4aa019a0ba…` |
| `resolve_absorption/control/{INPUTS,RESULT}.json` + `rows.jsonl` (600 solved pairs) | 1,743 / 2,210 / 169,478 | `4451d5f60979cbb3…` / `989101cbf7b37d8a…` / `297df0ee24dd7752…` |
| `resolve_absorption/noise_p05/{INPUTS,RESULT}.json` + `rows.jsonl` | 1,745 / 2,216 / 198,184 | `beed33474f66a674…` / `01a6037d2a5266d8…` / `67426e20e49d6862…` |
| `resolve_absorption/smooth_p05/{INPUTS,RESULT}.json` + `rows.jsonl` | 1,746 / 2,218 / 209,420 | `501785ce40f73bed…` / `abee3ebfd74f1480…` / `1850fd2c6201b088…` |
| `resolve_absorption/noise_p05/rows.smoke2pairs_4threads.{jsonl,json}` — the 2-pair 4-thread sizing smoke, **kept** rather than deleted when the arm was restarted homogeneously at 5 threads | 1,314 / 2,233 | `15fd5d5b5bb0ec35…` / `bcb4c9d3b5a8f22f…` |
| the three `launch_*/launch_manifest.json` + `run.log` | — | — |

The ~1.8 GB camera rasters were **not** retained: they are exactly rebuildable from the pinned
archive plus the pinned token field by the producers, whose shas are recorded in each `INPUTS.json`,
and the tier holds a 40 GiB reserve that both producers refuse to cross.

Code landed: `experiments/ddm_ren1_step0_probe.py`, `experiments/ddm_ren1_resolve_absorption.py`,
`src/tac/tests/test_ddm_ren1_step0_probe.py` (18 tests, all passing) at commit `2a0dc8f24`.
Lane pre-registered as `lane_ddm_ren1_renderer_provenance_and_refit_in_place_20260911` (L0).

The frontier is unchanged: **composition S 0.13638261682704697 @ 179,111 B `[contest-CUDA T4 n600]`
(move 48)**.
