# ddm_ua3 — `upstream/submissions/` + dependency-closure audit

Date: 2026-07-31 · Agent: ddm_ua3 · Cost: $0 · `[macOS-CPU advisory]` · `score_claim=false`
Pointer `0.1910828242 [contest-CPU]` UNMOVED. READ-ONLY: nothing under `upstream/` was
created, modified, moved, or executed. The three shipped binaries were hashed, never run.

Deliverable form per operator directive: **every result is a placed point on a named
surface** — surface, coordinates, the level set where the language flips, what moves it,
where we stand, which way it falls. No verdicts. No ranked "which is best" table. The
shipped submissions are LESSONS and CROSS-CHECKS, never a vehicle (no-old-lineage ban
applies here identically).

---

## 0. Denominator and scope

| Category | Count | Treatment |
|---|---:|---|
| Files under `upstream/submissions/` | 88 | enumerated (`find`) |
| Non-binary files (`.py`/`.sh`/`.md`/`.png`) | 83 | **all sha256'd + sized** |
| Read in full | 12 | `train_ren.py`, `neural_inflate/{inflate.py,inflate.sh,compress.sh}`, `damir_bearclaw_002/roi_preprocess.py`, `damir_bearclaw_003/{README.md,seg_middle_preprocess.py}`, `v4_qp_aq2_roi/generate_qpmap.py`, `av1_sharp1_adaptive/inflate.py`, `svtav1_45pct_unsharp/inflate.py`, `svtav1_av1grain_10bit/inflate.py` (to L88), `damir_bearclaw_003/inflate.py` (partial) |
| Read via structural diff | 3 | `av1_crf31_bicubic/preprocess.py`, `roi_v2/preprocess.py`, `av1_roi_lanczos_unsharp/preprocess.py` |
| Swept by targeted regex (encode params, decode operators) | 19 `compress.sh` + 19 `inflate.py` | every one |
| PNGs analysed numerically | 2 | pixel-level, PIL+numpy |
| Binaries | 3 distinct (5 instances) | **sha256 only — NOT executed, NOT disassembled** |
| `__pycache__/*.pyc` | 1 | not decompiled |

Unreachable / not done, and why: the two shipped binaries' internals (executing or
disassembling a 24.5 MB ffmpeg build is outside a $0 read-only audit and they are
provably decode-side-unused, §5); `upstream/.venv` wheel provenance (ddm_ua2 owns the
`pyproject.toml`/`uv.lock` defence reading — I report only *what imports*).

**Correction to my own prompt's premise:** MAIN listed "18 reference submissions." The
directory holds **19** submission subdirectories. Two of them —
`svtav1_45pct_unsharp` and `svtav1_cheetah` — are **byte-identical across all three
files** (`compress.sh` `bfbde10a…`, `inflate.py` `7858f285…`, `inflate.sh` `83ba8770…`),
so the count of *distinct* submissions is **18**. MAIN's 18 is right by accident, via a
different arithmetic. MEASURED.

---

## 1. MAIN's negative-existence claim — CHECKED, and it holds

MAIN wrote "as far as MAIN can tell … we have not systematically mined the shipped
submissions" and explicitly flagged it unverified. Today's #1 error class is exactly this,
so I tested it rather than inheriting it.

**Search performed:** `grep -rlE` for six distinctive shipped names
(`svtav1_cheetah|av1_crf31_bicubic|roi_gop300_c34|svtav1_spline_fg22|baseline_fast|h265_g16_512x384`)
plus `seg_middle_preprocess|generate_qpmap` over `.omx/**`, `docs/**`, `src/**`,
`tools/**`, `reports/**`, `experiments/**`, `reverse_engineering/**`, `.ralph/**`,
`.claude/**`, and the entire `~/.claude/projects/-Users-adpena-Projects-pact/memory/**`.
**Result: zero hits.** MEASURED, scope as stated. Not searched: git object bodies beyond
`git log -- 'upstream/submissions/*'` (which is empty — see below), and conversation
transcripts.

**Two false leads I ran down and killed:**
- `.omx/research/**` hits on `train_ren` are substring matches inside
  `train_render*`/`train_renderer*`. Not this file.
- `codex_findings_nerv_source_refresh_…` hits on `upstream/submissions/hnerv_muon` — that
  is the **contest-PR intake path convention** (a PR's own submission dir is named
  `submissions/<name>/`). Namespace collision, different object. The #412/#413/#414/#728
  intakes covered *contest PRs*; this directory is the *shipped reference set*. The
  distinction MAIN asked me to establish is **real**.

**Why it was invisible:** `.gitignore:256` ignores `upstream/` wholesale
(`git check-ignore -v` confirms; `git ls-files upstream/submissions` is empty). The pinned
snapshot is a local materialisation, never in our object store — so every tool we have
that reasons over tracked files, every diff-driven review, and every commit-scoped gate
has been **structurally blind** to these 88 files since day one. That is an *apparatus*
finding, not a diligence finding, and it generalises: **any audit surface keyed on
git-tracked paths cannot see the pinned snapshot.**

---

## 2. SURFACE: rate — the encode operating point

**Coordinates** (MEASURED, swept across all 19 `compress.sh`):

| Axis | Value | Spread across the family |
|---|---|---|
| Downscale | `scale=trunc(iw*0.45/2)*2` | **16 of 19 identical at 0.45**; `damir_bearclaw_003` uses 0.45×1.00 (width-only); `h265_g16_512x384` uses literal `512:384`; `no_compress` none |
| Codec | `libsvtav1 -preset 0` | 14 of 19; `libx265` 2; copy/none 3 |
| CRF | 31 · 32 · 33 · 34 | 33 is modal (9 of 19) |
| Film grain | `film-grain=22` | 12 of 19; `=0` in bearclaw_003 |
| GOP | `keyint=180` | 11 of 19; `300` (roi_gop300), `1` (baseline_fast) |
| Resampler | `flags=lanczos` | 17 of 19; `spline` in svtav1_spline_fg22 |

**The level set where the language flips.** `0.45 × (1164, 874) = (523.8, 393.3)` →
even-truncated → **522 × 392**, which is **1.0195× × 1.0208×** the SegNet input
`(512, 384)`. MEASURED. The universal 0.45 is not a tuned constant — it is *the smallest
scale that still stores everything the scorer can see*, to within 2%. `h265_g16_512x384`
makes this explicit by storing at exactly 512×384. **The entire shipped family has
converged on "store at scorer resolution, and vary only CRF."**

**Where we stand.** Our own-vehicle exact-protocol line is at **0.9639878** (07-31, v4d
refine stack, 359,750 B), with rate 0.239 of that. Their axis is a 1-D CRF sweep on a
frozen 522×392 lattice; ours is a description-carrier whose rate is a *composed*
KKT-waterfill quantity (`opportunity_pools_non_additive`, #766 waterfill rung + the
granularity re-race, which our own ledger sizes at ~gap-sized). We are not on their line
and cannot be moved along it.

**Which way it falls.** This surface contributes **one** transferable coordinate, and it
is a constraint rather than a lever: any carrier we design that stores *more than
522×392-worth* of spatial detail is paying for information the frozen scorer provably
discards at its own resize. That is corroboration of the frozen-scorer exact
factorization (`A_seg ≡ A_pose → (512,384)`) from an independent, never-consulted source.
It does not move rate for us; it forecloses a class of waste.

---

## 3. SURFACE: spatial prior — where the organizers think the scorers look

### 3a. The corridor polygon is ONE prior, copied — not five derivations

`preprocess.py` / `roi_preprocess.py` appear as 6 files. By sha256 they are **3 distinct
implementations**, and `roi_preprocess.py` (5235 B) vs `neural_inflate/preprocess.py`
(5234 B) differ by **exactly one removed blank line at L158** — functionally identical.
MEASURED, correcting MAIN's file-count framing ("×3 at 5,235 / ×3 at 5,234": actually
5235 ×3, 5234 ×2, 5131 ×1, 4742 ×1).

More decisive: the corridor polygon itself is **verbatim identical in all variants**:

```
(  0,  299, [(0.14, 0.52), (0.82, 0.48), (0.98, 1.00), (0.05, 1.00)])
(300,  599, [(0.10, 0.50), (0.76, 0.47), (0.92, 1.00), (0.00, 1.00)])
(600,  899, [(0.18, 0.50), (0.84, 0.47), (0.98, 1.00), (0.06, 1.00)])
(900, 1199, [(0.22, 0.52), (0.90, 0.49), (1.00, 1.00), (0.10, 1.00)])
```
(`av1_crf31_bicubic/preprocess.py:29-32`; `roi_v2/preprocess.py:64-67`;
`av1_roi_lanczos_unsharp/preprocess.py:73-76`; `damir_bearclaw_002/roi_preprocess.py:73-76`)

So the **spatial-prior sample size is n=1**, hand-authored, not 6 independent votes. Its
own docstring says so: *"the polygons were tuned by eye on each segment"*
(`av1_crf31_bicubic/preprocess.py:26-28`). Their stated rationale
(`av1_crf31_bicubic/preprocess.py:3-9`): *"PoseNet and SegNet both downsample to 512x384
before doing anything, and the corridor where cars/lanes/traffic live occupies the
lower-center triangle … Everything else (sky, buildings, the car's own hood, passing
trees) is high-entropy content that AV1 will happily spend bits on for no scoring
benefit."*

**Internal inconsistency worth recording:** the docstring names *the car's own hood* as
waste, but the implemented polygon's bottom edge is `y = 1.00` spanning `x ∈ [0.05,0.98]`
— it **preserves ~93% of the hood**. Stated prior and implemented prior disagree. The
implemented one is the one that ran.

### 3b. Cross-check against our measured geometry — AGREEMENT on shape, DIVERGENCE on sharpness

Their corridor top edge (the horizon) sits at mean `y = 0.49375` of frame height (8 top
vertices). Our `#623 g4` MEASURED horizon flip band is **row 212 ± 4 of 384** →
`y = 0.5521`.

| | Their hand-tuned prior | Our MEASURED (`g4`, n600) | Delta |
|---|---|---|---|
| Horizon | `y = 0.49375` | row 212/384 = `y = 0.5521` | **0.0584 of height = 22.4 rows @384 = 51 rows @874** |

**AGREEMENT (corroboration):**
- They smooth **above** the horizon only; our `g4` flip-mass map names "stable
  horizon/road-edge bands, the two Lane-corridor edges, the broad Movable/Undrivable
  mid-band, and the hood rim" — all of which fall *below* their top edge and are
  preserved. Same shape, independently reached.
- `generate_qpmap.py:92-96` spends bits (`QP −5`) on blocks with `len(unique) > 1 and
  road_frac > 0.1` (multi-class blocks containing road) and saves bits (`QP +5`) on
  blocks that are `sky_frac > 0.9` (uniform interior). That is **the codim-1 separatrix
  vs flat-interior split of our unified level-set flow**, arrived at as an encoder
  heuristic. Independent corroboration of "d_seg lives on the boundary."
- `seg_middle_preprocess.py:43-48` does the same split in the *photometric* domain:
  `edge = clamp(grad_mag(Y)/40, 0, 1); out = blurred*(1−edge) + mid*edge` — preserve
  gradient, blur interior. Same operator, different chart.
- `generate_qpmap.py:53` computes SegNet labels only on `fidx % 2 == 1` (odd frames) —
  scorer-faithful (SegNet reads `x[:, -1, ...]`, the pair's last frame). Corroborates our
  frame-parity handling.

**DIVERGENCE (the informative half):**
- Our `g4` rank-1 amortisable field is **"Movable rows 174–215, Lane→Road", reach
  159,604 events, cell-space Δd_seg 0.001353, 12 bytes**. Rows 174–215 spans
  `y ∈ [0.453, 0.560]`. Their corridor top edge at `y ≈ 0.494` **cuts through it**: rows
  174–189 — the upper ~38% of our highest-reach band — fall *outside* their corridor and
  get luma-denoised + chroma-collapsed. Their hand-tuned prior smooths part of the single
  most valuable band we have measured. MEASURED on both sides; the comparison is DERIVED
  (their normalized-y vs our 384-row index, same frame geometry).
- **Concentration is the real gap.** Our `g4`: **top 5% of pixels (9,831 of 196,608) carry
  68.72% of flip mass; top 10% carry 89.87%.** Their exploitable granularity is a 4-piece
  trapezoid (feathered at radius 24–32) and, in `generate_qpmap.py`, **63 blocks per frame**
  (`h_b = ceil(392/64) = 7` × `w_b = ceil(522/64) = 9`, `generate_qpmap.py:43-44`). They
  have the right *shape* at roughly **10–20× too coarse a resolution** to reach the
  concentration that exists.

**Which way it falls.** The organizers' prior is a low-resolution projection of a field we
have already measured at pixel resolution. It supplies **zero new geometry** — but it
supplies something we did not have: an *independent, adversarial-by-construction* witness
that the shape is real and not an artifact of our own pipeline. It also marks a specific
falsifiable spot (their corridor top edge vs our Movable band) where hand-tuning loses.

### 3c. `damir_bearclaw_003` — a submission that walked our exact trade and reported back

`README.md:21` states, unprompted and in their own words:

> *"once proxy errors are within a practically acceptable band, additional optimization
> pressure should shift toward rate rather than continuing to reward uniform proxy
> fidelity."*

That is our **`gap = RATE`** position (`opus5_arm_harvest_gap_is_rate`), reached
independently. `README.md:13`: *"SegNet appears to derive most of its useful semantic
structure from the central driving corridor, while the upper and lower filler regions are
heavily position-biased"* — a qualitative statement of what our class-geometry memo holds
quantitatively (Undrivable top IoU 0.995; MyCar bottom **static** IoU 0.994).

Their implementation (`seg_middle_preprocess.py:64-68`): keep rows `[H/4, 3H/4]`, i.e.
`[96, 288]` in 384-space; synthesise top and bottom bands at decode.

**Where that lands them, on our numbers:**
- **d_seg: they were right.** Our `g4` gives hood-rim flip mass **15,646 of 4,011,236 =
  0.39%**. Discarding the bottom quarter costs almost nothing in seg. Their instinct is
  MEASURED-correct.
- **d_pose: that is where it fell over.** Their own `README.md:17` names *"near-field
  anchor regions"* as pose-sensitive — and then discards exactly those. Our 07-31 finding
  is that **pose is the largest axis on the own vehicle at ~1.24 S, exceeding seg 0.431 +
  rate 0.239 combined**. Their README concedes *"It does not [beat the best previous
  score]"* without naming the mechanism. **Our pose-axis-weight finding names it.**

**Which way it falls.** bearclaw_003 is a *measured data point on the rate-vs-pose
exchange*, contributed by someone who moved down the rate axis by discarding the near-field
anchor and paid for it in pose. It is corroboration for the axis-weight ordering we
established 07-31 — and a caution against exactly the "throw away the static hood, it's
free" move that our own 0.39% seg-flip figure would otherwise invite. The seg argument for
discarding the hood is sound; the pose argument against it dominates.

---

## 4. SURFACE: neural placement — `neural_inflate/train_ren.py`

The contest's own reference for our problem class. MEASURED from source.

**Architecture** (`train_ren.py:26-43`, mirrored at `inflate.py:12-27`):
`PixelUnshuffle(2)` → `Conv(12→32) ReLU → Conv(32→32) ReLU → Conv(32→32) ReLU →
Conv(32→12)` → `PixelShuffle(2)`; forward is `(x/255 + residual).clamp(0,1)*255`.
Last conv **zero-initialised** (`train_ren.py:37-38`) ⇒ **exact identity at init**.
Parameter count DERIVED: `3488 + 9248 + 9248 + 3468 = 25,452`.

**Placement.** It is a **counted decoder** in our three-placement taxonomy
(`neural_codec_scientist_artist_identity_hybrid_raced_placements`): a residual
post-filter applied *after* Lanczos upscale to camera size, on top of a classical AV1
codec (`inflate.py:88-93`). It is **not** a renderer and **not** a receiver.

**Objective** (`train_ren.py:85-121`) — this is the part that matters:
- `loss_pose` = MSE over `posenet_out[h.name][..., :h.out//2]` per hydra head, inference
  vs GT. Gradient flows through `posenet.preprocess_input` (**not** wrapped in `no_grad`
  on the inference branch, L94) — i.e. they solved the same differentiable-YUV6 problem
  our `eval_roundtrip` non-negotiable names.
- `loss_seg` = `KL(log_softmax(logits_inf) ‖ softmax(logits_gt))`, `batchmean`. **Soft
  distillation, not argmax.** They optimise a surrogate of the scored quantity.
- `loss_temp` = `L1(corr_a, corr_b)` — L1 between the *corrections* on consecutive frames.
  A temporal-smoothness prior on the correction field itself.
- Weights: `w_seg = clip(lp0/ls0, 0.01, 10.0)` auto-calibrated at the identity baseline
  (`train_ren.py:192-196`); `w_temp = 0.005` hardcoded.
- Adam `1e-3`, cosine→`1e-5`, grad-clip 1.0, seed 1234, split at frame 1000.

**Where we stand — three specific reads:**

1. **Their d_seg surrogate is the one our own record says is wrong.** KL on softmax
   logits is not argmax disagreement. Our L68/#205 line is exact-flip-based
   (`CE-floor d_seg 0.00496`, `CE-residual = flicker`), and our `constants_are_poison`
   discipline forbids importing `w_seg = lp0/ls0` as anything but a scale-matching
   heuristic evaluated at one sample (`train_ds[0]`, n=1). Their weight is a **borrowed
   constant** in our exact sense: derive-or-race, never adopt.

2. **They trained the post-filter *through* the frozen scorers — which is the distinction
   our ERF-collateral law turns on.** Our 07-31 `erf_collateral_law` states: post-hoc
   injection on textured renders is net-worse *even with perfect GT* (+0.30 S), because
   the ~85 px ERF re-reads the stroke; recovery must be **born in-loop**. `train_ren.py`
   is *post-hoc in the pipeline* but *in-loop in the training* — the correction is shaped
   by the same ERF that will later read it. That is not a counterexample to our law; it
   is the **precise boundary of it**, and the shipped reference sits exactly on that
   boundary. Recording it sharpens the law's statement: the law is about *injection
   without gradient through the ERF*, not about *position in the pipeline*.

3. **It never pays its rate bill.** Grep for `ren_model` across all of
   `upstream/submissions/**` (excluding `.pyc`) returns hits at **only four lines**:
   `inflate.py:61,62,63` (load candidates) and `train_ren.py:181` (save path). MEASURED.
   `train_ren.py:181` saves to `submissions/av1_roi_lanczos_unsharp/ren_model.pt` — a
   *different* submission's directory — and `neural_inflate/compress.sh` does
   `rm -rf "$ARCHIVE_DIR"` then zips only `$ARCHIVE_DIR` (which receives only `0.mkv`).
   **No `compress.sh` in the tree ever places `ren_model.*` into an archive.** As shipped,
   `neural_inflate` raises `FileNotFoundError("ren_model not found")` at
   `inflate.py:74`.

   DERIVED price had it shipped: 25,452 int8 bytes + per-tensor fp32 scales →
   `25 × 25,452 / 37,545,489 = 0.01695` score, before bz2. **The contest's own neural
   reference demonstrates the mechanism and skips the accounting.** That is the gap
   between "a neural placement is legal" and "a neural placement is priced" — and pricing
   is the whole of our rate axis.

**Which way it falls.** As a *vehicle*: nothing — banned by the no-old-lineage discipline
and dominated by our own carrier anyway. As *intake intelligence*: it establishes that the
organizers' own answer to "put a neural net in the decoder" is a **25.5 KB zero-init
residual post-filter trained through the frozen scorers**, and that they did not solve
(or attempt) the byte accounting. Our `e_p` rank-1 pose carrier is ~2 KB
MEASURED-CLOSED — an order of magnitude cheaper than their unshipped 25.5 KB — which is
a real, favourable placement of our work against the reference, and the first external
yardstick we have for it.

---

## 5. SURFACE: free decode-side operators — the zero-byte axis

**This is the most under-exploited thing in the directory.** MEASURED across all 19
`inflate.py`.

Every non-trivial `inflate.py` applies an **unsharp mask at decode**, costing **zero
archive bytes**, gated on `if H != target_h or W != target_w`:

| Submission | Kernel | Strength α | Source |
|---|---|---|---|
| `roi_v2` | 9×9 binomial /65536 | **0.27** | `inflate.py:12,29` |
| `av1_roi_lanczos_unsharp` | 9×9 binomial | **0.40** | `inflate.py:11,29` |
| `svtav1_45pct_unsharp` / `svtav1_cheetah` / `damir_bearclaw_001` / `_002` | 9×9 binomial | **0.85** | `svtav1_45pct_unsharp/inflate.py:33` |
| `svtav1_av1grain_10bit` | 5×5 Gaussian σ=1.0 | **2.0** | `inflate.py` (`amount = 2.0`) |
| `av1_sharp1_adaptive` | 9×9 binomial | **variance-adaptive 0.4 → 1.2** | `inflate.py:36-44` |

**The level set where the language flips.** `av1_sharp1_adaptive` computes
`local_var` from the *decoded* luma via two `avg_pool2d` passes and sets
`alpha_map = 0.4 + 0.8 * local_var/(local_var + 100.0)` — sharpen structure, leave flats
alone. **Content-adaptive with zero side information**, because the adaptivity is
recomputed at decode from what is already there. That is the "inflate.py is a FREE
interpreter" doctrine instantiated by the organizers themselves, and it is the
*photometric* twin of `generate_qpmap.py`'s boundary-vs-interior split.

**What it actually is, structurally.** The gate `if H != target_h or W != target_w` means
the unsharp exists solely to **partially invert the downscale→upscale low-pass**. It is an
analytic, zero-byte, approximate deconvolution of exactly the operator our `R` chain
contains (`bicubic↑384→874 → uint8-STE → bilinear↓→512×384`). Our own record already says
that low-pass is the enemy (`R_surv`: "a low-pass kills naive sine (Gibbs aliasing)").

**Where we stand.** We attack that low-pass by training *through* `R` — compensation
purchased in carrier bytes. They attack it with a **0-byte closed-form operator at
decode**. These are two coordinates on the same surface, and ours is the expensive one.
The strength spread across the shipped family is **0.27 → 2.0, a 7.4× range on a knob
that costs nothing**, plus one adaptive variant — and there is no evidence in the tree
that anyone swept it against the exact scorer (the files carry no measurements).

**Which way it falls.** This is the single most directly actionable coordinate in the
audit, and it is a *rate* move disguised as a distortion move: a free decode-side operator
that recovers part of the low-pass buys back distortion that we are currently paying
carrier bytes to recover. It sits squarely on the `#766 waterfill` rung's input side —
every score-unit of distortion an α-sweep recovers for 0 bytes changes the waterfill's
per-pool exchange rate, and our own law says pools are non-additive so it must enter the
KKT solve, not be bolted on. It is **raced, never presumed** (per the neural-placement
discipline): the honest next measurement is an α sweep (including the adaptive form)
through the exact scorer on a byte-frozen archive, where α is the only variable.

**Hazard on the same surface.** `svtav1_av1grain_10bit/inflate.py:23-58` **reimplements
`yuv420_to_rgb` locally** rather than importing `frame_utils.yuv420_to_rgb`, with its own
BT.601 limited-range constants and a 10-bit branch. Our standing rule ("GT decodes ONLY
via `frame_utils.yuv420_to_rgb`; PyAV rgb24 manufactures ~100× phantom pose") targets the
*GT* side, so this is legal here — it is the candidate decode. But it is the same class of
divergence, on the candidate side, and it means that submission's colour transform is not
provably the reference path. Recorded as a coordinate to not repeat.

---

## 6. SURFACE: runtime closure — what is importable for free

**This is the finding with the widest blast radius.** MEASURED by direct import in
`upstream/.venv/bin/python` (Python 3.11), 35 distributions installed.

**PRESENT** — torch **2.10.0** · torchvision **0.25.0** · numpy **2.3.4** · pillow
**12.0.0** · av **17.0.0** · timm **1.0.22** · safetensors **0.6.2** · einops **0.8.1** ·
segmentation_models_pytorch **0.5.0** · huggingface_hub 1.1.2 · sympy · networkx · jinja2 ·
requests/urllib3/certifi · tqdm · typer_slim · pyyaml · packaging · filelock · fsspec ·
functorch · torchgen. Stdlib compressors: **lzma, bz2, zlib**.

**ABSENT** (`ModuleNotFoundError`, verified by import): **brotli · brotlicffi ·
zstandard · constriction · scipy · sklearn · cv2 · numba**.

**The level set where the language flips.** Our entire inherited L20–L32 coding lineage
presumes these: L23/L32 *"split brotli streams"*, *"brotli quality=11"*; L30 *"range
coding via `constriction.stream.queue.RangeDecoder`"* (PR103 silver). **None of those
modules exist in the contest runtime.** Any submission using them must self-install at
inflate time inside the 30-minute budget.

This is the **r5 lesson measured rather than remembered** — "prove the bootstrap, never
assume host site-packages" is now backed by a direct import probe against the actual
pinned venv, not by inference from the e4 brotli precedent. It converts a remembered
caution into a MEASURED fact, and it retroactively explains why the e4 brotli bootstrap
was necessary at all.

**The organizers' own answer.** `neural_inflate/inflate.py:32-53` uses **`bz2`** — stdlib
— for both its int8 and f16 weight paths, with a hand-rolled length-prefixed int8 +
per-tensor-fp32-scale container (`<I n_tensors`, then per tensor `<I name_len`, name,
`<I n_dims`, dims, `<f scale`, `<I data_len`, int8 payload). They chose the coder that
cannot fail to import. Given `lzma` is also present and is strictly stronger than `bz2` on
this class of payload, that is a coordinate they left on the table — and one we hold,
since our own carrier work already races `raw / zlib9 / Brotli-11 / raw-LZMA` per coder
(`g4` amortisation table selects `raw-LZMA` for the large sparse fields and `Brotli-11`
for the Movable field).

**Which way it falls.** Two consequences, in order of size:
1. **Bootstrap risk is real and now priced.** Any rate plan of ours that assumes brotli or
   constriction is present carries an unproven install step inside the 30-min budget.
   `lzma` is the strongest *guaranteed-present* coder. Our `g4` table already shows
   `raw-LZMA` winning the two largest sparse fields (4,107 B and 4,111 B) — so the
   guaranteed-present coder is *also* our measured best on the fields that matter, and the
   bootstrap dependency may be discardable at little or no cost. That is a real
   simplification available to the rate axis and it should be measured, not assumed.
2. `torch 2.10.0` + `torchvision 0.25.0` + `timm 1.0.22` + `segmentation_models_pytorch
   0.5.0` + `einops 0.8.1` are free imports at decode. Anything we vendor or re-implement
   from those is unnecessary weight in `archive.zip`.
   *(Round-1 self-correction: I first wrote that we do not import `einops`. **False** —
   `grep -rn "import einops" src/tac/` returns `src/tac/margin_saliency_map.py:124,393`,
   `src/tac/through_r/blind_coordinate.py:316`, and two test modules. Recorded because it
   is the day's dominant error class caught in my own draft. The useful statement stands:
   `einops` is a free decode-time import, so those uses cost zero archive bytes.)*

**Scope boundary with ddm_ua2:** they own the `pyproject.toml` `cpu`/`cu128` group reading
and `uv.lock` defence. I report only the *materialised, importable* set. The two must be
reconciled — a declared-but-unmaterialised group would change the "absent" list, and that
reconciliation is theirs.

---

## 7. SURFACE: hard-pair ranking — a cross-check that FAILED, honestly

`damir_bearclaw_003/images/pose0_occlusion_rank_04_pair_0252.png` (1048×846 RGB, 970,076 B)
encodes an external hard-pair claim: **pair 252, occlusion rank 4**.

Looked up in our own `#622 g3` n600 atlas
(`/Volumes/VertigoDataTier/pact/ddm_g3_score_atlas_n600_20260722T204000Z/ddm_g3_score_atlas_n600.jsonl`,
8.7 MB, 600 rows; pair 252 = frames [504, 505], scored seg frame 505):

| Our currency | pair 252 value | rank (hardest = 1) of 600 | median |
|---|---:|---:|---:|
| `score_rank` (joint distortion mass) | 0.0649215 | **568** | 0.0725 |
| `d_pose_pair` | 149.043 | **531** | 161.563 |
| `pose_binds_fraction` | 0.6905 | **483** | 0.7250 |
| `pose_sensitivity_l2_diagnostic` | 3.6813 | **268** | 3.5332 |

**I tried to refute the divergence and failed.** The obvious rescue is that their ranking
is a *sensitivity* currency (∂pose/∂occlusion) while `score_rank` is *realised debt* — a
pair can be highly sensitive yet have low achieved error. Our atlas carries a sensitivity
field precisely for this, and pair 252 sits at **268/600, essentially at the median**
(3.68 vs 3.53). The currency hypothesis does not rescue agreement on any of the four.

**What survives.** Their rank-4 claim does **not** reproduce on our n600 atlas under any
pose currency we hold. Leading unrefuted explanation, labelled **INFERRED**: the filename
says `pose0`, which likely means occlusion sensitivity of **pose dimension 0 specifically**,
whereas `d_pose_pair` aggregates the scored 6 dims — we do not carry per-dimension
sensitivity in this atlas view, so the comparison may simply be to a quantity we have not
measured. Their ranking code is not shipped (only the PNG), so it cannot be settled from
this directory.

**Which way it falls.** Treat the external hard-pair signal as **non-transferable** until
a per-dimension pose-sensitivity column exists. This is consistent with `g3`'s own honest
finding — *"broad joint debt, not a strong heavy tail"* (top-10 pairs = 1.98% of joint
mass) — under which *any* hand-curated top-N pair list is weak evidence, theirs and ours
alike. Their own README pre-empts this: *"increasingly diminishing practical value from
chasing ever smaller scalar deviations."* A hand-picked example is not an n600 ranking,
and our `allergic-to-non-n600-scale` discipline says so from the other direction.

---

## 8. SURFACE: class-index order — independent external corroboration (strongest single result)

Our class order is a standing hazard: the memo records that **luma-sorting comma10k
`class_values=[41,76,90,124,161]` gives the WRONG order and "bit us 3×."** The corroboration
so far has been internal (our own `gt_n96.npz` cache).

`damir_bearclaw_003/images/segnet_classes_frame_0171_mask.png` is **512×384 — exactly the
SegNet input size — with exactly 5 unique RGB values**: a SegNet argmax map at scorer
resolution, shipped inside the pinned snapshot, never before consulted. Measured per
colour (area %, vertical centroid, row p05/p95) and matched against our MEASURED geometry:

| Their colour | area % | v-centroid | rows p05–p95 | Our MEASURED class | our area % | our rows |
|---|---:|---:|---:|---|---:|---|
| `(64,255,64)` | 49.46 | 95.0 | 9–184 | **2 = Undrivable (incl. sky)** | 49.3 | 9–182 |
| `(255,220,64)` | 25.16 | 335.2 | 292–379 | **4 = MyCar / ego-hood** | 25.6 | 290–379 |
| `(0,0,0)` | 23.74 | 240.3 | 193–282 | **0 = Road** | 22.9 | ground/mid-lower |
| `(64,128,255)` | 1.34 | 186.0 | 169–203 | **3 = Movable / cars** | 1.56 | 174–215 |
| `(255,64,64)` | 0.30 | 210.2 | 182–240 | **1 = Lane markings** | 0.59 | thin |

Area and row ranges match to within single-frame variance (this is frame 0171; ours is an
n96 mean — which is exactly why the two thinnest classes, Lane 0.30 vs 0.59 and Movable
1.34 vs 1.56, show the largest relative spread).

**Second, independent witness in the same directory:** `generate_qpmap.py:90` computes
`sky_frac = (cell == 2).sum()/cell.size` and `:91` `road_frac = (cell == 0).sum()/cell.size`.
**Class 2 = sky/Undrivable, class 0 = Road**, in executable code, written by someone with
scorer access.

**Which way it falls.** Two independent artifacts inside the pinned snapshot — a pixel map
and a script — confirm the **comma10k canonical order `[Road, Lane, Undrivable, Movable,
MyCar]`** and refute the luma-sort order (which would place MyCar at index 2, i.e. would
predict the 49.5% *top* region is MyCar — it is not). This does not change any number we
hold; it converts a rule we had to keep asserting into one with an external, adversarial
witness. Given the memo records three prior recurrences, that is worth the line.

---

## 9. Rule-118 boundary: the shipped binaries

sha256, MEASURED, not executed:

| File | sha256 | bytes | instances |
|---|---|---:|---:|
| `ffmpeg-new` | `d2909412756cb59a1e172855b60691d404c811302dde80fb2d1feec106dc37f5` | 24,480,808 | **3, all identical** (`upstream/`, `av1_roi_lanczos_unsharp/`, `roi_v2/`) |
| `lib/libSvtAv1Enc.so.2.3.0` | `ba1e44ea800041f277bf46847fae63f3578e23077d02ca01b2e1793befb54f51` | 8,213,392 | **2, identical** |

Both are mode `644` — **no execute bit set as shipped**.

**Decode-side usage: none.** `grep -n "ffmpeg\|LD_LIBRARY"` over *every*
`submissions/*/inflate.sh` and `submissions/*/inflate.py` returns **rc=1, empty**. The
binaries are referenced only by 4 `compress.sh` files
(`av1_crf31_bicubic`, `av1_roi_lanczos_unsharp`, `neural_inflate`, `roi_v2`), each setting
`LD_LIBRARY_PATH` for the encode step.

**Correcting the premise in my own tasking.** "Submissions may ship binaries" is true but
misleading: they ship them **in the submission directory, never in `archive.zip`**, and
they run them **at compress time, never at inflate**. `upstream/evaluate.py:63` charges
`archive.zip` bytes only, so 32.7 MB of encoder tooling costs exactly **0** — textbook
rule-118 (generic tool = free). This says nothing about whether a *decode-side* binary
would be legal; the shipped set contains **zero** evidence on that question, and I will not
manufacture any. What it does establish is that the organizers' own reference keeps heavy
tooling strictly on the free side, which is the conservative reading of the same rule we
operate under.

`inflate.sh` invokes bare `python -m "submissions.${SUB_NAME}.inflate"`
(`neural_inflate/inflate.sh:27`) — no venv activation, no interpreter pin. The free-import
set of §6 is therefore whatever `python` resolves to in the harness, which is the
strongest argument for preferring stdlib coders.

---

## 10. Round-1 adversarial review of my own findings

**Tried to refute, survived:**
- *Binary identity* — re-hashed all 5 instances independently; byte-for-byte equal. Sizes
  match MAIN's independently-obtained figures exactly (24,480,808 / 8,213,392).
- *Class-order corroboration* — the strongest possible objection is that the PNG is a
  rendering with arbitrary colours, so colour→index is unknown. It is: I never claimed a
  colour→index map from the PNG alone. The mapping is fixed by the **second, independent**
  witness (`generate_qpmap.py:90-91`, `cell == 2` = sky, `cell == 0` = road) and the PNG
  supplies the *geometry* (area %, row bands) that must then agree — and does. Two
  artifacts, two mechanisms, one conclusion.
- *Corridor-vs-Movable-band divergence* — tried the objection that Gaussian feathering
  (radius 24–32 at camera scale ≈ 10–14 rows at 384) softens the cut enough to be moot. It
  does not close a 22.4-row offset, and `outside_blend` is 0.50 in `neural_inflate` but
  1.00 by default. The overlap with rows 174–189 survives.
- *`neural_inflate` never ships its weights* — the strongest objection is that my grep
  scope was too narrow. Scope was all of `upstream/submissions/**` excluding `.pyc`, for
  the literal `ren_model`; four hits, all accounted for. A `.pyc` cannot contain a
  `compress.sh` copy step. Survives.

**Tried to refute, and the finding CHANGED:**
- I initially read `av1_roi_lanczos_unsharp/inflate.py` and `roi_v2/inflate.py` as loading
  the REN model, from a regex hit on `REN`. **False — it matched inside `STRENGTH`.** Only
  `neural_inflate` touches REN. Caught before it entered any claim; recorded because it is
  the exact shape of the day's dominant error class, one layer down.
- I expected the pair-252 divergence to dissolve into a sensitivity-vs-realised-debt
  currency difference. **It did not** (§7). Reported as a failed cross-check rather than
  massaged into agreement.
- I wrote in my own draft of §6 that we do not import `einops`. **False** — 5+ call sites
  in `src/tac/`. A negative-existence claim I asserted without searching, in an audit whose
  brief named that as the day's #1 error class. Caught and corrected pre-commit; left
  visible in §6 rather than silently edited out. The lesson is that the discipline has to
  fire on *my own asides*, not only on the headline claims.

**Prior work called SOUND, with the reason:**
- `#622 g3` and `#623 g4` are sound and were the load-bearing counterparties here. `g4`'s
  own round-1 review caught three defects before sealing (ordinal-Pose6 misuse, Accelerate
  FP flags, stale-checkpoint self-entry) and its receipt re-validated on read. `g3` reports
  its subset/full correlations (`r = 0.595` top24, `r = 0.234` control) and *prohibits*
  subset-only promotion on that basis — that self-imposed limit is what let me state §7
  as a clean negative instead of over-reading a 10-pair list. Both were reusable without
  re-derivation, which is the whole point of them.
- MAIN's file enumeration was accurate on every byte count I checked; the only correction
  is 19 dirs / 18 distinct (§0) and the preprocess multiplicities (§3a).

**Labelled honestly:** the corridor↔`g4` row comparison is DERIVED (normalised-y ↔
384-row index across a shared frame geometry), not directly measured in one coordinate
system. The `pose0` per-dimension explanation in §7 is INFERRED. The `0.01695` REN rate
price is DERIVED from a MEASURED parameter count. Everything in §6 and §9 is MEASURED.

**What I did not do:** execute or disassemble the binaries; open the 1048×846 pose PNG
beyond dimensions and mode; reconcile §6 against `pyproject.toml` groups (ddm_ua2's scope);
run any scorer forward.

---

## 11. Placement summary — where this audit leaves each surface

| Surface | Their coordinate | Our coordinate | Direction |
|---|---|---|---|
| Rate / encode | 522×392 lattice, CRF 31–34 sweep | 359,750 B composed carrier, rate 0.239 | Not our line. Yields **one constraint**: storing beyond 522×392 is provably wasted. |
| Spatial prior | 4-piece hand trapezoid; 63 blocks/frame | 5% of pixels = 68.7% of flip mass | Same shape, **10–20× coarser**. External corroboration; no new geometry. |
| Neural placement | 25.5 KB residual post-filter, **unpriced** | `e_p` rank-1 pose carrier ~2 KB **measured-closed** | We are an order of magnitude cheaper *and* we pay the bill. |
| Free decode operator | unsharp α ∈ {0.27, 0.40, 0.85, 2.0, adaptive}, **0 bytes** | we buy low-pass compensation in carrier bytes | **Most actionable.** 0-byte distortion recovery re-prices the `#766` waterfill. Race it. |
| Runtime closure | `bz2` (stdlib, cannot fail) | our lineage presumes brotli/constriction — **absent** | `lzma` is the strongest guaranteed coder, and `g4` already selects `raw-LZMA` for the largest fields. Bootstrap may be discardable. |
| Hard pairs | pair 252 rank 4 | pair 252 rank 268–568/600 | **Non-transferable.** Failed cross-check, reported as such. |
| Class order | class 2 = sky, class 0 = road (2 witnesses) | comma10k canonical `[Road,Lane,Undriv,Movable,MyCar]` | **External corroboration.** Luma-sort refuted from outside our pipeline. |
| Rate-vs-pose exchange | bearclaw_003 discarded near-field anchor, lost | pose = 1.24 S, the largest axis | Their loss is **explained by** our axis weight. Corroborates the ordering. |

**Apparatus finding, above all of these:** `upstream/` is gitignored
(`.gitignore:256`), so every audit surface keyed on tracked files has been structurally
blind to these 88 files for the life of the project. That is grade-5 orphan signal —
built-elsewhere, unwired-here, invisible to every gate by construction — and the fix is a
closure audit keyed on the *filesystem*, not the index.
