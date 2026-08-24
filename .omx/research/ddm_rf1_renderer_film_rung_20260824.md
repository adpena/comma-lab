# ddm_rf1 — the last open renderer rung, and the ceiling that outranks it

`date_utc: 2026-08-24` · `axis: [macOS-CPU advisory]` · `score_claim: false` ·
`promotable: false` · `frontier_moved: false`

**Verdict: REFUSED at 2.7749× the matched base — the mildest refusal on the renderer ladder by
12.8×, and 97.59% of it is pose.** `d_seg` rose only 1.2384×; `d_pose` rose 94.97×. My prior-law
prediction is **FALSIFIED** (§3): the rung came in *milder* than my ≥10× floor, not harsher.

`verdict_scope`: **INSTANCE** — `film_amortized_flat_w96`, archive
`34855e3c43e564d48adc492d919afa81662ebff847386d36bbf1a07304b26d21`, 179,290 B, measured on the
`[macOS-CPU advisory]` axis against the matched-instrument mst1 CPU base.

---

## 0. The finding that outranks the row

Before the rung: **the renderer axis cannot reach 0.12 even if the renderer ceases to exist.**

This is arithmetic on ar1b's census, not a new measurement. ar1b line 13 already prices the semantic
renderer at **30,856 B = 72.8045%** of the 42,382 B fixed-distortion demand; my independent
computation reproduces that share exactly. What ar1b does not state is the consequence:

| renderer axis at its physical limit | value |
|---|---:|
| entire renderer residue | 30,856 B |
| share of the 42,382 B demand | 72.8045% |
| **S with the renderer annihilated** (0 B, 0 added distortion) | **0.12767413177489606** |

Deleting every renderer byte — impossible, the frames have to come from somewhere — still leaves S at
**0.1277, above 0.12**. The renderer is therefore **never the route**; it can only ever be a
contributor to a joint move. Every rung below is priced against that ceiling, not against the goal.

Per-rung ceilings at *zero* added distortion:

| rung | bytes | % of 42,382 B demand | best-case S |
|---|---:|---:|---:|
| `film_amortized_flat_w96` | 1,078 | 2.544% | 0.14750207968096807 |
| `pointwise_svd_w96_r32` | 5,191 | 12.248% | 0.14476340180677658 |
| `nested_group_dense_w72` | 10,879 | 25.669% | 0.14097599608141767 |

## 1. The bars, derived BEFORE the measurement

Exchange rate **6.658590e-07 S/B**, CITED from `ddm_tx1_toolbox_crosswalk_20260819.md` §0 — not
re-derived. Credit for the 1,078 B cut: `25 × 1078 / 37,545,489` = **0.0007177959514657007 S**
(rj1 published 0.000717796002; agreement to 5e-11).

**Matched-instrument base** — mst1 `advisory_r1`, the dx2 EXACT archive
(`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`): `d_seg 0.0003474`,
`d_pose 0.00014701`, 180,368 B, **S 0.19318153076125097**. My recomputation from its own components
reproduces that receipt to 2.8e-17.

| bar | value |
|---|---|
| **BAR-SEG** (seg leg alone) | Δd_seg < **7.177959514657007e-06** → d_seg_max 0.000354577959514657 |
| **BAR-POSE** (pose leg alone) | d_pose_max **0.00015256585279559186** → Δd_pose < 5.5558527955918715e-06, i.e. d_pose may rise **3.78%** |
| **BAR-JOINT** | S_candidate < **0.19318153076125097** |

The pose bar is the severe one: a 3.78% rise in `d_pose` spends the entire credit by itself.

### 1.1 Instrument match — verified on all three legs

Base, W72 (Arm A), and this candidate all carry `device=cpu`,
`lineage=PYAV_YUV420_TO_RGB`, `runtime_decoder=AVVideoDataset`, GT video sha
`2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`, `n_samples=600`. Same
instrument, same lineage, same n. The receipts themselves carry the cross-lineage warning and
quantify the fork against contest-CUDA: **1.4425× on d_seg, +1.4061e-04 ADDITIVE on d_pose**.

### 1.2 No-op detector — the renderer is isolated

The measured object is the advertised one, and the byte saving lands where rj1 said:

- The WD2S packet header decodes to **`form=flattened, W=96, depth=4, rank=0`** — full width, all
  four blocks, one trunk FiLM. The receiver's form dispatch is a genuine 3-way branch
  (`dense`/`factorized`/`flattened`), with the flattened path applying a single `flat_film` at the
  trunk instead of per-block maps. Not enum padding.
- The archive is a single stored ZIP member `p`; the entire −1,078 B lands in it.
- Within `p`: the **135,852-byte suffix from offset 44,416 is byte-identical** to dx2 — carrier,
  HPAC model, residual table and framing untouched. Only 31 prefix bytes differ: the `u32LE` at
  offset 10, which decrements by **exactly −1,078**, and 29 bytes at 13,529–13,559, immediately at
  ar1b's nominal semantic-stream boundary (13,560).
- **Packet vs physical:** the packet shrank 473 B (36,130 → 35,657) while the physical stream shrank
  1,078 B — a **2.28× amplification**. The byte credit is coding-sensitive, not just a parameter
  count. (Consequence for any retrain: see §5.3.)

## 2. The measurement

**REFUSED** — but by far less than anyone expected, and for a reason that is almost purely pose.

| quantity | matched CPU base (dx2) | `film_amortized_flat_w96` | ratio |
|---|---:|---:|---:|
| `d_seg` | 0.0003474 | **0.00043022** | **1.2384×** |
| `d_pose` | 0.00014701 | **0.01396208** | **94.9737×** |
| archive B | 180,368 | 179,290 | −1,078 |
| **S** | 0.19318153076125097 | **0.5360625194757408** | **2.7749×** |

S recomputed from components; the printed display reads **0.54** and differs from canonical by
0.0039. Inflate produced 1 raw file of 3,662,409,600 B, STRICT validation passed. n=600.

### 2.1 The decomposition — pose is 97.59% of the damage

| term | value | × credit |
|---|---:|---:|
| rate credit (1,078 B) | −0.000717796 S | 1.0× |
| seg cost | +0.008282000 S | **11.5×** |
| pose cost | +0.335316785 S | **467.1×** |
| distortion total | +0.343598785 S | **478.7×** |
| **net ΔS** | **+0.342880989 S** | — |

The realized rate credit matches rj1's published 0.000717796002 to five figures — as on W72, **the
rate claim is exactly sound and the distortion claim is what fails.**

The seg leg is nearly intact: `d_seg` rose only **1.2384×**, overshooting BAR-SEG by 11.5×. The pose
leg is the whole story: `d_pose` rose **94.97×**, against a BAR-POSE that allowed **3.78%**.

### 2.2 Against W72 — milder on every matched metric except pose share

Same instrument, same lineage, same n, same metric in each row:

| metric | `film_w96` | `W72` | |
|---|---:|---:|---|
| net ΔS | 0.3429 | 6.6717 | **19.46× milder** |
| S ratio vs matched base | 2.7749 | 35.5364 | **12.81× milder** |
| exchange ratio (damage ÷ credit) | 478.7 | 922.0 | **1.93× milder** |
| `d_seg` multiple of base | 1.2384 | 67.6930 | **54.66× milder** |
| `d_pose` multiple of base | 94.97 | 13,171.98 | **138.69× milder** |
| **pose share of damage** | **97.59%** | 65.31% | **1.49× harsher** |

**Mildness does predict distortion** — strongly, and on every axis. The one place film_w96 is worse
is the *share* of its damage carried by pose, and that is the interesting result.

## 3. Prior-law prediction, adjudicated — FALSIFIED

I predicted: *"REFUSED, and by ≥10× — but by LESS than W72's 46.3×"*, with the falsifier *"it lands
within 3× of break-even, OR it is refused by MORE than W72 despite the milder edit."*

**Measured: 2.7749× the matched base. 2.7749 < 3.0. The first falsifier clause FIRES. FALSIFIED.**

It is falsified in the direction I did not expect. I predicted a *harsher* refusal than it delivered:
the rung came in **milder** than my floor, not harsher than my ceiling. The "≥10×" leg was wrong by
3.6×. The "less than W72" leg was right, and by a wider margin than I thought (12.8× on the S ratio).
Count it as a miss: five concordant arms measuring dx2 at a sharp optimum led me to over-predict the
penalty for a *conditioning* edit, because the prior was built on *capacity* edits.

### 3.1 Correcting the coordinator's adjudication — a units mismatch

MAIN independently adjudicated this FALSIFIED, which is the right verdict, but reached it by the
wrong route: it compared **this rung's exchange ratio (478.7×)** against **W72's S ratio (46.3×)** and
concluded the milder edit was "refused by MORE than W72". Those are two different quantities. Held to
the same metric, film_w96 is refused by **less** than W72 on every axis (§2.2). The prediction fails
on the *≥10×* clause, not the *worse-than-W72* clause. Two further corrections:

- **Pose share is 97.59%**, not 97.9% (`0.335316785 / 0.343598785`).
- **The n=2 exponent does not reproduce.** MAIN reports damage/credit ~ bytes^−1.010, implying
  renderer damage is roughly constant in the byte cut. From these two rows I get
  damage ~ bytes^**+1.2836** and exchange-ratio ~ bytes^**+0.2835** — neither is −1.010, and damage
  is plainly *not* constant (0.3436 vs 6.6790, a 19.4× spread). MAIN was right not to bank it; I do
  not bank it either, and I additionally cannot reproduce it. n=2 across two different mechanisms is
  not a within-family fit and I decline to fit one.

### 3.2 The FiLM-carries-pose hypothesis — supported, not established

MAIN proposes that FiLM is the temporal conditioning, pose is a temporal quantity, and amortizing
four block-local FiLM maps into one trunk map destroys precisely what PoseNet reads.

**What the receipt supports.** The two rungs damage the two axes in sharply different proportions.
W72 cut *capacity* (width 96→72) and broke both legs — seg 67.7× base, pose 13,172× base. film_w96
kept full capacity and full spatial mixing, changed only *conditioning*, and broke essentially one
leg — seg **1.2384×**, pose **95×**. A conditioning-only edit that leaves seg almost untouched while
moving pose two orders of magnitude is the signature the hypothesis predicts.

**Why that is not proof.** n=1 per mechanism. The receipt cannot separate "FiLM specifically carries
pose" from the weaker "any small renderer perturbation is pose-selective, and W72's cut was simply
large enough to break seg as well". The renderer-carries-pose result already predicts pose-dominance
for *any* renderer edit; what needs explaining is the *degree*.

**What would decide it, cheaply.** The third rung. `pointwise_svd_w96_r32` (175,177 B, sha
`ea9303074b3083f45d12bd22dac56ef66ad17de6d316887d3224911977d756d1`) **retains per-block FiLM and
W96** and cuts only the pointwise operators — the exact complement of this edit. If FiLM carries
pose, SVD should be markedly *less* pose-selective than 97.59% at comparable damage. rj1 closed that
rung for *candidacy*; as a **mechanism probe** it is decisive, already built, already byte-closed, and
costs one advisory. I did not fire it — one advisory was my charter — and I recommend it as the next
cheap row.

## 4. A correction the campaign should carry: W72 is 35.5×, not 46.3×

`ddm_w72_distortion_advisory_20260823.md` publishes "**46.3× worse**". Its S arithmetic is exact — I
recompute S(W72) = **6.864979038642395** from its own components and reproduce the memo digit for
digit. But its *denominator* is the dx2 **contest-CUDA** pointer (0.14821987563243377) while its
numerator is a **macOS-CPU/PyAV advisory**. The receipt for that very run carries the cross-lineage
warning that forbids the comparison.

| comparison | ratio |
|---|---:|
| W72 CPU advisory ÷ dx2 **CUDA** pointer | 46.3162× (published) |
| W72 CPU advisory ÷ **matched CPU** base | **35.5364×** |

The verdict does not move — both are refusals by more than an order of magnitude — but the number
should be carried correctly, and any discriminator's control arm must use the matched figure.

Arm A's matched-instrument decomposition, which §5 uses:

| term | value | × credit |
|---|---:|---:|
| rate credit (10,879 B) | 0.007243880 S | 1.0× |
| seg cost | +2.316915000 S | 319.8× |
| pose cost | +4.362126387 S | 602.2× |
| **distortion total** | **+6.679041387 S** | **922.0×** |

Pose is **65.31%** of the damage, reproducing w72's 65.3% on the matched base.

## 5. LEG B — the retraining discriminator, pre-registered

### 5.1 The confound Leg B exists to remove

Every renderer number the campaign holds — W64 `S≈14.8829`, W72 `S 6.864979038642395`, and §2 — was
produced the same way: take dx2's **trained** W96 operators, restructure the architecture,
byte-close, measure. The weights were never trained for the structure they were measured in. That is
one experiment repeated three times, and it cannot distinguish:

- **H1 — the family is dead:** the renderer has no slack; any cut costs more than it buys; or
- **H2 — the edit is the confound:** un-retrained surgery is catastrophic, and joint optimization
  recovers most of the cut.

rj1 itself says the rungs "may retain a useful fraction of [the] cut **after joint optimization**"
(line 141) and marks its whole distortion column `UNMEASURED`. Nobody has separated H1 from H2.

### 5.2 The arm choice — the MILDEST rung, not the largest

The instinct is to run the discriminator on W72 because it buys the most. That is the wrong arm. Run
it on **`film_amortized_flat_w96`**:

1. **A negative is decisive.** Un-retrained severity is monotone in edit magnitude, and §2 extended
   the ordering to three points: `d_seg` multiples of base are W64 **91.6×**, W72 **67.7×**,
   film_w96 **1.2384×**. If the *mildest* edit cannot pay for itself **even with full retraining**,
   the negative covers every renderer cut ≥ 1,078 B. A W72-only negative covers W72.
2. **The control is free and perfectly matched** — §2 measures it, same instrument, same lineage.
3. **It is the cheapest to recover** — one trunk FiLM replacing four block-local maps is the
   smallest perturbation on the ladder.

| | Arm A (control) | Arm B (treatment) |
|---|---|---|
| rung | `film_amortized_flat_w96` | `film_amortized_flat_w96` |
| weights | dx2 operators, **un-retrained** | same warm start, **retrained** |
| measurement | §2 of this memo | pre-registered below |

The only variable is training.

### 5.3 Objective — and why it is not a proxy

**Score-aware: frozen SegNet + PoseNet in the loop.** Not L2-to-dx2-frames, not parameter agreement,
not PSNR. Two campaign results forbid the proxy:

- rj1 line 151 records a **349× understatement** from agreement-based scoring of exactly these rungs.
- The renderer-carries-pose result (three magnitudes, one direction) means PoseNet scores the
  **frames**; an L2 objective has no term that tracks the pose head.

Eval roundtrip simulated in the inner loop; EMA shadow is the inference checkpoint; per-stage
checkpoints; resumable-from-disk — CLAUDE.md non-negotiables, not choices.

**Rate is NOT a loss term — and it is NOT fixed either.** The pre-registered trap: the byte count is
structural, so it does not belong in the objective, but the WD2S packet is **entropy-coded**, so
retrained weight *values* compress differently. §1.2 measured the sensitivity directly — a 473 B
packet change produced a 1,078 B physical change, 2.28×. rj1's 1,078 B is an *initialization-time*
figure and **does not transfer to a retrained packet**. Arm B's credit must be **re-measured by
re-encoding the retrained packet and rebuilding the archive**, never inherited.

**Solve jointly, never by composition.** jg5 measured a composed candidate at d_pose 0.00326804 =
**467.3×** the pointer, S 0.3192 not the projected 0.156: seg edits bought −0.012847 S on seg and
cost +0.172 S on pose, a **13.4× loss**. Composition of two separately-finished candidates is a
measured failure mode on this exact object class.

### 5.4 Stopping rule

Derived, three-way, whichever fires first:

- **(i) patience** — realized joint ΔS vs the matched base fails to improve for N consecutive stage
  checkpoints;
- **(ii) budget** — wall-clock cap reached;
- **(iii) EARLY KILL** — at the mid-point checkpoint, retrained `d_seg` is not within 10× of the
  matched base. The cheap falsifier: a dead family must not consume the full budget.

### 5.5 The falsifier

> **KILLS THE FAMILY.** If `film_amortized_flat_w96`, warm-started from the exact dx2 operators and
> retrained to its stopping rule, still measures joint **ΔS > 0** against the matched CPU base — i.e.
> the retrained archive does not beat **S 0.19318153076125097** — then renderer re-representation is
> **closed as a byte source at every cut ≥ 1,078 B**, because the mildest available edit, given full
> retraining, could not pay for itself.

> **KEEPS IT ALIVE.** If the retrained rung measures joint **ΔS < 0**, un-retrained editing was the
> confound, H2 holds, and the fire-order escalates to W72 and then to the full joint mechanism.

### 5.6 What Leg B does NOT settle, even if it succeeds

- **It cannot produce the route.** §0 binds: renderer annihilation lands 0.12767, above 0.12. A
  perfect Arm B result here is worth 1,078 B = **2.544%** of the demand.
- **"Family alive" ≠ "candidate."** Arm A is now measured: exchange ratio **478.7×**, of which
  **467.1× is pose**. Retraining must buy nearly three orders of magnitude for this rung to break
  even, and essentially all of it must come from the pose leg — the seg leg is only 11.5× over and
  would be cheap to close alone. Even a 10× improvement leaves the rung **48× from break-even**.
  Leg B answers whether retraining moves the needle, not whether it produces a row.
- **The target is now specific, which is the one thing §2 bought Leg B.** Arm B does not need to
  recover a diffuse "distortion" penalty. It needs to recover a **94.97× pose regression** while
  holding `d_seg` near its already-fine 1.2384×. That is a much narrower and more testable objective
  than the charter could state this morning.
- **No verdict transfers across rungs by interpolation** — rj1's rule stands.

### 5.7 What a successor must build — inventory, not guesswork

rj1 gates admission on two mechanisms it declares `NOT_SOLVED`. Both are **PARTIAL**, not
memo-only, and neither needs new math:

| mechanism | status | what exists | what blocks reuse |
|---|---|---|---|
| in-compile exact-object **compensation** | **PARTIAL** | Schur/GN solver `experiments/ddm_qs1_frame0_schur_coupled_solve.py:591 solve_one`; exact-object solve `ddm_qs5_resolve_compensation.py:482 solve_exact_object`; the QS4 cure as an executable assert at `ddm_qs1…:1001 assert_compensation_matches_compile_object`; counted decode codec `ddm_qs2_compensation_overlay_runtime.py` | `CP135Surface.load()` (`ddm_qs1…:276`) takes **no arguments** and hard-refuses any archive but CP135's 186,252 B (`:550`). Retarget = parameterize on `(runtime_dir, archive_path, raw_path)` |
| **carrier re-solve** | **PARTIAL, stronger** | solve/splice/parse-back/byte-close chain `ddm_up3_carrier_splice.py` with 3 controls + 35 tests; `carrier_section_from_archive:276` **already takes `(archive_path, runtime_dir)`** — the natural retarget seam. This chain **shipped a real T4 pointer move**: 0.15659459685822907 → 0.15652626435208142, Δ −6.833250614765585e-05 | pinned to the up3/jg4 objects |

**One custody risk, and it is P0.** The compensation encoder half — `encode_cap1`,
`encode_cps3_coefficients`, `decode_cps3`, `solve_damped_least_squares` — is **not in git**. It is
imported by path from
`/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/`. Every other
copy of `coefficient_ar1_codec.py` on both SSDs and in the repo is **decode-only**. One disk, one
copy, outside version control, on the volume that is currently at 100%. Vendoring it is the
shortest-path item and it is independent of any renderer decision.

**Also measured:** the DX2 container already carries a **compensation slot** in its RX1 model parse
(`runtime/residual_archive.py:255/:434/:471`); it is **empty** in all three rj1 rungs. The rungs ship
uncompensated by construction.

**The trainer already exists too — Leg B is a rebase, not a build.**
`experiments/ddm_wd3_scorer_aware_width_distillation.py` is the joint retraining machinery: both
frozen scorers in the loss, a faithful eval roundtrip (`:806-830` — bilinear lift to 874×1164,
round-STE, then each scorer's own `preprocess_input`), and full-state resume. A live
`d_seg + d_pose + rate` dual-constrained objective template exists at
`experiments/ddm_jo3_joint_objective_entrypoint.py` (frozen scorers `:575-583`, upstream YUV6 patched
for gradient reach `:574`, `loss.backward()` `:754`, fail-closed on non-finite or zero residual
gradient `:757-761`) — reuse that plumbing rather than rebuild it.

Two pre-registration items fall out, and both are cheap:

1. **The warm start is orphaned.** rj1 writes `renderer_initialization.pt` at
   `ddm_rj1_renderer_joint_move.py:584` and **nothing reads it** — no `--init-from` / `--warm-start`
   flag exists in `experiments/`, `src/tac/`, or `tools/`. The existing warm-start surfaces are
   `--init` and WD4's packet-bytes path. Gap 1 is a small birth adapter, not new science. It is also
   a live instance of the campaign's own default-off/orphan class: a retained artifact with no
   consumer.
2. **WD3 already carries the exact pose term — the coordinator's redirect is refuted.** MAIN advised
   designing Leg B against `tools/train_ddm_cl1_hpac_capacity.py`, on the grounds that its loss has a
   rate term and `grep -c pose` returns 0, making "does a pose term change the exchange ratio?" the
   Leg B question. Both halves are wrong, and I verified each at source:
   - **Wrong component.** `cl1` instantiates `IntegerHPAC` (`:1056`) — the **HPAC token probability
     model**, ar1b's separate 13,515 B section. It renders **zero** frames
     (`grep -c 'render|rgb'` = 0) and contains **no scorer at all** — not just pose is missing;
     `segnet`, `posenet` and `scorer` return nothing. Its loss (`:1320-1322`) is token cross-entropy
     plus a *weight-bits* penalty. It does not train the renderer.
   - **Wrong premise.** The renderer trainer, WD3, has **79** scorer references, loads
     `upstream/models/posenet.safetensors` (`:86`), and its total loss (`:782-789`) is
     `calibrated_seg + pose_score + duals·(margin, teacher_kl, decode, teacher_pose)` where
     `pose_score = sqrt(clamp(10.0 * pose_mse, min=1e-20))` (`:775-776`) — **literally the contest's
     √(10·d_pose) term, unweighted, in the objective**, against `original_pose6`.

   So the renderer trainer already optimizes the exact score functional including the nonlinear pose
   term. Leg B's question is therefore **not** "add a pose term"; it is the question as chartered —
   **does retraining at all recover the cut** — and §2 makes it much more pointed than it was this
   morning: the thing Arm B has to recover is a **94.97× pose regression**, against a trainer that
   was already minimizing exactly that quantity for the *dense* form. MAIN's underlying instinct
   (that the objective's treatment of pose is the crux) survives and is sharpened; its pointer does
   not.

3. **Pin the eval roundtrip explicitly.** The R operators in-tree are **not interchangeable** — wd2
   takes its MSE at 874×1164 with no downsample back (`:857-871`); wd4 **inverts** the direction,
   computing loss at native 384×512 and downsampling the *teacher* into it (`:815-826`); the
   `lifted/*` trainers default `exact_path=False`. Only the contest-exact form
   (bicubic → 874×1164, uint8 STE at camera res, bilinear → 384×512) is the scored R. Arm B must
   name its R at pre-registration; inheriting a default here silently changes the objective.

### 5.8 A corpus defect surfaced in passing — the orphan +2.396e-4

rj1 line 150 closes a path by citing "QS4's `+2.396e-4` pose damage". **The qs4 memo does not
contain that number** — its pose term is `+1.378369737898914e-5`, an order of magnitude smaller.
The figure appears **22 times** across at least 10 memos and one canonical equation module, with
**three different mechanisms** attached: a compensation carried onto a different *lattice*
(`ddm_sa3…:54`), a *stale* compensation (`ddm_wc2…:749`), and the *removal* of elements
(`ddm_na7…:248`). rj1's "solved for another renderer *object*" is a fourth. I did not find a primary
receipt for it in `.omx/research/*.md` or `src/tac/`. This is load-bearing — it is the stated reason
a whole compensation-transfer path is closed — and it should be traced to a receipt or relabelled.
**Scope:** absence claimed only for `.omx/research/*.md` and `src/tac/`; it may exist in a JSON
receipt or an SSD store I did not sweep.

## 6. NOT CLAIMED

- **Not a score.** `[macOS-CPU advisory]`, `score_claim: false`, `promotable: false`. The pointer is
  untouched. Only `upstream/evaluate.py` on contest hardware produces a score.
- **The pose column is not decision-usable on its own.** This run's GT is `PYAV_YUV420_TO_RGB`, not
  the `DALI_NVDEC` authority lineage. The receipt quantifies the fork as **+1.4061e-04 additive** on
  d_pose. My base/candidate delta is same-lineage and therefore internally valid; the *absolute*
  d_pose is not comparable to a contest-CUDA row.
- **CPU seg deltas are upper bounds on CUDA.** There is no transfer law. A CPU refusal could be
  smaller on CUDA — that is the only direction that could rescue a marginal row.
- **This does not close the renderer family.** It closes one rung, un-retrained, on one axis. The
  retraining question is Leg B and Leg B **was not fired**.
- **No trainer, no Modal job, no paid action, no burn.** No `upstream/` file was modified. MAIN owns
  the scorer lane and every heavy launch.
- **§0's ceiling is arithmetic on ar1b's census, not a re-derivation of it.** I reproduced ar1b's
  72.8045% share exactly; I did not independently re-audit the 66,591 B residue map.
- **The 2.28× packet-vs-physical amplification is reported at face value.** I measured both numbers;
  I did not isolate the mechanism that produces the gap.
- **The +2.396e-4 orphan is scoped.** I claim only that I did not find a primary receipt in
  `.omx/research/*.md` or `src/tac/`. It may exist in a JSON receipt or an SSD store I did not sweep.
- **"Callable" in §5.7 means the entry point exists and is imported in-tree** — not that I ran it.
  I executed no compensation, carrier, or training code.
- **The severity-monotonicity argument in §5.2 is an ordering observation on three points** (W64
  91.6×, W72 67.7×, film_w96 1.2384×), not a law. rj1's no-interpolation rule still forbids
  predicting a rung from its neighbours.
- **FiLM-carries-pose is a HYPOTHESIS, not a finding.** §3.2 states what the receipt supports
  (a conditioning-only edit left seg at 1.2384× while moving pose 95×) and what it cannot separate
  (FiLM-specific vs any-small-perturbation pose selectivity). It is n=1 per mechanism. I name the
  decisive probe; I did not run it.
- **I did not fire the `pointwise_svd_w96_r32` mechanism probe.** One advisory was my charter. Its
  recommendation in §3.2 is a proposal to MAIN, not a result.
- **I do not bank any damage-vs-bytes exponent.** I report that I cannot reproduce MAIN's −1.010 and
  give what I do compute (+1.2836 / +0.2835), explicitly as unfitted n=2 arithmetic across two
  different mechanisms — not a law, not a within-family fit.
- **The trainer findings in §5.7 are source reads, not runs.** I verified WD3's loss composition and
  cl1's model class by reading them. I executed neither.

## 7. Payload custody

Canonical retained root: `/Volumes/APDataStore/pact/ddm_rf1_renderer_film_rung/`.
**Vertigo was at 100% (8.4 GiB free) at preflight and was read-only for this arm**; nothing was
written there.

- `candidate_runtime_r1/` — assembled by `tools/assemble_candidate_runtime.py`; receiver pin
  **DERIVED** by hashing the archive at its destination, written, re-read via AST, re-checked.
  No digest was hand-typed. The tool reported `pin already matched the archive; unchanged`, so
  rj1's seal was correct and this was a verification, not a repair.
- `attempt_r1/work/` — `archive.zip`, `extracted/`, `inflated/` (full raw payload **retained**),
  `provenance.json`, `inflated_outputs_manifest.json`, `report.txt`, `contest_auth_eval.json`.
- `VERDICT.json`, `CUSTODY_INVENTORY.json` — bars, adjudication, and `sha256` + byte count for every
  retained file.

| artifact | bytes | sha256 |
|---|---:|---|
| candidate `archive.zip` | 179,290 | `34855e3c43e564d48adc492d919afa81662ebff847386d36bbf1a07304b26d21` |
| raw inflated `0.raw` | 3,662,409,600 | `03871df432a33136c8576c8111ba052a6bc135a0e09a3ba4bc033d46fb07855e` |
| inflated aggregate | — | `d8de02da36fe114ac2826eadaa41c0cc96d503062c80045d8f4f0e6ad3aeff4e` |
| **retained tree** | **3,781,779,310** (77 files) | see `CUSTODY_INVENTORY.json` |

Timings: inflate 918.6 s · evaluate 444.5 s · total 1,364.3 s. Cost $0 (local). Nothing was measured
and discarded; the 3.66 GB raw payload is retained in full, not certified-and-deleted.

### A small defect in the assembly tool, reported not patched

`assemble_candidate_runtime.py` normalizes the `ARCHIVE_BYTES` integer literal on every re-pin: this
tree's `179_290` was rewritten to `179290`. The value is identical and `pin_was_stale` correctly
reported `False`, so nothing here is affected. But the tool's own docstring says "a re-pin must touch
the pin and nothing else", and this changes the file — and therefore the output tree hash — even when
the pin does not change. That is the same shape as the `ddm_fs3` incident the docstring cites, where
a same-instrument comparator correctly refused a tree because whitespace moved. Worth a one-line fix
(skip the write when both constants already match) before some future comparator trips on it.

STORES CONSULTED: `.omx/research/ddm_rj1_renderer_joint_move_20260823.md` (full — the rung table, the
W64 prior at line 149, the 349× agreement understatement at line 151, the `NOT_SOLVED` gates) ·
`ddm_w72_distortion_advisory_20260823.md` (full — Arm A, and the ratio corrected in §4) ·
`ddm_tx1_toolbox_crosswalk_20260819.md` §0 (exchange rate, **CITED not re-derived**) ·
`ddm_ar1b_archive_residue_purchase_20260822.md` (the 66,591 B census; renderer 30,856 B / 72.8045%;
the semantic span `[13560,44416)`) · `ddm_dg2_diagonal_distortion_verdict_20260824.md` ·
`ddm_qs4_collateral_suppression_20260813.md` + `ddm_qs5_resolve_compensation_20260813.md` (the
compensation line and the orphan-citation check) · the jg-line (`jg1`/`jg2`/`jg3`/`jg5`) and
`up2`/`up3` for the carrier chain and jg5's 13.4× composition loss · the mst1 base receipt
`/Volumes/VertigoDataTier/pact/ddm_mst1_manufactured_stage_split/advisory_r1/work/contest_auth_eval.json` ·
the sealed rj1 runtime tree (`cpr1/wd2_receiver.py` form dispatch, `runtime/residual_archive.py`
compensation slot, `runtime/compensation_overlay.py`, `cpr1/carrier_codec.py`) ·
`experiments/ddm_wd3_scorer_aware_width_distillation.py`, `ddm_wd2`/`ddm_wd4`, `ddm_jo3_joint_objective_entrypoint.py`,
`ddm_qs1_frame0_schur_coupled_solve.py`, `ddm_up3_carrier_splice.py` ·
`tools/assemble_candidate_runtime.py` + `tools/fire_local_advisory.py` (read before use) ·
`experiments/contest_auth_eval.py` (the `._*` / `__pycache__` manifest filters at `:98/:163/:214`) ·
`.omx/state/canonical_task_status.jsonl`.

**Ledger note — the charter's task ids are not repo-resolvable.** `#1222`, `#1224`, `#1225`, `#1237`
are all above the repo ledger's numeric ceiling (`task_id` range **383..1181**, 228 distinct ids), and
`#877`/`#1034`/`#1122`/`#1140`/`#1142`/`#1147`/`#1219` are absent from it. They are harness-side
TaskList ids the repo never received — the TASK-LEDGER-SPLIT genus recurring. I therefore cite the
**content** of those claims through the repo memos that carry it (rj1 and w72 quote them verbatim),
never the bare ids, and I could not independently verify any claim that exists only as an id.

Also read at source for §5.7: `tools/train_ddm_cl1_hpac_capacity.py` (`:1056` model class, `:1320-1322`
loss, `:846` `--rate-lambda`) and `experiments/ddm_wd3_scorer_aware_width_distillation.py`
(`:86` PoseNet load, `:775-776` pose term, `:782-789` total loss, `:806-830` eval roundtrip,
`:833-845` `scorer_forward`).

## 8. What fires next

- **CHEAPEST DECISIVE ROW — the `pointwise_svd_w96_r32` mechanism probe.** Owner: MAIN or a
  designated arm. One local advisory, ~1,400 s, $0. The rung is already built and byte-closed
  (175,177 B, `ea9303074b3083f45d12bd22dac56ef66ad17de6d316887d3224911977d756d1`) and retains
  per-block FiLM and W96 — the exact complement of this edit. It discriminates FiLM-carries-pose
  from generic-renderer-pose-selectivity in one shot. rj1 closed it for candidacy; this fires it as a
  **probe**, which is a different question.
- **QUEUED WITH A FIRE ORDER — Leg B (§5).** Owner: MAIN-designated successor with the governed
  trainer. Fire trigger: the SVD probe returns, plus a WD3 rebase onto DX2 and the birth adapter for
  `renderer_initialization.pt`. Arm A is measured and free (§2).
- **P0 CUSTODY, INDEPENDENT OF ANY RENDERER DECISION — vendor `cpr1_sub4`.** The compensation
  **encoder** (`encode_cap1`, `encode_cps3_coefficients`, `decode_cps3`, `solve_damped_least_squares`)
  exists in exactly one place: `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/`,
  outside git, on the volume that is **currently at 100%**. Every other copy in the repo and on both
  SSDs is decode-only. This is a single-point-of-failure on a mechanism rj1 gates admission on.
- **TRACE OR RELABEL the orphan +2.396e-4** (§5.8) — 22 citations, four attached mechanisms, no
  primary receipt found in the scope I searched, and it is the stated reason a compensation-transfer
  path is closed.
- **ONE-LINE POLISH — `assemble_candidate_runtime.py`** (§7): skip the write when both pin constants
  already match, so an unchanged pin cannot alter the output tree hash.

Own-vehicle frontier: **dx2 — S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`** — gap to
0.12 = 0.028220 ⇒ shed 42,382 B at fixed distortion, or 150 B at zero distortion. **UNMOVED by this
row.**
