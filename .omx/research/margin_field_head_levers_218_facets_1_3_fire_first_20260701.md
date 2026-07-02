# #218 MARGIN-FIELD HEAD levers (facets 1 & 3, BYTE-FREE) — fire-first + wire-in

**Date:** 2026-07-01 · **Lane:** `lane_1B_margin_field_head_levers_20260701` · **Advisory:**
`[macOS-CPU advisory — REALIZED-through-R CPU-SegNet; fp32 EMA render; NON-PROMOTABLE until byte-closed exact eval]`

Orphan pursued: the MARGIN-FIELD HEAD levers **facets 1 & 3** (the sibling persistence builder
covered only facet-2 = `TopologyLossGauge` clDice/Betti). Facet-1a = fixed simplex-ETF head;
facet-1b = additive-margin softmax; facet-3 = per-class logit-adjustment (Menon 2007.07314).
All **byte-free** (or byte-negative). Target: the Lane↔Road 57%-of-flips erasure tail (#209).

## THE PIVOTAL FINDING (why the "sweep on cached gt_n600" had to change shape)

The requested probe was "a per-class additive logit offset on the frozen SegNet argmax". The
witness's OWN partition is `argmax_k phi_k` (K=5 SDF fields → palette → RGB → R → frozen SegNet).
The **legal, byte-free** offset is a per-class additive shift `b_k` on the witness `phi` (a
power-diagram / Laguerre reweight), which folds into the already-counted `out_sdf.bias`
(`phi = W·h + bias`; adding `b` to the bias adds `b` to `phi` for all pixels → **0 extra bytes**).

**Measured (n4, then n96): the phi-argmax proxy is INVALID for this witness.** Task-space
`d_seg = mean(argmax(phi) != gt_lstars)` = **0.249**, but the REALIZED-through-R
`d_seg = mean(argmax(SegNet(R(render(phi)))) != gt_lstars)` = **0.00301** (n4) — a 60–80× gap.
The witness is trained so `SegNet(render) ≈ GT`, **not** so `argmax(phi) ≈ GT`; the palette +
texture + R round-trip decouple phi-argmax from the SegNet argmax. So the sweep MUST be
evaluated **realized-through-R** (frozen CPU SegNet), never on phi-argmax. (Render fidelity
confirmed: realized baseline 0.00301 matches the per-stage attribution ~0.004.)

## FIRE-FIRST RESULT (real witness, REALIZED-through-R, n96)

Checkpoint: `levelset_n600_v2_attrclean_20260630T194549Z/levelset_witness_ema_BEST.npz` (ep650,
hosc, T=0.31, self-orient). GT: `gt_n96.npz` (96 real pairs). Grid `{-0.5,-0.25,0,+0.25,+0.5}`,
focus classes `{0=Road, 1=Lane, 3=Movable}`. Per-class 1-D sweeps + independent-argmin joint +
the analytic Menon offset `b_k=-log π_k` as a no-search baseline. Class priors (from cached L*):
Road 0.228, Lane **0.0063**, Undrivable 0.492, Movable **0.0175**, MyCar 0.257.

| quantity | value |
|---|---|
| BASELINE realized d_seg (b=0) | **0.003244** |
| baseline per-class | Road 0.0047 · **Lane 0.235** · Undriv 0.0006 · **Movable 0.021** · MyCar 0.0005 |
| Road (c0) 1-D | −0.5→0.003322 · −0.25→0.003253 · 0→0.003244 · +0.25→0.003268 · +0.5→0.003322 |
| Lane (c1) 1-D | −0.5→0.003246 · −0.25→0.003245 · 0→0.003244 · +0.25→0.003245 · +0.5→0.003243 |
| Movable (c3) 1-D | −0.5→0.003325 · −0.25→0.003264 · 0→0.003244 · +0.25→0.003256 · +0.5→0.003326 (all ≥ baseline) |
| **WINNER offsets** | Lane +0.5 (Road 0, Movable 0 — every class's 1-D argmin; measured n96) |
| **WINNER realized d_seg / Δ** | 0.003243 / Δ −6.36e-7 (−0.02% rel) — negligible |
| Menon analytic Δ (n96) | offsets (Lane +2.57, Movable +1.59, Road −1.1, Undriv −1.86, MyCar −1.2) OVER-correct → **+4.02e-3 (+124% rel WORSE)**: a global offset cannot recover localized erasure |

### VERDICT (honest, non-fake)

**A post-hoc GLOBAL per-class Laguerre offset is a NEGLIGIBLE realized-d_seg lever** on the
converged witness (−0.02% rel; ≤ −1% across all swept offsets). Mechanism: the trained `out_sdf.bias` is already near-optimal
in the constant-per-class subspace, and — decisively — the Lane erasure is a **spatially-localized
finest-scale** failure (dashes below the argmax margin), NOT a global class bias. A constant
offset boosts Lane *everywhere*, so recoveries at true-Lane boundaries are cancelled by
false-positive Lane elsewhere (net ≈ 0). This is the same reason "store-the-flip-pixels" linear
sidecars NO-GO'd ×3 and "SegNet sees REGIONS not pixels".

**Where the value actually is (facets 1 & 3 at TRAIN time, not post-hoc):** the constant offset
is the crudest realization of facet-3. The training-time facets are a strict superset:
* **facet-1a ETF head** removes the minority-class *norm collapse* from epoch 0 (all classes get
  equal-norm, max-equiangular prototypes) — the erasure never forms; also a **rate win** (the
  fixed frame is regenerable → the 5×96 head weight is free).
* **facet-1b/3 realized per-class margin hinge** widens the realized SegNet decision margin MORE
  for the rare erasure-prone classes (Lane/Movable) DURING training, so the finest-scale islands
  survive R — a spatially-adaptive lever the post-hoc constant cannot be.
These need a real n600 training run to measure (out of the $0-CPU containment scope) — the wire-in
below enables them; the post-hoc probe establishes that the *inference-time constant* ceiling is small.

## MODULE (committed, 24 tests, head 2623afd0c)

`src/tac/boundary_math/laguerre_logit_offset.py` (numpy authority + MLX-parity notes):
`power_diagram_argmax`, `apply_offset_to_sdf_bias` (byte-free fold), `menon_logit_adjustment_offsets`,
`simplex_etf` + `etf_gram_offdiag`, `additive_margin_logits`, `laguerre_offset_sweep` (engine),
`per_class_disagreement`. Rides `tac.margin_saliency_map` (#141) + `tac.boundary_math.margin_polytope`
(no new saliency). Probe: `experiments/probe_laguerre_logit_offset_sweep.py`.

## WIRE-IN SPEC (LEVELSET trainer; ALL default-off byte-identical; py_compile + --help verified)

`experiments/train_LEVELSET_witness_realized_through_R_mlx.py` (**NOTE: this file is UNTRACKED /
git-ignored in the repo — the wire-in is applied ON DISK and verified; committing it is the
owner's call**). New flags (default → byte-identical):
* `--head {softmax,etf,additive-margin}` (default softmax). `etf` → `out_sdf.weight = simplex_etf(5,
  hidden_dim)` + `out_sdf.freeze(keys=["weight"])` (facet-1a; MLX freeze verified: weight grad
  None, bias trainable, ETF cos=−1/(K−1) preserved).
* `--additive-margin FLOAT` (facet-1b base margin target when head==additive-margin).
* `--logit-adjust-per-class` + `--logit-adjust-tau FLOAT` (facet-3 Menon: rare-class margin boost
  `tau·relu(−log π_c)`).
* `--margin-field-head-weight FLOAT` (0=off). The realized through-R **per-class margin hinge**:
  `per_pix_tgt = Σ(lstar_oh · b_c); L += w · mean(relu(per_pix_tgt − _signed))`, reusing the SHARED
  `_signed` (composes with LEVER-3/4/B; extends `_seg_levers_on`).

## FOUR LEGS

### 1. Canonical equation (FORMALIZATION_PENDING — spec; register on first real n600 anchor)
`witness_head_laguerre_power_diagram_v1`:
`decoded_partition(p) = argmax_k ( phi_k(p) + b_k )`, `b ∈ R^K` folded into `out_sdf.bias` (Δbytes=0);
`b=0` ⇒ Voronoi/argmax; `b≠0` ⇒ power diagram (Laguerre weights). Menon init `b_k=−τ·log π_k`.
`d_seg/db` is **realized-through-R** (frozen SegNet), NOT phi-argmax. EmpiricalAnchor (advisory,
n96): baseline 0.003244, best post-hoc-offset Δ −6.36e-7 (−0.02% rel), Menon analytic +4.02e-3 (+124% WORSE). Producers:
`experiments/probe_laguerre_logit_offset_sweep.py`. Consumers: `tac.boundary_math.laguerre_logit_offset`,
the LEVELSET trainer margin-field-head lever. `# FORMALIZATION_PENDING: register after a real n600
training-facet row (ETF/hinge) lands — the post-hoc constant is a weak ceiling, not the headline number.`

### 2. DAG FEED (append to `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_*.md`)
**FEED-mfh (#218 facets 1&3, margin-field HEAD, BYTE-FREE):** phi-argmax proxy INVALID (0.249 vs
realized 0.00301); sweep MUST be realized-through-R. Post-hoc global per-class Laguerre offset =
NEGLIGIBLE realized lever (−0.02% rel; ≤ −1% across all swept offsets, n96) — the Lane erasure is localized-finest-scale not
global bias (constant boost self-cancels). Value = TRAIN-TIME: ETF head (norm-collapse fix +
rate win) + per-class realized-margin hinge (rare-class Lane/Movable margin widening). Wired
default-off into the LEVELSET trainer (`--head`, `--logit-adjust-per-class`, `--margin-field-head-weight`).
Module committed (2623afd0c, 24 tests). NEXT: real n600 run `--head etf --logit-adjust-per-class
--margin-field-head-weight <w>` → realized n600 d_seg vs baseline (the promotion gate).

### 3. DSL gauge SPEC (add to `src/tac/witness_dsl/gauge.py` — NOT edited here per directive)
New `GaugeComponent.HEAD_GEOMETRY` + chart `HeadGeometryGauge`:
* `SOFTMAX="softmax"` (default learned out_sdf; counted 5×d weights) — `--head softmax` (byte-identical).
* `ETF="etf"` (facet-1a: FROZEN simplex-ETF weight; regenerable → **counted_bytes ≈ −(5×d) rate win**;
  `d_seg_through_R` PENDING) — `--head etf`.
* `ADDITIVE_MARGIN="additive_margin"` (facet-1b: 0-byte train-time margin target) —
  `--head additive-margin --additive-margin M --margin-field-head-weight w`.
* `LOGIT_ADJUST="logit_adjust"` (facet-3: 0-byte train-time Menon rare-class boost) —
  `--logit-adjust-per-class --logit-adjust-tau τ --margin-field-head-weight w`.
GaugeCost cells: SOFTMAX measured=True (baseline); ETF/ADDITIVE_MARGIN/LOGIT_ADJUST measured=**False**
(PENDING, None numerics per NO-FAKE) until a real n600 row; provenance = this memo + probe.

### 4. Compute benchmark ($0, CPU-only — containment)
Probe entirely CPU (numpy fp32 render + frozen CPU-torch SegNet); **no MLX-GPU** (no contention
with any live run; none was running). Trunk cached once/pair (self-orient 3-iter fixed point);
each swept offset = cheap re-compose (`tex = logit(rgb/255) − softmax(phi/T)@palette`, exact
inversion of the ONE-CODEPATH) + one batched SegNet forward (chunk 32). n96: ~460s cache + ~18
realized calls. MLX `out_sdf.freeze(keys=["weight"])` verified for the ETF head.

## RISKS / HONESTY ANCHORS
* **n96 subset, fp32-EMA render, macOS-CPU advisory** — NON-PROMOTABLE. The pointer (0.19110) is
  UNMOVED. The promotion gate is a real n600 training run of the train-time facets, byte-closed,
  exact-eval'd.
* **The headline is a NEGATIVE for the post-hoc constant offset** — reported plainly (not narrated
  as progress). It correctly redirects the lever's EV to the train-time facets (wired, default-off).
* **facet-1b overlaps existing levers**: the realized-margin hinge is the same family as
  `margin_hinge` seg_form + LEVER-3/LEVER-4; the NEW content is the PER-CLASS (Menon) target and
  the ETF head. Borrowed-substrate accounting: ETF + per-class-target = ours-original; the hinge
  scaffold is reused.
* **Trainer is untracked/ignored** — wire-in lives on-disk (verified); flagged for owner.
