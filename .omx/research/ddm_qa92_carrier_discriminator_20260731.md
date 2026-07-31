# ddm_qa92 — THE Lane-carrier RECEIVER DISCRIMINATOR (gc12 rung 0, task #801)

**Model: claude-opus-4-8.** The ONE demanded measurement of the gc12 wall-branch birth-completion
ladder (`.omx/research/ddm_gc12_wall_branch_convocation_20260731.md`, b4d317538d, §4). Fires the free
scorer slot. Discriminates the Lane-carrier family at MECHANISM level BEFORE any build.

**Pointer honesty FIRST: 0.1910828242 [contest-CPU] UNMOVED.** Every number below is
**[macOS-CPU advisory]**, `score_claim=false`, `research_only`. This unit is MEANS — a $0 discriminator,
not a score mover. No exact row moves here.

## ANSWER (lead)

**Rung 2 SKIPS to burn-4 (Contrarian bound P·O = 0.017 S < 0.05).** Painting the erased super-nucleus
Lane structure ONTO the control_tail render — even with the **PERFECT GT-RGB oracle** — is
**net-NEGATIVE**: it recovers 41% of the target Lane flip-mass but the frozen SegNet's large ERF responds
to the injected structure with **+0.317 S of collateral** (off-target flips), so the **JOINT ΔS is +0.300
S (WORSE)**. The flat 1-2KB prototype tier is worse still (recovery 19%, joint ΔS +0.225 S). The
paint-on-texture carrier family (b)/(d) is **not worth a dedicated rung-2 arm**; the slot goes to the
burn-4 charter. This CORROBORATES fp1's verdict ("the receiver is the wall") on the localized case fp1
left open, and it REFINES QA91's pool downward.

| quantity | value | reading |
|---|---|---|
| BASE d_seg (control_tail ep499) | **0.0049411** | == pa1r NEW BEST endpoint EXACTLY (harness validated) |
| BASE Lane S-units | 0.12438 | 25% of the 0.494 seg term (matches QA91's ~0.13 Lane share) |
| **P** = erased super-nucleus Lane pool remaining | **0.04189 S** | REFINES QA91's 0.134 *upper* bound: the continuation to ep499 already recovered most of it |
| n erased super-nucleus components (of 9035) | 4041 | the actual still-erased set at the control_tail endpoint |
| **O** oracle recovery frac (pixel flip-mass) | **0.40732** | oracle recovers 41% of the target pool (comp-level 0.421) |
| **F** flat recovery frac (pixel flip-mass) | **0.19394** | flat recovers 19% (comp-level 0.233) |
| **P·O** recovered pool (target-only) | **0.01706 S** | < 0.05 Contrarian bound → rung 2 SKIP |
| P·F recovered pool (target-only) | 0.00812 S | flat recovers even less |
| **collateral (oracle)** off-target added flips | **+0.31698 S** | REAL receiver physics (identity-fill control = bit-identical, 0 artifact) |
| collateral (flat) | +0.23300 S | |
| **JOINT ΔS oracle** (additive-S, sole authority) | **+0.29992 S (WORSE)** | = −P·O + collateral (identity holds exactly) |
| **JOINT ΔS flat** | **+0.22487 S (WORSE)** | |
| per-class ΔS oracle [Road,Lane,Undriv,Mov,MyCar] | [+0.125, +0.034, +0.055, +0.065, +0.020] | EVERY class worse — incl. Lane net +0.034 despite 41% target recovery |
| per-class ΔS flat | [+0.130, +0.026, +0.024, +0.028, +0.018] | |
| falsifier route | **RUNG2_SKIP_TO_BURN4** (O≥0.25 so family not formally closed; F<0.7·O so (e1)>(b) IF it fired; but P·O<0.05 dominates) | |

## PRIOR-LAW PREDICTION (stated before the gate; gc12 §4 certifies QA92 genuinely-unpredicted)

The corpus BRACKETED but did not predict the decisive quantity. Tishby/Atick predicted HIGH recovery
(lane-like strokes in context are scorer-expected structure). Daubechies predicted PARTIAL + AA-sensitive
(high-freq along-normal energy shaved by bicubic-R+uint8) and demanded AA-at-camera-res or "tier-1
measures aliasing". QA91 bounded the pool ≤ ~0.134 S. My own prediction (fp1-continuation): the receiver
wall (fp1's 0.008305 flat-field floor) would generalize to the localized case ⇒ paint dominated by
receiver loss. **MEASURED:** within-target recovery is PARTIAL (O=0.41, closest to Daubechies) — but the
DECISIVE, unpredicted quantity is the **COLLATERAL**: even the oracle paint injects +0.317 S of
off-target flips, so the JOINT ΔS is net-POSITIVE (+0.30 S worse). **No seat predicted the collateral
would dominate the recovery by ~18×.** That is exactly what earned QA92 the slot: a genuine discriminator,
not a re-anchor. Deepest read CONFIRMED and SHARPENED: the frozen SegNet's ~85px-r50 ERF makes localized
paint a *neighborhood* perturbation, not a *local* one — you cannot inject a lane stroke without the
receiver re-reading the surrounding Road/Undrivable/Movable cells.

## §1 The harness (all [macOS-CPU advisory]; frozen CPU-torch SegNet = authority, NEVER MPS)

`experiments/ddm_qa92_carrier_discriminator.py` (seeded, resumable-per-chunk, atomic; n600 in 475 s;
custody `/Volumes/VertigoDataTier/pact/ddm_qa92_20260731/`). The contest-exact chain, reusing the
pa1r/fp1/TR1 deploy surfaces verbatim:

1. **BASE render** (`model.render_frame(idx)` → (384,512,3) float, MLX-CPU stream = the trainer's own
   `realized_gate`) → deploy **R** (`_torch_R_to_camera_uint8`: torch-bicubic up to 874×1164 → uint8) →
   frozen SegNet (`cpu_verdict_d_seg_argmax_batch`, chunk 120 / seg_batch 12) → base realized argmax +
   base d_seg. **Validation: base d_seg 0.0049411 == pa1r control_tail endpoint verdict EXACTLY** — the
   chain reproduces the parent's own realized number, so the discriminator is trustworthy.
2. **Erased super-nucleus set** (folds fp1's deferred per-component erasure item): GT Lane (`lstars==1`,
   comma10k class 1) → scipy 8-conn label → keep >5px (super-nucleus) → a component is **erased** iff
   <50% of its GT-Lane pixels are classified Lane in the BASE pass. Union = the target support T.
3. **AA COMPOSITE at camera res PRE-R (Daubechies binding requirement).** Support +1px binary dilation →
   **bilinear** upscale 384→camera to a soft alpha ∈ [0,1] (bilinear, not bicubic, so the matte never
   overshoots [0,1]; the soft edge IS the anti-aliasing). Composite onto the bicubic-upscaled base FLOAT
   camera frame (before uint8): `comp = (1−α)·base_cam_float + α·fill` → round/clamp → uint8. This puts
   the AA stroke edges at the 874×1164 camera resolution, then lets uint8 + the SegNet-preprocess
   downsample act on them — exactly what Daubechies required so tier-1 measures the receiver, not aliasing.
   - **Tier-1 (oracle / Wyner bound):** `fill = gt_f1` (the real GT camera RGB — the very pixels SegNet
     read to produce `lstars`).
   - **Tier-2 (flat realizable):** `fill = [77.43, 86.71, 118.53]` (fp1's SOLVED margin-optimal Lane
     prototype, `proto_solved[1]`) — what a 1-2KB parametric flat-stroke carrier would paint.
4. Each tier → R → uint8 → frozen SegNet → realized argmax → per-tier per-class flip counts, target/
   off-target flip counts, component-level recovery.

**Compositing-artifact control (airtight):** an IDENTITY-fill composite (fill = the base render itself at
the erased-Lane support, same alpha) is **bit-identical** to base (max |Δd_seg| = 0.0 over 6 pairs). ⇒
the compositing op introduces ZERO artifact; **100% of the collateral is real receiver physics**, not a
seam/aliasing/rounding artifact. (The Daubechies concern is discharged: the AA is correct AND the
remaining collateral is the genuine ERF response to injected content.)

## §2 The decisive picture (additive-S is the sole authority; no axis priority)

The falling rule the memo pre-registered runs cleanly:

- **O = 0.407 ≥ 0.25** ⇒ the carrier family does NOT *formally* close (a perfect oracle DOES recover >25%
  of the target flip-mass). verdict_scope stays FORMULATION — the family is not paradigm-killed.
- **F = 0.194 < 0.7·O = 0.285** ⇒ IF rung 2 fired, it would fire (e1) (textured/solve-seeded), not (b)
  (flat) — flat strokes capture only 48% of the oracle ceiling; the recovery needs textured content.
- **P·O = 0.017 S < 0.05 S (Contrarian bound, ADOPTED)** ⇒ **rung 2 is SKIPPED; the slot goes to the
  burn-4 charter directly.** The pool the carrier targets (P = 0.042 S) is already small — the
  continuation to ep499 recovered most of QA91's 0.134 bound for free — and the oracle recovers only
  0.017 S of it, below the bound.
- **The over-riding honest read (the collateral):** even the oracle composite is **net +0.300 S WORSE**;
  flat is +0.225 S worse. A real carrier would have to buy back ~0.22-0.32 S of collateral with a
  reconcile tail merely to break even, then net-recover ≤ 0.017 S. That is dominated. The receiver ERF
  physics that fp1 measured on the ALL-FLAT frame **generalizes to the localized paint-on-texture case**:
  you cannot inject Lane structure onto this render without the SegNet re-classifying its neighborhood.

## §3 What this re-prices (inputs to burn-4)

1. **QA91's pool is REFINED down.** The "≈0.134 S recoverable" was an *upper* bound (total super-nucleus
   Lane mass at the ep399 burn endpoint). At the control_tail ep499 endpoint the ACTUAL still-erased
   super-nucleus flip-mass is **P = 0.042 S** — the continuation (rung-1's physics, running before it) has
   already recovered ~0.09 S of the pool for free. The remaining pool is smaller than the plan assumed.
2. **The birth arm in burn-4 defaults to KD-from-birth / plain-continuation, NOT paint.** Output-side
   compositing (b)/(d) is measured net-negative even at the oracle; the (e1) solve-seeded path (which
   RE-RENDERS rather than composites, staying on the manifold) keeps a distinct premise — but its priority
   drops: the pool it would target is 0.042 S and the receiver ERF is the same wall it must cross by
   re-rendering, not compositing. (e1)'s falsifier (survival <50% ⇒ closes) is unchanged and it is now
   gated behind burn-4, not rung 2.
3. **The carrier (b) build gate now carries a hard collateral clause:** any Lane carrier must include a
   reconcile tail proven to absorb ≥0.22-0.32 S of ERF collateral, or it is net-negative by construction.
   Given P = 0.042 S, that is a ~5-8× unfavorable ratio at the oracle ceiling — the MDL bet MacKay priced
   (~100× favorable IF QA92 holds) does NOT hold: QA92 does not hold at the JOINT level.

## §4 verdict_scope ledger

- BASE d_seg 0.0049411 (== pa1r endpoint): **MEASURED** (n600, frozen CPU-torch SegNet authority).
- P = 0.042 S erased super-nucleus pool: **MEASURED** (exact scipy 8-conn on gt_n600 lstars vs base
  realized argmax; refines QA91's 0.134 upper bound).
- O = 0.407 / F = 0.194 recovery fractions; P·O = 0.017 S: **MEASURED** (n600, both tiers, AA composite).
- collateral +0.317 (oracle) / +0.233 (flat); JOINT ΔS +0.300 / +0.225: **MEASURED**; artifact-free
  (identity-fill control bit-identical). Metric identity JOINT = −P·O + collateral verified exact.
- Rung-2 route = SKIP-to-burn-4 (P·O < 0.05): **MECHANICAL** per gc12 §4 (Contrarian bound adopted).
  Carrier family (b)/(d): NOT formally closed (O≥0.25) but not worth a dedicated arm at material size;
  verdict_scope FORMULATION (paint-on-texture Lane compositing @ control_tail ep499); family/paradigm
  untouched. (e1) distinct premise preserved but priority-dropped, gated behind burn-4.

## §5 STORES-CONSULTED (recall-first, path+sha)

gc12 `.omx/research/ddm_gc12_wall_branch_convocation_20260731.md` (b4d317538d; §4 QA92 charter +
falsifiers + Daubechies AA-binding dissent + Contrarian ΔS≥0.05 bound) · fp1
`.omx/research/ddm_fp1_class_field_projection_20260731.md` (c90254b5ef/8870930cc4; receiver floor
0.008305 all-flat, "receiver is the wall", QA91 super-nucleus inventory + prototypes.npz proto_solved[1]
= [77.43,86.71,118.53], custody `/Volumes/VertigoDataTier/pact/ddm_fp1_20260731/`) · pa1r
`.omx/research/ddm_pa1r_pool_a_race_20260730.md` (fdb48e2c26; parent ckpt control_tail
`stage_seg_trunk_tau_final.npz` sha a2dc86b8a8982456…, d_seg 0.0049411, S 0.67325) · deploy surfaces:
`experiments/train_witness_realized_through_R_mlx.py` (`_torch_R_to_camera_uint8`,
`cpu_verdict_d_seg_argmax_batch`) + `experiments/train_tr1_partition_renderer_mlx.py`
(`render_frame`, `realized_gate`) + `experiments/ddm_fp1_class_field_projection.py` (`load_frozen_module`) ·
gt cache `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (sha cf8d83605d2198ef…; gt_f1 camera RGB +
lstars + margins) · frozen-scorer laws (SegNet reads REGIONS, ERF r50~85px, comma10k class order
[Road,Lane,Undriv,Movable,MyCar], Gibbs/step-native). Closed forks re-checked, none re-opened: (b)/(d)
are the fp1-reformulation family MEASURED here; (e1) is NOT nv1's null-snap (opposite sign) and is gated
behind burn-4; fp1's flat-field floor is the ALL-FLAT case this composites-on-texture beside.

## §6 custody
`/Volumes/VertigoDataTier/pact/ddm_qa92_20260731/`: `qa92_verdict.json` + 5× `chunk_*.npz` (per-pair
accumulators) + `qa92_custody_manifest.json` (ckpt/gt/code sha256 + deterministic rebuild command) +
`run.log`. Certified-rebuildable (frozen ckpt + gt cache + seeded harness, no RNG). No `/tmp` in evidence.

**Task #801 completion evidence lives here + FEED-qa92.** Pointer **0.1910828242 [contest-CPU] UNMOVED.**
[no-triality] [p0-ledger-ok]
