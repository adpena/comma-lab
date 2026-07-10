# Texture trunk (T of W=(G, ξ, T)) — the band-designed per-class stationary texture trunk (#395 P0)

**Operator GO 2026-07-10** *"Yes please pursue that texture trunk as a p0."* **$0 local BUILD; NO training
launches** (owed-16 A/B owns the machine — the OFF arm is LIVE, run dirs READ-ONLY). **Pointer contest-CPU
0.19110 UNMOVED — everything here is MEANS** `[macOS advisory · research-signal · NON-PROMOTABLE]`. Remaining
gap to sub-0.15 = **0.0411 S**.

**Module:** `src/tac/boundary_math/texture_trunk.py` (+ 31 tests). **Trainer wire-in:** the LEVELSET entry
point `experiments/train_levelset_witness_realized_through_R_mlx.py` (`--texture-trunk`, default-OFF,
byte-identical when off). **DSL:** `tac.witness_dsl.TextureTrunk` lever (composable `--dsl-lever TextureTrunk`;
0 orphaned flags). **Equations:** `texture_trunk_band_is_stem_passband_v1` (registered). **DAG:** FEED-textrunk.

---

## 0. Answer-first

The live witness composes RGB as `sigmoid(softmax(phi/τ)@palette + out_tex(h))·255` where
`out_tex = Linear(hidden→3)` is a LINEAR readout of the PARTITION trunk's hidden features. Its expressible
textures are bounded by the partition trunk's frequency content, and the partition objective pulls those
features SMOOTH off the boundary while texture wants IN-region oscillation (the P12 antagonism). The
mod32cap witness (d_seg 0.0048) already beats the flat-paint floor (0.0416) **8.7×** via a MODEST `out_tex`
(‖W‖~0.06) — texture is worth a lot AND the linear head is capacity-starved for it.

**The build:** ADD a TINY SEPARATE texture trunk T whose BASIS is texture-native — a fixed rule-118 Gabor
bank pinned to the MEASURED SegNet stem pass-band `[period-4 Nyquist .. band_hi=8]` render-px, per-class
fitted coefficients (375 counted at the default band; ~2.5e-4 S uncoded), PLACED through the partition
softmax masks, trained JOINTLY through the seg loss. T does NOT read G's hidden state (that is the point:
DECOUPLE the two objectives so G stays smooth off-boundary and T oscillates in-region). Keep G untouched.

**Position vs the Unit A refutation (#394, `roadlane_grating_composition_refuted_v1`, the load-bearing
caveat):** Unit A MEASURED that a period-4 grating as a region-wide HARD FILL is ANTAGONISTIC (+0.228 d_seg:
SegNet is context-dominated; injected region-wide edges flip argmax wholesale). This trunk is the OPPOSITE of
that refuted move — it is **texture-as-LEARNED-modulation** (the family that SURVIVES: the mod32cap out_tex
existence proof), never a fixed grating TARGET, never injected. Its three guards are exactly Unit A's cure:
(a) **joint seg-loss gradient-through** punishes any texture that flips argmax (the #300 invariant); (b)
**softmax-mask placement** is scene-placement-aware BY CONSTRUCTION (no region-wide push); (c) **annulus
attenuation** damps texture in the boundary annulus where flips are cheap. The A/B carries Unit A's row as
its P12/P7 falsifier: if the trunk, even trained jointly, drives d_seg UP, the band-native basis makes
wholesale flips too easy and the arm is refuted.

---

## 1. The band is DICTATED BY MEASUREMENT (clause-B minimal-dim, not a guess)

`tac.through_r.stem_perception` read the frozen SegNet EfficientNet-B2 `conv_stem` verbatim
(`segnet_texture_perception_20260710.md`, `segnet_stem_nyquist_alias_wall_v1`): stride-2 ⇒ the finest
surviving texture PERIOD = `2·stride = 4` seg-input px (below it, texture aliases away before the first
MBConv — dead). The through-R price list (`segnet_through_r_texture_price_list_v1`): the ONLY texture that
flips Road/Lane out of the flat Undrivable basin is a **period-4** high-contrast luminance grating;
**period-2 aliases, period-8/16 read flat**. So the trunk's bank support = `[4, band_hi=8]` render-px
(the render grid IS 384×512 = the seg grid, so render-px ≡ seg-px for the band). `TextureBandSpec.__post_init__`
REFUSES any out-of-band period; `band_limit_report` proves every feature's 2-D FFT peak ∈ band (measured
3.99–8.02 px, all-in-band). The band = the stem transfer pass-band; the counted coeff dim (F·K·3) IS the
geometry's bound. `Law: texture_trunk_band_is_stem_passband_v1` (registered).

**Design:** F = periods{4,6,8} × orient{0,45,90,135} × phase{cos,sin} = 24. `W_tex (F,K,3)` + `bias (K,3)`
= 375 counted params. Forward: `texture[...,p,:] = Σ_k soft[...,p,k]·(bank@W_tex[:,k,:] + bias[k])[p]`,
optional `×g(margin)` annulus attenuation. Added into the pre-sigmoid `base + out_tex` term.

## 2. The #300 invariant (witness-OWNED gradient-through, NOT a compose-addend)

`W_tex`/`bias` are TRAINABLE MLX witness parameters receiving seg-loss gradient every step — the trunk trains
JOINTLY through the frozen SegNet, so the texture it paints CANNOT flip argmax (the loss punishes it — the
exact opposite of Unit A's un-gradiented injection). The deterministic Gabor bank is a FROZEN buffer keyed
`tex_trunk.bank_B` — the `_B` suffix is the trainer's canonical rule-118 marker, so `measure_witness_blob_bytes`
/ `_quantize_blob_from_flat` / `_load_decoder_params` ALL exclude it from the COUNTED byte-close blob
(regenerated free at decode from (H,W,spec)); only `W_tex`+`bias` are counted. VERIFIED in-process: the bank
contributes **0** counted bytes; default-OFF is byte-identical (submodule not created ⇒ params/EMA/ckpt/
byte-close unchanged, the `film_per_layer` idiom). Resume arch-drift guard extended with a texture-trunk hint.

## 3. Training dynamics — the grokking-scaling input (arXiv 2606.30388; CITED)

`.omx/research/papers_checked_grokking_scaling_2606.30388_20260710.md`. A fresh T beside a converged G is a
shell-core setup (T's coeffs on the random-init shell; G in its generalization core). Consequences:
(1) a warm-start deployment risks a grokking DELAY — the `TextureTrunk(window=…)` warm-start-epochs param
must budget enough tail to clear the abrupt transition (judge T at the settled checkpoint, not mid-transient
— the EMA-shadow-lag / early-run-rise confound sister); (2) T's coefficient group wants its OWN lr/decay
derived from the paper's stopping-time scaling laws, not G's converged small lr (OWED: read the PDF exponents
before the warm-start arm; the from-scratch arm does not need it); (3) engage T at/after G's island-birth via
the #315 event-triggered schedule (geometry must exist before it can be textured), not a naked fixed epoch.

---

## 4. THE MATCHED-COUNTED-BYTES 3-ARM A/B (spec; QUEUED behind owed-16 + #385 GO)

**Question:** does a texture-NATIVE band basis beat the partition-trunk-bounded linear/MLP head at EQUAL
counted bytes? **All arms n600, realized-through-R, byte-closed d_seg (never a proxy); ADVISORY until a
byte-closed contest-CPU exact row.** Gated behind owed-16 freeing the machine (the governor REFUSES today:
system-admission 139.3 > ceiling 115.7 GiB — CONTAINMENT, info not obstacle).

| arm | head | counted Δbytes | isolates |
|---|---|---|---|
| **A1 linear (control)** | current `out_tex = Linear(hidden→3)` | 0 | the byte-floor reference |
| **A2 widened MLP head** | `out_tex = MLP(hidden→H'→3)`, H' sized so ΔW ≈ 375 | ≈ +375 | MORE capacity on the PARTITION basis |
| **A3 texture trunk** | `--texture-trunk` (this build), 375 coeffs | ≈ +375 | band-NATIVE basis at MATCHED bytes |

A2 vs A3 is the load-bearing comparison (matched bytes ⇒ isolates BASIS, not capacity). A1 is the "is the
extra capacity worth ANY bytes" floor. **A2 needs a small wire-in (OWED before the A/B):** add `--out-tex-hidden H'`
(default 0 = current linear, byte-identical) to `build_levelset_rgb_witness` — replace `self.out_tex = Linear(hidden,3)`
with a 1-hidden MLP when H'>0; pick H' so `hidden·H' + H'·3 + H' + 3 ≈ 375` (at hidden=256 ⇒ H'≈1 is too
coarse; use H' from `H'·(hidden+4)+3 ≈ 375` → H'≈1–2; if the granularity is too coarse, match on a common
target of ~1.5 KB instead of 375 B and scale all three arms' budgets up together). This diff is ~15 lines,
same default-OFF/byte-identical idiom; NOT built here (P0 scope = the texture trunk).

**P2 noise-floor (BEFORE any winner call):** run A1 twice at matched seed/config (or read the run's own
seed-replica band) to establish the n600 d_seg noise σ; an arm "wins" only if `Δd_seg > 3σ`. Small-signal
near the goal → RELATIVE-not-absolute (MEMORY): report Δd_seg / remaining-gap-0.0411, not raw Δ.

**P7 falsifier (pre-registered):** A3 d_seg ≥ A1 d_seg (within noise) ⇒ the band-native basis does NOT help
— either capacity was not the limit, OR the antagonism (Unit A) leaks through the joint loss (the band basis
makes wholesale flips too easy). A3 ≥ A2 ⇒ band-native basis ≤ generic-capacity — the basis claim REFUTED,
texture is capacity-bound not basis-bound. verdict_scope on either = FORMULATION (this trunk formulation),
not the texture-lever FAMILY (the mod32cap out_tex existence proof stands).

**P12 composition rows (holistic facets, never a headline composite):**
- **vs self-orient / DirectionalBasis** (`--dsl-lever DirectionalBasis`): the directional BASIS reallocates
  Fourier frequency along the boundary tangent (a G-side, distortion-geometry lever); T is a texture-BASIS
  lever on the cell interiors. Different layers ⇒ compose; the row measures whether T's interior texture and
  the directional edge basis are additive or antagonistic (expected additive — disjoint spatial support).
- **vs the seed/condition arbiter v2** (`seed_condition_program_design_20260710.md`, CROSS-FINDING UPDATE):
  seed/condition operate on the EXISTING `out_tex` head (init + loss layers); T is an ARCHITECTURE layer
  (band-native basis). They stack: CONDITION's per-class-λ_c spectral-band bias BIASES T's coeffs toward the
  period-4 band; a transfer-SEED inits T's coeffs from a donor. The row: does CONDITION-on-T beat CONDITION-on-linear-out_tex?
- **per-class + pose + rate facets** always read separately (per-class d_seg vs anchors · island-birth ·
  d_pose vs need · rate), never a composite — the holistic-check-in binding.

**Pinned commands** (emit + validate today with `--dry-run`; drop `--dry-run` when owed-16 frees the machine
AND the base config's 3 naked-epoch schedule-governance triggers are resolved — a pre-existing base-config
gate, orthogonal to this lever):
```bash
# A3 texture trunk (from-scratch arch arm). --dry-run VALIDATES today (governor REFUSES the spawn: owed-16).
.venv/bin/python tools/launch_witness_run.py \
    --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
    --num-pairs 600 --epochs 1000 --dsl-lever TextureTrunk --dry-run
# A3 warm-start deployment (grokking-aware; needs the PDF lr/decay exponents first, §3):
#   ... --resume-from <converged-G-run-dir> --warm-start-weights-only --dsl-lever TextureTrunk
#   (TextureTrunk(window=<tail>) once the transition-time scaling is pinned)
# A1 control = the same command WITHOUT --dsl-lever TextureTrunk (byte-floor reference).
# A2 widened-MLP = A1 + --out-tex-hidden H'  (OWED: the ~15-line wire-in above).
```

---

## Canonical-vs-unique decision per layer

| layer | decision | rationale |
|---|---|---|
| module home `boundary_math/` | ADOPT_CANONICAL | sibling of `lever_b_generator` (MLX + numpy reference + dataclass + rule-118 `_B` table) — same idiom serves. |
| Gabor bank `_B` naming | ADOPT_CANONICAL | the trainer's `name.endswith("_B")` free-table convention — FORK here would silently COUNT 4.7M floats (a rate disaster). |
| compose-by-addition into `_compose_rgb` | FORK_PRINCIPLED | the whole point is a SEPARATE trunk decoupled from G's hidden features (P12 antagonism) — a shared-`out_tex` extension would re-couple. |
| default-OFF byte-identity | ADOPT_CANONICAL | the `film_per_layer` idiom (submodule not created when off) — required for a clean matched-bytes A/B. |
| DSL `Lever` factory | ADOPT_CANONICAL | config-orphan rule: a lever is not built until it is a `Lever` in the DSL (never a hand-added trainer flag). |
| annulus attenuation from softmax peak | FORK_PRINCIPLED | reuses the SDF-softmax already computed (no seg forward at compose time) — a live-margin seg-logit gate would be a chicken-egg. |

## Observability surface

- **inspectable per layer:** `band_limit_report(H,W,spec)` (per-feature FFT peak period) · `counted_bytes(spec)`
  (exact counted-coeff rate) · the frozen `bank_B` buffer + trainable `w_tex`/`bias` are readable from any ckpt.
- **decomposable per signal:** counted vs free bytes split (bank excluded, verified 0); per-class W_tex slices
  (`w_tex[:,k,:]`) read each class's texture independently; annulus gate is a separable multiplier.
- **diff-able across runs:** `w_tex` is a small (F,K,3) tensor — two runs' texture heads diff directly; the bank
  is deterministic (regenerates identically) so any drift is in the counted coeffs alone.
- **queryable post-hoc:** the equation `texture_trunk_band_is_stem_passband_v1` + the design memo pin the band;
  `__cfg_texture_trunk*` provenance scalars in the resume sidecar record the arch flags.
- **cite-able:** every claim tags MEASURED (band peaks, byte-exclusion) / DERIVED (band law, grokking application)
  / CITED (arXiv 2606.30388) / advisory.
- **counterfactual-able:** default-OFF byte-identity + the matched-bytes A2/A3 arms are the built-in counterfactuals
  (basis-vs-capacity); the P7 falsifier is pre-registered.

## Cargo-cult audit per assumption

1. **"texture wants period-4"** — HARD-EARNED (MEASURED price list `segnet_through_r_texture_price_list_v1`;
   the band floor is the measured stem Nyquist, not a JPEG/HEVC default). Unwind if a live A/B shows the trunk
   wants breadth beyond 8 px (would contradict period-8/16-read-flat — re-measure the stem).
2. **"a band-native basis beats the linear head"** — CONJECTURED (the mod32cap ‖W‖~0.06 modest-out_tex hints
   capacity-starvation, but the A2/A3 matched-bytes arm is what decides; NOT asserted). Pre-registered falsifier §4.
3. **"texture helps d_seg at all in composition"** — CONTESTED by Unit A (region-wide fill HURTS +0.228). The
   trunk's learned-modulation + joint-loss + mask-placement + annulus guards are the audited cure; the A/B
   re-tests it as jointly-trained modulation, NOT the refuted fill. HARD-EARNED caveat welded on.
4. **"stationary per-class texture (shared across pairs) suffices"** — CONJECTURED (Fable synthesis: T is a
   per-video texture PROCESS, not a per-frame map). Unwind path: if per-pair texture variation matters, the ξ
   phase-advection sub-lever (default-off, spec'd) adds per-pair phase — measured only if the stationary arm underfits.

## 18-shared-assumption profile (lite)

ADOPT_CANONICAL (serves): EMA-shadow-at-inference · archive.zip byte-close · eval_roundtrip-through-R ·
canonical scorer-preprocess · seeded/deterministic · resumable-per-stage. FORK_PRINCIPLED: the compose
path (separate trunk, not a shared head) · the texture basis (band-native Gabor, not partition-trunk
features). UNCLEAR_NEEDS_EMPIRICAL: whether the band-native basis beats generic capacity (the A2/A3 arm) ·
whether stationary-per-class suffices vs per-pair (the ξ-advection sub-lever). No score-affecting assumption
is silently inherited — the two FORKs are the substrate's reason to exist; the two UNCLEARs are the A/B's job.

---

## Triality legs
- **DAG:** FEED-textrunk (appended this landing).
- **DSL:** `tac.witness_dsl.TextureTrunk` lever (composable; 4 flags MAPPED, 0 orphaned — `completeness().unmapped`
  has no `texture-trunk` entry). Registered default-OFF with duty-to-measure = the §4 A/B.
- **Equations:** `texture_trunk_band_is_stem_passband_v1` (registered,
  `src/tac/canonical_equations/texture_trunk_band_20260710.py`; VERIFIED_VIA_SOURCE_INSPECTION; cites
  `segnet_stem_nyquist_alias_wall_v1`).

**#385 one-liner (for `DUAL_CHAIN_BRIEF_385_20260710.md`):** #395 texture trunk BUILT — band-designed per-class
stationary texture trunk (T of W=(G,ξ,T)) landed as `tac.boundary_math.texture_trunk` + LEVELSET trainer
`--texture-trunk` (default-OFF byte-identical) + DSL `TextureTrunk` lever + `texture_trunk_band_is_stem_passband_v1`;
375 counted coeffs on a rule-118-free stem-Nyquist Gabor bank, learned-modulation (NOT Unit A's refuted
grating-fill), guards = joint-loss + mask-placement + annulus; matched-bytes 3-arm A/B SPEC'd + queued behind owed-16.

**Pointer 0.19110 UNMOVED — this is a BUILT MEANS, not a score. The lever moves the pointer only through the
byte-closed matched-bytes A/B exact row.**
