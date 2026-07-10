# SEED / CONDITION PROGRAM DESIGN — driving convergence with the price-list-optimal texture (2026-07-10)

**Task (operator):** *"If we can at least use that to help seed then we have tools for applying optimal
gradient surgically across entire training run to drive convergence from there"* + *"Or condition."*
`$0 · design + build-spec · NO heavy launches (CONTAINMENT — owed-16 A/B owns the machine).`
**Axis `[macOS-CPU advisory · design-signal · NON-PROMOTABLE]`. Pointer contest-CPU 0.19110 UNMOVED —
everything here is MEANS; it moves only through a byte-closed `upstream/evaluate.py` n600 row < 0.19110.**

**Subagent** `seedcond-design-r2` (predecessor died at session limit; NO checkpoint rows existed → fresh).

---

## 0. Answer-first — recommended arm + the cheap arbiter

> The two operator framings (SEED vs CONDITION) are **not competitors — they are the two ends of ONE
> question**: *can the price-list-optimal period-4 luminance grating be installed as a WEIGHT INIT of the
> `out_tex` head (SEED, $0, one-shot), or must the coord-INR TRUNK be TRAINED toward the period-4 band over
> the whole run (CONDITION, a per-class-λ spectral loss)?* The MEASURED structured-init caveat already tells
> us which is likely, and a **$0 epoch-0 realized-through-R probe that needs NO machine** decides it cleanly.

**Recommended sequencing (MVP-first, CONTAINMENT-safe):**
1. **RUN THE SEED epoch-0 probe first ($0, no machine, no training)** — it is the decisive arbiter (§5.1).
2. **Branch A** (seeded `out_tex` realizes well below the flat floor at epoch 0) ⇒ **SEED is the arm**;
   BOTH becomes the aggressive stack. Cheap, no whole-run loss overhead.
3. **Branch B** (seeded `out_tex` does NOT realize at epoch 0 — the trunk can't emit period-4 from a
   readout init) ⇒ **CONDITION is the arm**: the trunk must be trained toward the band by the per-class-λ
   spectral loss, actuated by the costate controller across the whole run. The warm-start fine-tune probe
   (§5.2, needs the machine) is its confirmer, spec'd for **after owed-16 frees the machine**.

The single load-bearing empirical fact this rests on is already MEASURED: the structured-init block's caveat
(trainer L3430–3432) says a structured **phi** (partition) init gave **NO epoch-0 realized win** because
`out_tex` was random and *"the render is texture-dominated at init."* That is the price-list thesis restated
inside our own trainer — and it is exactly the half the SEED-texture arm supplies.

---

## 1. The mechanism, read verbatim (where texture lives in OUR witness)

The levelset witness composes RGB at `_compose_rgb` (trainer L1343–1352):

```python
phi  = out_sdf(h)                       # (...,K) SDF fields  → the PARTITION
tex  = out_tex(h)                       # (...,3) texture head
base = softmax(phi/temp) @ palette      # (...,3) per-class colour (SDF-pinned)
rgb  = sigmoid(base + tex) * 255        # texture ADDS on top of the palette base
```

- **`palette` (5,3)** = per-class colour base (anchored to per-class GT-mean logit; learnable).
- **`out_tex` = Linear(hidden→3)** = the per-pixel additive texture. Read from mod32cap EMA-BEST it is
  **ACTIVE but MODEST** (‖W‖ mean 0.06) — the witness spends most budget on the flat palette base and adds
  a small texture perturbation (FEED-texture DIAGNOSTIC-3). This is the head the price list says must carry
  a **period-4 stem-Nyquist luminance grating on Road/Lane** and **flat on Undrivable/MyCar/Movable**.
- **Render res = 384×512** (argparse default, trainer L9981–9982) = the **seg-input / L\* resolution**.
  The price-list tiles were pushed through the REAL R (`render_grid_to_camera_uint8`) at this **same 384
  grid**, so a period-4 grating written directly into the 384 render grid is exactly the winner that was
  measured (§4 rate). **No D-pre-imaging is needed for the LUMINANCE arm** (render grid == price-list tile
  grid). D-pre-imaging (§frame0/UNIT-C, `Dᵀ`) is required ONLY for the optional chroma-HF sub-lever, whose
  384-plane luma-null pattern must be pre-imaged if ever rendered at a finer grid.

**Class self-detection (NON-NEGOTIABLE, MEMORY canonical-order rule):** the target texture per class MUST be
placed by **self-detected spatial/static signature** (`identify_island_classes` / the static-core detector
already used by structured-init + area-constraint), NEVER a hardcoded class index. Comma10k canonical order
is `[Road, Lane, Undrivable, Movable, MyCar]`; luma-sorting is FORBIDDEN (bit us 3×). Road = bright-on-dark
period-4; Lane = dark-on-bright period-4 (polarity MEASURED, price list §2).

---

## 2. ARM SEED — initialize `out_tex` (and palette) with the price-list optimum

**What.** Extend the existing `structured_init` block (trainer L3433+) — which today pretrains ONLY
`model.sdf`/`out_sdf` toward the static-core phi target and leaves `out_tex` random — to **also** fit
`out_tex` (+ optionally nudge `palette`) so the epoch-0 RENDER realizes the price-list-optimal texture:
period-4 luminance grating on self-detected Road/Lane regions, flat basins on Undrivable/MyCar/Movable.

**How (two candidate fit mechanisms, pick per the epoch-0 probe):**
- **(a) least-squares readout fit** — solve `out_tex.weight` s.t. `sigmoid(base + out_tex(h)) ≈ grating_target`
  on the frozen random trunk features `h`. Cheapest; tests whether the *readout alone* can emit period-4.
- **(b) short subsampled Adam pretrain of the trunk+out_tex** toward the grating target (the SAME pattern
  structured-init already uses for out_sdf: *"the trunk must be ADAPTED"*). Tests whether a *few hundred
  steps* installs the periodicity. This is the SEED/CONDITION boundary made concrete — if (a) fails and (b)
  succeeds in ~600 steps, the answer is "trunk-trainable but not readout-installable" (a weak-CONDITION). # MAGNITUDE_DISMISSAL_OK: "weak-CONDITION" is a BRANCH LABEL in the P7 falsifier tree (names which arm the probe outcome selects — the short-pretrain SEED variant), not a magnitude-based defer/kill; nothing is dismissed on absolute ΔS here — every branch routes to a build arm.

**#300 starvation guard (NAMED — the load-bearing distinction).** The killed lever was **seed-COMPOSE**:
compose a FIXED external seed render INTO the witness render → the witness free-rides the seed → island
gradients STARVE → forced anneal seed→0 (`seed_compose_weight_at_epoch`, trainer L2001; BUILD #300). SEED
here is **structurally different**: it initializes the witness's **OWN `out_tex` weights** — the witness
OWNS the texture, the seg-loss gradient flows THROUGH `out_tex`, there is no external addend to free-ride on.
This is the **same non-starving mechanism as structured-init-of-out_sdf** (which does not starve). The
invariant guard: **seed WEIGHTS, never compose a fixed render**; and any downstream loss reads the
witness-alone logits `_slog_wa` (the #300 `--witness-alone-island-loss` guard) so nothing free-rides.

**Second-order SEED risk (P11 bounded-harm):** a good `out_tex` seed could be a LOCAL optimum that blocks a
better one, OR (benign) could zero the seg-gradient because the seed already solved it (the goal). The P11
guard is **measured-through-recovery**: the arm is admitted only if `seed→train` ≥ `scratch→train` at equal
epochs (the warm-start probe §5.2). A seed that wins epoch-0 but loses after training is a labelled
LOCAL-TRAP negative, not a kill of the paradigm.

**Trainer touch-points (exact):** `structured_init` block L3433–3540 (add an `out_tex`/palette fit stage
after the out_sdf pretrain, gated by a new `--seed-tex` flag); the grating target built from the price-list
generator (`tac.boundary_math.lever_b_levelset_generator.stem_nyquist_max_freq_cycles_per_unit` +
`src/tac/through_r/stem_perception.py`'s tile generator, both landed) + the static-core / island
self-detector for class regions. EMA is created AFTER init (L3429 note) so the shadow starts at the seed.

**Byte accounting.** SEED adds **0 archive bytes** (a weight init; the archive ships the TRAINED weights).
The period-4 grating STRUCTURE the trained `out_tex` learns is rule-118 FREE *for v8* (generic parametric
family); in the witness/mod32cap path the `out_tex` weights are simply part of the counted decoder. SEED
does not by itself change the counted/free split — it changes the trajectory/optimum. Any RATE win is
INDIRECT (a witness biased to the minimal price-list texture may hit the same d_seg with a smaller
`out_tex`/fewer code dims) and is **MEASURED at byte-close, never asserted**.

---

## 3. ARM CONDITION — a per-class-λ spectral-band loss across the whole run

**What.** A loss-side prior that biases the witness's REALIZED (or witness-alone) render spectrum on
self-detected Road/Lane toward the **period-4 stem-Nyquist band**, and toward **flat** on the three
flat-winnable classes — applied across the WHOLE run (operator: *"surgical gradient across entire training
run"*), actuated by the costate controller's **per-class λ_c** machinery.

**Why the loss-side, not just the seed.** The smooth-stage curriculum prior (which MEMORY records as
d_seg-RAISING) is *exactly wrong* for Road/Lane: it low-passes away the period-4 signal the stem needs.
CONDITION replaces that prior on those classes with a band-TARGETED one. Where SEED installs the answer at
init, CONDITION trains the trunk to KEEP emitting the band — the correct arm when the trunk cannot express
period-4 from a readout init (Branch B).

**Exact canonical template (the CONDITION arm is a structural sibling of an already-landed lever).** The
Chan-Vese area constraint (trainer L5066–5086 loss; L4544–4585 setup) is the exact pattern:
`_area_lambda = {class: lambda_c}` **DERIVED-LIVE** from GT areas, per-class term
`E_area,c = (λ_c/2)·relu(m_c − A_GT_c)²`, consumes the SHARED realized seg forward, default-off ⇒
byte-identical, backed by a canonical equation. CONDITION mirrors it with a **spectral** quantity:

```
E_band,c(θ) = (λ_c/2) · [ w_hi · offband_power_c(render) + w_lo · onband_deficit_c ]      # Road/Lane
E_flat,c(θ) = (λ_c/2) · highfreq_power_c(render)                                          # U/MyCar/Movable
```

- `offband_power_c` = fraction of the class-region render power OUTSIDE the period-4±Δ band (measured by a
  fixed DFT / a period-4 matched filter on the render restricted to the self-detected class mask).
- `onband_deficit_c` = shortfall of power INSIDE the period-4 band (drives Road/Lane TOWARD the band).
- `flat` term penalizes any HF power on the basin classes (spend texture budget only where it flips argmax).
- **`λ_c` is the costate actuator.** Wire it to the SAME per-class-λ surface as the area constraint /
  ladder-island homotopy (#303 costate controller, `ladder_island_homotopy` + `_area_lambda`). The
  duty-to-measure queue ranks Road/Lane high (they carry the negative flat floor). This is APPARATUS reuse,
  not a new controller.

**Whole-run application (P11 anneal discipline).** Unlike seed-compose, there is nothing to anneal to zero
here — the band prior is a STANDING objective. The P11 anneal applies to the λ_c RAMP (coarse→fine
engagement, the same shape the area/persist ramps use), with a **pre-registered payoff** (Δd_seg on Road/Lane
through byte-close) and **bounded harm** (a per-class-λ cap so the band term never dominates the CE/argmax
signal — the L4 confound "term_domination >40% of total loss" alarm applies directly).

**Trainer touch-points (exact):** loss body L5066–5140 (add the `E_band`/`E_flat` term next to
`area_constraint` / `code_spectral`, reading `_slog_wa` witness-alone logits + the render); setup block
L4544–4585 (derive `λ_c` live, self-detect Road/Lane); the period-4 band constant from
`stem_nyquist_max_freq_cycles_per_unit` (landed). Needs a NEW canonical equation
`segnet_period4_band_conditioning_v1` (registered WITH the build, area-constraint precedent).

**Byte accounting.** CONDITION adds **0 archive bytes** (loss-side prior; ships trained weights). Same
indirect-rate logic as SEED, measured at byte-close.

---

## 4. ARM BOTH — SEED + CONDITION with anneal discipline

Seed `out_tex` at init (§2) AND hold the band prior across the run (§3). BOTH is the natural stack IFF the
epoch-0 probe (§5.1) shows the seed realizes (Branch A) — then the band prior PRESERVES the seeded band
against the smooth-stage low-pass instead of installing it from scratch. P11 discipline: pre-registered
payoff (BOTH ≥ max(SEED, CONDITION) at equal epochs through byte-close), bounded harm (λ_c cap +
measured-through-recovery), no silent regression (the L4 immune-system liveness + term-domination alarms).

**Composition sign hypotheses (P12) — see §6.**

---

## 5. Decisive gates (P7 falsifiers + P2 noise-floor)

### 5.1 THE ARBITER — SEED epoch-0 realized-through-R probe ($0, NO machine, NO training)

The cheapest, most decisive gate, and it does **not contend with owed-16** (a single forward eval, not a
training run). Build the price-list-optimal `out_tex` target; fit it by mechanism (a) least-squares readout
and (b) 600-step subsampled Adam; measure **realized d_seg through R** (via `tac.through_r.measure_through_r`,
bounded-n first then n600) at **epoch 0, no further training**.

- **P2 noise-floor:** the reference arms are the MEASURED flat-paint floor **d_seg 0.0416** (all-palette,
  n600) and the random-`out_tex` structured-init realized **0.586 ≈ random 0.506** (the caveat). A seed
  "win" must clear the flat floor by a margin ≫ the n600 CPU-argmax tie noise (≈8.5e-9, negligible) — use a
  **≥2× cut below 0.0416** as the significance bar (P-philosophy waterfill-or-justify; relative-to-gap: a
  cut to ~0.02 is ~50% of the flat-floor slack).
- **P7 falsifier — Branch decision:**
  - readout-fit (a) realizes ≤ ~0.02 ⇒ **SEED-readout WINS** (the head alone emits period-4) → SEED arm,
    cheapest possible.
  - (a) fails but Adam-fit (b) realizes ≤ ~0.02 in ~600 steps ⇒ **trunk-trainable, not readout-installable**
    → weak-CONDITION (short pretrain suffices) → SEED-via-short-pretrain arm.
  - BOTH (a) and (b) fail to clear the flat floor at epoch 0 ⇒ **SEED provides no epoch-0 win** → the trunk
    needs whole-run band training → **CONDITION is the arm** (§3), confirmed by §5.2.
- **Verdict scope:** FORMULATION (an epoch-0 realized probe falsifies *installability*, not the price-list
  paradigm — the price list is already MEASURED-through-R true).

### 5.2 CONFIRMER — warm-start fine-tune probe (needs machine; SPEC ONLY, run AFTER owed-16)

Resume from mod32cap EMA-BEST (`levelset_n600_witness_mod32cap_20260706T115554Z`); run K∈[100,300] warm-start
epochs per arm {SEED-init, CONDITION-λ-band, BOTH, and a CONTROL = plain warm-start}; measure Δd_seg through
the **byte-close decode** (n600, the AXIS-9 authority — a review SEAL is invalid on a borrowed/unrun number).

- **Resumability/CONTAINMENT (P0):** per-stage checkpoints, EMA-shadow, atomic writes, `--resume-from`;
  governed launcher only; DO NOT launch while owed-16 holds the machine.
- **P2 noise-floor:** the CONTROL warm-start arm IS the matched control — any arm must beat CONTROL by more
  than the run-to-run Δd_seg dispersion (measure 2 seeds of CONTROL for σ; the L4 liveness stamp on every
  verdict row guards against a frozen-run false green).
- **P7 falsifier:** an arm whose byte-close Δd_seg ≤ CONTROL + σ ⇒ that arm REFUTED at this formulation (not
  the paradigm). SEED that wins epoch-0 (§5.1) but ≤ CONTROL here = LOCAL-TRAP (P11 measured-through-recovery
  caught it). CONDITION that fails here ⇒ the period-4 band prior does not survive the joint objective ⇒
  route to the v8 explicit-grating-primitive carrier instead (where structure is free, not learned).

---

## 6. P12 composition rows vs the active lever set

Active levers currently on the machine / in the DSL: `directional_basis` (freq-along/across, self-orient),
`structured_init` + `lane_prior_phi1` (paint), `area_constraint_birth` (per-class λ), `ladder_island_homotopy`
(per-class λ homotopy), margin-saliency (#141), the smooth-stage curriculum.

| pair | predicted sign | rationale (pre-registered; measure to confirm) |
|---|---|---|
| SEED-tex × structured_init/lane_prior_phi1 | **COMPLEMENTARY (synergy)** | structured_init seeds the PARTITION (out_sdf); SEED-tex seeds the REALIZATION (out_tex). Together they supply the caveat's missing half → an epoch-0 realized partition. Highest-value stack. |
| CONDITION-band × directional_basis | **SAME AXIS — redundancy OR reinforcement (MUST A/B jointly)** | the directional basis allocates Fourier features along the tangent; the band prior targets the OUTPUT period-4 spectrum. Overlapping frequency axis → could double-count or reinforce. P12 joint A/B required before either claims its isolated Δ. |
| CONDITION-band × area_constraint | **ORTHOGONAL by construction** | area = per-class MASS (spatial integral); band = per-class SPECTRUM. Disjoint loss quantities; stack additively (same argument as the frame0 orthogonality proof). |
| SEED/CONDITION × costate per-class λ (#303) | **not composition — the HOME** | CONDITION IS actuated by the costate λ surface; the duty-to-measure queue ranks Road/Lane. Fold into the controller, don't parallel it. |
| SEED-tex × smooth-stage curriculum | **ANTAGONISTIC on Road/Lane** | the smooth prior low-passes the period-4 seed away (MEMORY: smooth-stage RAISES d_seg). CONDITION exists partly to counter this; SEED alone under an unmodified smooth stage may DECAY. Measure the seed's survival across the smooth stage. |

---

## Canonical-vs-unique decision per layer

- **`out_tex` head fit (SEED):** **ADOPT_CANONICAL** — reuse the structured_init pretrain pattern
  (out_sdf → extend to out_tex); same subsampled-Adam-toward-a-target mechanism, same EMA-after-init
  ordering, same rule-118-free train-time-init framing. No fork.
- **Per-class-λ band loss (CONDITION):** **ADOPT_CANONICAL** — structural sibling of the Chan-Vese area
  constraint (`_area_lambda` DERIVED-LIVE, shared realized-seg forward, default-off byte-identical, canonical
  equation). Fork ONLY the quantity (spectral vs mass); reuse the λ_c derivation + costate wiring verbatim.
- **Class-region selection:** **ADOPT_CANONICAL** — the self-detecting static-core / `identify_island_classes`
  path (FORBIDDEN to hardcode indices; MEMORY canonical-order rule).
- **Grating target generator:** **ADOPT_CANONICAL** — `stem_perception.py` tile generator +
  `stem_nyquist_max_freq_cycles_per_unit` (both landed); no new geometry.
- **Chroma-HF sub-lever (optional):** **FORK_PRINCIPLED** — needs the exact-D pre-image (UNIT-C), which the
  luminance arm does not; defer to v8 chroma carrier, out of scope for the primary luminance SEED/CONDITION.

## Observability surface

- **Inspectable per layer:** epoch-0 realized-d_seg per fit mechanism (a/b) per class (§5.1 emits a
  MeasurementRow per class via `tac.verdicts`); the `out_tex` ‖W‖ + per-class render power-spectrum
  (on/off-band fraction) logged each verdict (score-neutral telemetry → DEFAULT-ON per the observability rule).
- **Decomposable per signal:** the CONDITION loss emits `terms_out["band_condition"]` per class (like
  `area_constraint`/`code_spectral`), so the band term's share of total loss is queryable (feeds the L4
  term-domination >40% alarm).
- **Diff-able across runs:** SEED-vs-CONTROL-vs-CONDITION warm-start arms are checkpoint-diffable (Δd_seg
  through byte-close, per-class); the epoch-0 probe is a standalone re-runnable JSON verdict.
- **Queryable post-hoc / cite-able:** `experiments/results/seedcond_epoch0_probe_<utc>/` (verdict JSON +
  per-class rows); each row cites (mod32cap ckpt sha, price-list eq id, fit mechanism, n).
- **Counterfactual-able:** the price-list target is parametric (period/polarity/colours/phase) → sweep
  without retraining; the λ_c band prior sweeps amplitude without re-seeding.

## Cargo-cult audit per assumption

- **"a good `out_tex` seed will realize through R at epoch 0"** — **CARGO-CULTED / UNMEASURED.** The
  structured-init caveat is evidence it may NOT (out_tex was random there, but a period-4 seed is the exact
  thing that was missing). Unwind = §5.1 IS the test; do not assume, measure.
- **"seeding out_tex weights won't starve like seed-compose (#300)"** — **HARD-EARNED (DERIVED).** Distinct
  mechanism: owned weights + gradient-through vs external free-ride addend; identical to non-starving
  structured-init-of-out_sdf. Guard named (weights-not-compose; witness-alone logits).
- **"period-4 at the 384 render grid needs no D-pre-image"** — **HARD-EARNED (DERIVED).** Render res 384 ==
  price-list tile grid == seg-input res; the price list measured the winner AT this grid through the real R.
  (Chroma-HF sub-lever is the exception — CARGO-CULT flagged, forked out.)
- **"the smooth-stage prior is wrong for Road/Lane"** — **HARD-EARNED (MEASURED, MEMORY: smooth stage RAISES
  d_seg; price list: sub-period-4 detail is aliased-away).** This is WHY CONDITION exists.
- **"λ_c band conditioning helps d_seg through byte-close"** — **CONJECTURED, pre-registered falsifier §5.2.**
  Not registered as a law until §5.2 measures it.

---

## Verdict + scope

**verdict_scope: FORMULATION-level design + build-spec — no kill, no score, no launch.** The SEED/CONDITION
program is DERIVED from measured anchors (price list §segnet_texture_perception; structured-init caveat;
Chan-Vese area-constraint template; #300 starvation lesson). Its load-bearing claim (installability of the
period-4 grating) is CONJECTURED with a $0 no-machine falsifier (§5.1). No config edited, no lever landed,
no launch fired — CONTAINMENT honored, owed-16 keeps the machine. **Pointer 0.19110 UNMOVED (means).**

## Triality legs

- **DAG:** FEED-seedcond (appended this landing).
- **DSL:** **OWED-pending-§5.1-arbiter (NOT orphaned — crisp stubs below).** Per MVP-first + the
  "don't build the loss-side lever before the cheap gate arbitrates" discipline, the CONDITION band-loss
  Lever + its canonical equation are named build items to land AFTER §5.1 picks the arm. The SEED Lever is
  a thin sibling of `DirectionalBasis`/structured-init and its stub is turnkey below. Landing them now would
  risk orphaning the wrong arm; the stubs keep them held-not-lost.
- **Equations:** SEED = no new law (weight init). CONDITION = `segnet_period4_band_conditioning_v1` OWED WITH
  the build (area-constraint precedent). Both existing laws consumed: `segnet_stem_nyquist_alias_wall_v1`,
  `segnet_through_r_texture_price_list_v1`.

### DSL Lever factory stubs (default-OFF; land AFTER §5.1)

```python
def SeedTexPriceList(fit: str = "adam", steps: int = 600) -> Lever:  # noqa: N802 — SEED arm
    """Seed out_tex (+palette nudge) with the price-list-optimal period-4 luminance grating on
    self-detected Road/Lane + flat basins on Undrivable/MyCar/Movable, via least-squares readout
    (fit='ls') or short subsampled Adam (fit='adam', steps). Sibling of --structured-init (seeds
    out_sdf); seeds WEIGHTS not a compose-addend (no #300 starvation). Default-OFF => byte-identical.
    Class regions self-detected (NEVER hardcode index). rule-118-free train-time init (ships 0 bytes)."""
    if fit not in {"ls", "adam"}:
        raise ValueError("SeedTexPriceList: fit must be 'ls' or 'adam'")
    return Lever("seed_tex_price_list",
                 overrides={"--seed-tex": True, "--seed-tex-fit": fit, "--seed-tex-steps": int(steps)},
                 notes="SEED: price-list period-4 grating out_tex init (Road/Lane) + flat basins")

def BandConditionPriceList(w_hi: float = 1.0, w_lo: float = 1.0,   # noqa: N802 — CONDITION arm
                           window: int | None = None) -> Lever:
    """CONDITION: per-class-λ_c spectral-band loss biasing self-detected Road/Lane render spectra to the
    period-4 stem-Nyquist band (offband penalty w_hi + onband deficit w_lo) and the 3 basin classes to
    flat. λ_c DERIVED-LIVE + actuated by the costate per-class-λ surface (#303, area-constraint sibling).
    Whole-run (window=None) per operator 'surgical gradient across entire run'. Reads witness-alone logits
    (#300 guard). Default-OFF => byte-identical. Law: segnet_period4_band_conditioning_v1 (OWED w/ build)."""
    ov = {"--band-condition": True, "--band-condition-w-hi": float(w_hi),
          "--band-condition-w-lo": float(w_lo)}
    return Lever("band_condition_price_list", overrides=ov,
                 epochs_delta=window,  # None => full-run standing objective
                 notes="CONDITION: period-4 band prior on Road/Lane, flat on basins; costate-λ actuated")
```

## Stores consulted
`segnet_texture_perception_20260710.md` (price list + stem Nyquist) · `fable_synthesis_texture_partition_20260710.md`
(W=(G,ξ,T), obligation matrix, texture-legibility gap) · `frame0_chromahf_dofs_20260710.md` (UNIT-C: f0
seg-freedom, chroma-HF 384-band-design + Dᵀ pre-image) · trainer `_compose_rgb`/`structured_init`/`seed_compose`/
Chan-Vese-area-constraint/costate-λ (verbatim L1343, L3433, L2001, L5066/L4544) · `stem_perception.py` +
`lever_b_levelset_generator.stem_nyquist_max_freq_cycles_per_unit` · MEMORY L12/L17/L71 · CLAUDE.md §WITNESS
CAPSTONE + §DEFAULT-OFF-orphan + §config-orphan-lever-registry · P2/P7/P11/P12 (`design_philosophies_eightfold_20260709`).
**Pointer 0.19110 UNMOVED (means).**

---

## CROSS-FINDING UPDATE (2026-07-10, appended by main — Unit A composition refutation lands ON the arbiter)

**#394 Unit A MEASURED (n600 through-R, matched arms, confound-free):** the period-4 grating as a
Road/Lane REGION FILL is ANTAGONISTIC in composition — Δd_seg **+0.228** vs scene-flat (Road 0.0165→0.9985,
Lane→1.0). Mechanism: SegNet is context-dominated; the price-list tile win is context-free; in-scene the
flat scene-mean colour already wins Road, and injected period-4 edges flip the argmax wholesale. Equation
`roadlane_grating_composition_refuted_v1`; memo `v8_geocoder_close_20260710.md`.

**Consequence for THIS program (P10/P12 applied):**
1. **The SEED arm's grating TARGET is pre-refuted at the region level.** The epoch-0 arbiter as spec'd
   (fit `out_tex` to the period-4 grating target → realized d_seg vs the 0.0416 floor) is now PREDICTED to
   fail by a stronger measurement than the probe itself — running it as spec'd would re-measure Unit A's
   row. Do NOT run the fit-to-grating form.
2. **What survives (the honest re-target):** the witness's own LEARNED texture (mod32cap `out_tex` ‖W‖~0.06)
   beats the flat floor 8.7× — so texture-as-learned-modulation works; texture-as-hard-grating-fill does not.
   The SEED arm re-targets to the WITNESS-LEARNED spectrum family (seed `out_tex` from a converged sibling's
   texture head — a transfer-seed, not a synthesized target); CONDITION (soft band bias on a TRAINED system,
   witness free to place/attenuate) becomes the PRIMARY arm, carrying Unit A's antagonism row as its P12
   composition entry: the bias must be scene-placement-aware, never a region-wide spectral push.
3. **Arbiter v2:** (a) transfer-seed epoch-0 probe (same $0 shape, target = donor witness out_tex) vs
   (b) CONDITION-only fine-tune probe — both still gated behind owed-16 freeing the machine. The flat-fill
   result also STRENGTHENS the null arm: scene-flat + geometry may be the Road carrier entire (v8 side).
verdict_scope of this update: FORMULATION (grating-fill formulation refuted; texture family alive via the
witness-learned existence proof).
