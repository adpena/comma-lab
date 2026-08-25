# Chroma attribution rung — v7.5.3 ladder design — 2026-07-10

`research_only=true` · **means != ends**: a MEANS (a registered-OFF ladder rung + its pre-registered
A/B). Pointer contest-CPU **0.19110 UNMOVED**; only a byte-closed `upstream/evaluate.py` n600 row
< 0.19110 moves it. NO launch, NO dispatch, NO run touched.

## STORES CONSULTED

- **CLAUDE.md §WITNESS CAPSTONE "Chroma is a d_seg lever (operator 2026-06-25 'Chroma too')"** — SegNet
  reads RGB ⇒ its argmax depends on chroma; chroma carries argmax-relevant signal in the codim-1 boundary
  annulus; chroma's PRIMARY value is d_seg; PoseNet reads YUV6 (chroma secondary); ANY witness d_seg
  verdict that ignored chroma is provisional.
- **`.omx/research/t5_crucible2/ADVISORY_v752_fresh_eyes_20260710.md` P0-4** — chroma is silently inherited
  by the sealed launch (v6/v7 base emits `--seg-chroma-boundary-weight 0.1 / -margin-band 1.0 /
  -start-epoch 450 / -start-event annulus_plateau`); the launch must "remove chroma from launch-1, or
  amend/reseal as a composed treatment **with its own prior measured add-back receipt. Do not inherit it
  accidentally.**"
- **`.omx/research/ADVISORY_evaluator_video_geometry_20260710.md`** — the obligation matrix: SegNet =
  frame1 RGB hard argmax at 384×512; PoseNet = 4 luma polyphases + AVERAGED U/V per 2×2 block (so chroma
  reaches PoseNet ONLY through the 2×2 block means); the exact pose-null = luma-null per pixel + zero-sum
  chroma per 2×2 block; recovered Pose Jacobian energy 95.97% luma / 4.03% chroma; frame0 is Seg-free.
- **`.omx/research/fullstack_fractal_optimal_synthesis_20260710.md`** — the v7.5.3 ladder (Δ4); this rung
  joins it.
- **`.omx/research/t5_crucible2/SPEC_v752_20260709.md`** — the ladder table (chroma = rung 3, "UNMEASURED
  add-back; ablation ≠ add-back — S5-N10"); §B P2 noise-floor + single-seed caveat discipline; the
  owed-16 decision-rule precedent (`_CRUCIBLE_V753_TRUNK_BASIS_DECISION_RULE`).
- **The #276 chroma-DOF probe RECALLED (not re-derived)** — see the receipt below.

## The prior measured add-back receipt (the one P0-4 demands)

**FOUND: commit `7d3be7e00` (witness #276/LEVER-4c), probe artifact `a3e9f0bd` (n96, 100% L*-match to
the frozen SegNet, 2026-07-03), DAG FEED-chromalever.** It is the **DOF-EXISTENCE receipt**, and it is
a **REMOVAL ABLATION, not an ADD-BACK**:

> Removing chroma (constant-luma render) FLIPS **7.54% of Lane→Road + 4.38% of Movable→Undrivable**,
> with **93.4% of chroma-flips in the `margin<1` fragile annulus** (→ 33.7% at margin<0.25). The flips
> are **LUMA-INDEPENDENT** (desat-only still flips 3.1%). SegNet margin-gradient energy is **78.8% luma /
> 21.2% chroma**. ⇒ chroma is a PROVEN INDEPENDENT d_seg boundary sharpener, ORTHOGONAL to the geometry
> levers.

Durable form: equation **`chroma_decides_lane_and_movable_at_annulus_v1`** (the MEASURED DOF) + the
mechanism equation **`chroma_boundary_annulus_match_hinge_v1`** (both registered in
`.omx/state/canonical_equations_registry.jsonl`); activation-ledger relative-significance seed row
`seg_chroma_boundary_276` (UNMEASURED / NEVER-FIRED / duty-to-measure) in
`.omx/state/lever_relative_significance.jsonl`.

**The critical S5-N10 distinction (why this receipt does NOT license accidental inheritance):** the
7.54%/4.38% ablation is a **WORTH** (how much the *default RGB render's* chroma is already doing), NOT the
**GAIN** of the add-back MATCH term. Removing chroma HURTS by that much; it does not follow that the
annulus chroma-MATCH loss IMPROVES d_seg by that much (or at all). The add-back ΔS is **UNMEASURED**.
Therefore the correct disposition per P0-4 is: **hold chroma OUT of the clean launch-1 trunk and put the
add-back MATCH term on its OWN registered-OFF ladder rung with a pre-registered A/B** — which is exactly
this rung. (The inherited base's `--seg-chroma-boundary-weight 0.1` pins are the launch executor's / amber
sibling's sealed concern; this rung does not touch them.)

## The rung, DERIVED not vibed

**What the chroma arm changes.** An additive, **0-byte** chroma-MATCH loss on the SHARED
realized-through-R render `f1` (the SAME render SegNet's forward and the `_signed` margin field come
from — no 2nd render, no 2nd SegNet forward):

```
L_c = w · mean_{ann} ‖chroma(f1) − chroma(GT)‖²      over the annulus 1[GT margin < band]
chroma := rgb − BT.601-luma          (LUMA-INVARIANT by construction)
```

It pulls the witness's OWN per-pixel rendered chroma (the existing `self.out = Linear(hidden,3)` head,
which HAS per-pixel chroma capacity but converges to a near per-class CONSTANT palette because the seg CE
only rewards argmax) toward the GT boundary chroma the constant palette cannot paint. It is **NOT** a
full-RGB reconstruction and **NOT** the render on/off toggle (`--chroma`/`--no-chroma`); it is the
LEVER-4c trained appearance-match term.

- **Trained chroma head vs inherited/fixed vs chroma-from-luma:** this rung is the *trained chroma head*
  formulation (supervise the per-pixel RGB head at the annulus). The "inherited/fixed" option is the
  accidental-inheritance P0-4 flags (rejected — not a clean attribution). "chroma-from-luma" is degenerate
  (the constant palette is exactly the from-luma habit that fails; the whole point is per-pixel deviation).

**Matched bytes / matched conditions.** The add-back adds **ZERO archive bytes and ZERO decoder params** —
it only redistributes what the existing RGB head learns. So the ON/OFF arms are **BYTE-MATCHED by
construction** (a pure-d_seg lever at zero rate cost — the cleanest possible attribution). Matched
conditions: both arms warm-start from the SAME pre-chroma checkpoint, same seed, same everything except
`--seg-chroma-boundary-weight`.

**Which score terms it can move (ADVISORY_evaluator obligation matrix):**
- **d_seg via frame1 RGB argmax** — the PRIMARY, intended axis. SegNet reads frame1 RGB; chroma is a
  genuine per-pixel argmax actuator at the codim-1 annulus. Frame0 is Seg-free ⇒ chroma there cannot help
  Seg.
- **d_pose ONLY through the 2×2 scorer-grid block-mean chroma** — WEAK, incidental. PoseNet keeps 4 luma
  polyphases + AVERAGED U/V per 2×2 block, and recovered Jacobian energy is 95.97% luma / 4.03% chroma.
  Both frames reach Pose but only frame1 reaches Seg. ⇒ optimize the add-back for **d_seg FIRST**; report
  any d_pose motion (before/after √(10·d_pose) on exact decoded bytes) as a side effect, never the target.
  A structural pose-null (if ever wanted) is luma-null + zero-sum chroma per 2×2 block — but this rung is
  a d_seg lever, not a pose-null construction.

**Pre-registered decision rule + noise floor (P2).** Encoded in
`_CRUCIBLE_V753_CHROMA_ADDBACK_DECISION_RULE` (witness_autoconfig.py). Three-way outcome, single source:

- **PAYS** — `IF realized Δd_seg(ON−OFF) < −max(noise floor, 3σ seed-band) AND surviving annulus flips
  shift toward higher GT-chroma agreement` → adopt at the winning `w` (own increment; 2nd-seed confirm
  elevates INSTANCE→FORMULATION before any family verdict).
- **WASH** — `ELIF |Δd_seg| ≤ noise floor` → OFF, `verdict_scope=FORMULATION` (the near-per-class constant
  palette already captures the argmax-relevant chroma; per-pixel match irrelevant at this operating point).
  **Do NOT escalate to "chroma is not a d_seg lever"** — the DOF is MEASURED-GREEN (a3e9f0bd); only the
  ADD-BACK formulation washed.
- **WORSE** — `ELSE` → OFF + register the antagonism (P10; the match perturbs the formed boundary the
  geometry levers already placed).

**Noise floor** (SPEC §B P2): a single-seed Δ is INSTANCE-until-floor-bounded; composed floor =
`max(cell granularity 8.477e-7 per corrected cell, the through-R proxy δ_R band, an UNMEASURED across-seed
3σ)`. The d_seg for the verdict is the **realized-through-R argmax on the frozen CPU-torch SegNet, n600
real-gt, byte-closed** — NEVER the through-R proxy for the verdict.

**verdict_scope pre-declared: FORMULATION.** This is ONE add-back formulation (post-hoc annulus
chroma-MATCH on the formed boundary, on THIS render + curriculum + operating point). A wash/negative
falsifies THIS formulation, not the family (chroma-as-d_seg-lever) and not the paradigm (the DOF stands).
Reformulation queue if it washes: (i) engage chroma EARLIER (before the palette hardens); (ii) per-class
chroma targets (Lane/Movable only — the 2 classes the DOF flips); (iii) a chroma CAPACITY-ROUTING term
(route mod-dim into the annulus) rather than a match loss; (iv) tighter band (0.25) to concentrate on the
33.7% knife-edge.

## The DSL lever (leg 1) — REUSED, not duplicated

`grep` of `src/tac/witness_dsl/` found the chroma surface ALREADY BUILT (no duplication):
- **`SegChromaBoundary(weight, margin_band, start_epoch, window)` → `Lever`** in
  `curriculum_dsl.py:3375` — emits ONLY real trainer flags `--seg-chroma-boundary-weight /
  -margin-band / -start-epoch`; fail-closed on `weight<0`, `margin_band<=0`, `start_epoch<0`.
- **`ChromaBoundaryGauge`** (gauge.py:503) with charts CHROMA_ACTIVE (`--chroma` default ON = the GREEN
  DOF baseline), LUMA_ONLY (`--no-chroma` = the ablation), ANNULUS_CHROMA_SHARPEN
  (`--seg-chroma-boundary-weight`).
- **Reference twin** `tac.boundary_math.chroma_boundary_match` (`bt601_chroma` / `annulus_mask` /
  `chroma_boundary_term` / `chroma_boundary_loss`) — bit-faithful, $0-testable, LUMA-INVARIANCE proven.
- **Trainer wiring EXISTS** (`experiments/train_levelset_witness_realized_through_R_mlx.py` ~L4285–4313,
  L4933; default `--seg-chroma-boundary-weight 0.0` ⇒ byte-identical when not composed). **No new trainer
  wiring was built** — the routing flags pre-date this rung.

**This rung's DSL-leg contribution = registering the EXISTING `SegChromaBoundary` as the v7.5.3 ladder rung
`chroma_annulus_addback_ab`** (`_CRUCIBLE_V753_LADDER`, registered-OFF, argv-inert). Duty-to-measure is
already tracked (activation-ledger seed `seg_chroma_boundary_276`, NEVER-FIRED).

## The ladder slot (deliverable 3)

Added to `crucible_v753_ladder()` as rung 11 (before the operator-GO terminal rung):
`("chroma_annulus_addback_ab", "SegChromaBoundary", "…")`. Ladder now = **11 rungs + operator-GO = 12
entries**. Registered-OFF: the default `crucible_v753(off)` config is BYTE-IDENTICAL (lever-name parity
with v7.5.2(self_orient=False) VERIFIED; `seg_chroma_boundary` NOT composed). Completeness unchanged (the
3 chroma flags stay mapped; no new unmapped/stale). Coordinated with the launch executor by touching ONLY
the v753 ladder surface (NOT the crucible_v752 launch path, NOT the live run dir, NOT amber).

## The A/B measurement plan (deliverable 4) — a PLAN, not a launch

Fire at the next machine window (post-launch, or on the live run's checkpoints via warm-start arms). All
argv below is the DSL-emitted form; NO heavy launch here (the machine is owned by the live run).

**Arms** (0-byte BYTE-MATCHED; warm-start from the SAME pre-chroma checkpoint `ckpt_C` = a stage boundary
where the annulus is FORMED, e.g. the annulus_plateau ~ep450 or a terminal ~ep675 of the live run):

| arm | chroma flags (emitted by SegChromaBoundary) | note |
|---|---|---|
| **OFF** | `--seg-chroma-boundary-weight 0.0` | chroma RENDER stays ON (#205 default); annulus MATCH inactive |
| **ON-a** | `--seg-chroma-boundary-weight 0.05 --seg-chroma-boundary-margin-band 1.0 --seg-chroma-boundary-start-epoch 0` | engage on warm-start |
| **ON-b** | `--seg-chroma-boundary-weight 0.10 --seg-chroma-boundary-margin-band 1.0 --seg-chroma-boundary-start-epoch 0` | sweep w |

Exact launch config (DSL, both arms same seed, `--resume-from ckpt_C`):
```
# OFF arm — the sealed v7.5.3(off) trunk, chroma add-back explicitly 0.0
crucible_v753 (trunk_basis=off)  --resume-from <ckpt_C>  --seg-chroma-boundary-weight 0.0
# ON arm — compose SegChromaBoundary(weight=0.05, margin_band=1.0, start_epoch=0)
crucible_v753 (trunk_basis=off)  --resume-from <ckpt_C>
    --seg-chroma-boundary-weight 0.05 --seg-chroma-boundary-margin-band 1.0 --seg-chroma-boundary-start-epoch 0
```

**Cells** (ep-matched checkpoints): `ckpt_C+150` and `ckpt_C+300`, both arms. At each cell:
**byte-close → realized-through-R d_seg on the frozen CPU-torch SegNet, n600 real-gt** (never the proxy
for the verdict) + the annulus-flip-shift diagnostic (did surviving flips move toward GT chroma?) + the
before/after √(10·d_pose) side-effect report on exact decoded bytes.

**Decision rule:** `_CRUCIBLE_V753_CHROMA_ADDBACK_DECISION_RULE` (PAYS / WASH / WORSE above). Noise floor
composed as above; single-seed ⇒ INSTANCE until a 2nd seed bounds variance. Cost: warm-start arms are
short (2×~300ep from a checkpoint) — well within a machine window; $0 CPU byte-close for the verdict.

## Triality

- **Leg 1 (DSL):** `SegChromaBoundary` factory (pre-existing, REUSED) registered as v753 ladder rung
  `chroma_annulus_addback_ab` in `_CRUCIBLE_V753_LADDER`; decision rule constant
  `_CRUCIBLE_V753_CHROMA_ADDBACK_DECISION_RULE`. 11 chroma-focused tests (33 in the suite) green, ruff-F
  clean.
- **Leg 2 (DAG):** FEED-chroma-rung appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **Leg 3 (equations):** disposition = **deferred-to-first-measured-row.** The DOF equation
  (`chroma_decides_lane_and_movable_at_annulus_v1`) and mechanism hinge
  (`chroma_boundary_annulus_match_hinge_v1`) are ALREADY registered (from #276); the rung's own **add-back
  ΔS** empirical anchor lands only when the A/B produces a byte-closed row (no new equation invented on a
  MEANS with no measurement).

## Honest gaps

1. The add-back ΔS is **UNMEASURED** — this rung delivers the FIREABLE arm + pre-registered rule, not a
   number. The 7.54%/4.38% is a removal ablation (WORTH), not the add-back (GAIN).
2. The warm-start `ckpt_C` epoch (annulus_plateau ~450 vs terminal ~675) is a plan parameter, not yet
   pinned to the live run's actual checkpoint cadence (the live run dir is READ-ONLY / executor-owned; the
   exact ckpt is chosen at fire time).
3. Across-seed variance is UNMEASURED (single-seed spine) ⇒ any first-arm result is INSTANCE-scoped until a
   2nd seed bounds the floor (the rule already encodes this).
4. The inherited-chroma P0-4 pin on the LIVE launch (`--seg-chroma-boundary-weight 0.1`) is NOT resolved
   here — that is the launch executor / amber sibling's sealed concern; this rung is the clean add-back
   attribution surface it should be measured against, not a patch to the live config.

**Pointer delta: none. Launches: none. Dispatches: none. Runs touched: none.**

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — "The DSL lever (leg 1) — REUSED, not duplicated" names the composable lever (`SegChromaBoundary`) and its trainer flags (`--seg-chroma-boundary-margin-band`, `--seg-chroma-boundary-start-epoch`, `--chroma` / `--no-chroma`), each independently inspectable at compile time via `curriculum_dsl.py` / `gauge.py`.
2. **Per-signal decomposition** — "The prior measured add-back receipt (the one P0-4 demands)" is the per-signal chroma attribution; "The rung, DERIVED not vibed" carries the derivation of the rung's weight.
3. **Run-to-run diff** — "The A/B measurement plan (deliverable 4)" pre-registers the exact OFF arm (sealed v7.5.3(off) trunk, chroma add-back explicitly 0.0) against the ON arm (`SegChromaBoundary(weight=0.05, margin_band=1.0, start_epoch=0)`) — a matched two-arm diff, not a single-arm read.
4. **Post-hoc query** — `.omx/state/lever_relative_significance.jsonl` and `.omx/state/canonical_equations_registry.jsonl`; the trainer is `experiments/train_levelset_witness_realized_through_R_mlx.py`, resumable via `--resume-from`.
5. **Cite-chain** — the "STORES CONSULTED" section is the recall chain; "Triality" records the DAG/DSL/equations legs.
6. **Counterfactual hooks** — the lever is default-OFF (weight 0.0 = byte-identical), so ON/OFF is the counterfactual; "Honest gaps" enumerates what the A/B cannot yet decide.
