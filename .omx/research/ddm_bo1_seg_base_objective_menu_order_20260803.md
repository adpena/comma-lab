# ddm_bo1 (second charter) — why seg has not moved: objective vs menu vs order, separated by measurement

- arm: `ddm_bo1` · date_utc: 2026-08-03 · axis: **$0, scorer-free** (0 SegNet/PoseNet forwards; the
  scorer slot was neither requested nor touched). `score_claim=false`, `promotable=false`.
  **Pointer `0.1910828242` [contest-CPU] UNMOVED.** Live best `S = 0.7910689`, seg leg `0.4311790`,
  gap `0.6189279` (seg gap `0.4015190` = **64.9%**; 1% of gap = 7,301 flips = 9,295 B — both
  recomputed, both reproduce the charter).
- charter note: the charter said *"first launch — a prior spawn attempt never started."* **Refuted:**
  `.omx/research/ddm_bo1_base_objective_menu_order_20260802.md` exists and is committed. That unit's
  scope was pose pair-coherence / Q3 placement / within-build order; THIS unit is the seg-specific
  second charter. Zero overlap is duplicated; its §4/§7 order results are cited, not re-derived.
- consumes: `upstream/modules.py` · `src/tac/optimization/direct_description_joint_descent.py` ·
  `experiments/train_tr1_partition_renderer_mlx.py` · `ddm_mg1_margin_geometry_cure_20260803` ·
  `ddm_pt2_lever_port_to_tr1_20260803` · `ddm_dc1_menu_sweep_and_ms8_mq1_reconciliation_20260802` ·
  `ddm_dd1_displacement_dimensionality_20260803` · `ddm_br1_basis_race_and_drop_surface_20260803` ·
  `ddm_wf2_waterfill_reprice_20260803` · `ddm_gt2r` (`43a68c18ab`, address law; receipts
  `/Volumes/VertigoDataTier/pact/ddm_gt2_20260803/`) · `ddm_tw1` (#869) · `ddm_pc2` · `ddm_ja1` ·
  `ddm_sf1` ·
  `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/*.zip` (6 archives, hashed this unit) ·
  `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/` (receipts read; caches NOT re-derived).
- consumers: [MAIN, ddm_mg1, ddm_wf2, live arms gt2r/sg3r/cg1r/bn1r/gk1r]
- tokens: [no-triality] [p0-ledger-ok]

---

## §0 ANSWER FIRST — the three explanations, separated

**(3) ORDER — mechanism MEASURED at byte level this unit; but it is the transmission, not the
cause.** NEW RECEIPT: all six on-disk frontier archives
(`qa66_celldrop50 → pj2 → cp1_pj2_fold → pw1 → ms8 → dc1_fold`) carry **byte-identical**
`state/tokens.dr7t` (sha256 `305a2be96a292967268105241170d0f9d94e5dd7342b0ede573916ad46c53d91`,
346,478 B), byte-identical `renderer.sec` (`71e776c3…`) and `selector.sec` (`8fc5c3fc…`). Only
`pose_warp.stp` and `manifest.json` differ. **Every frontier revision since v4d rewrote ONLY the
pose stream.** The loop cannot move seg because it never touches a seg-carrying byte — and the
reason it never does is priced, not mis-ordered: `ja1`'s invalidation edges make a base re-open cost
~1.5 h of re-solves (`token_base` invalidates `pose_solve`+`photo_fit`+`selector`; `sf1`'s
stale-partner law), while pose/rate moves are solves at zero invalidation. A greedy revision market
under that asymmetry buys pose/rate every time. **Within-build order is NOT shown suboptimal**
(`ja1` DAG honored; `bo1` 08-02 §4/§7 stands; `ea1`'s scope limit carried).

**(2) MENU — the DISCRETE menus are measured closed; "menu misconfigured" is REFUTED for coding
menus. What is open on this axis is not a menu — it is the missing REALIZATION instrument.**
Per-menu status in §2. Basis: 0.83% to IDENT, falsifier fired (`br1`). Coder: shut on the shipped
EXPLICIT stream (`br1` M2: 1.109× after dead-zero removal; `oc1` RAW 50/50) — but `gt2r`'s
implicit-vs-explicit axis is measured NOT shut (§2.2b address law). `s_t`: exactly degenerate with the translation
triple (`dc1`, 4.5e-16) — the only dead-codeword menu, and its win was SEARCH not format. Selector:
535 B byte-exact, indexes nothing per-region (`dd1` #8). Alphabet coarsening: measured
monotone-harm (`gc6` S2). Drop: live base IS the knee, both directions realized-dominated (§2.3).
The open items are: the `clip(rint())` amplitude dead zone (every priced displacement ≤0.32 of a
16-px cell — `dd1` #8, `rs2`), the four reachable-but-unraced seg forces (`pt2`, verified live at
`train_tr1_partition_renderer_mlx.py:2188-2213`), and the priced-but-unrealized separatrix
carriers (`mf1` 0.0156 B/flip, `dd1` Lane offsets 703 B vs 250,403 flips — **both DESCRIPTION
ceilings; realization unmeasured by both arms' own statements**, `dd1` #2).

**(1) OBJECTIVE — the only axis with a derived, unraced, no-free-parameter cure — but the
diagnosis is TRAINER-SPECIFIC, and the charter's "the live TR1 trainer" is TWO trainers.**
- **Correction solver** (`direct_description_joint_descent.py:2436-2440`, the live seg-touching
  stage): seg leg = `CE + 0.05·relu(0.1 − m)`. `mg1`'s derivation VERIFIED here independently
  (§1.2): the hinge is **10–16× below parity with CE at the separatrix**. The one term aimed at
  what S measures is nearly silent.
- **Burn trainer** (`make_loss_fn`, seg_form `ce` → event-switch → `tau_softplus(τ=0.3)`; margin
  levers default-off): `tau_softplus`'s push at m=0 is `sigmoid(0) = 0.5` — **already inside
  mg1's parity band [0.50, 0.80]**, and it decays to ~0 in the bulk (sigmoid(−m/τ); ≈1.6e-6 at
  m=4). **IF the event fired before ep399, the burn plateaued FLAT under a boundary-aimed
  objective — evidence AGAINST objective-misaim for the burn and FOR a capacity/realization
  wall.** Whether the switch fired by ep399 is an OPEN RECEIPT CHECK (burn run event telemetry;
  not located in this unit's time-box) — it decides how much of H1 survives at the base.

**The decisive measurement (§4): the `mg1` hinge-weight A/B on the correction solver —
ADJUDICATED YES, with five amendments.** It needs the scorer slot ⇒ sealed spec below, NOT fired.

---

## §1 THE BASE OBJECTIVE, AT SOURCE — the itemized diff

**S seg term, verified at source** (`upstream/modules.py:111-113`):
`diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float(); diff.mean(...)` — uniform per-pixel,
sign-of-margin only, logits discarded, no per-class or per-edge structure. (Also re-verified:
SegNet reads frame_1 only, `:108`; both scorers share the identical `interpolate` `D`, `:73/:109`.)

**Live legs.** A = correction solver: `100·(ce_seg_loss_mlx + 0.05·margin_floor_hinge_mlx(f=0.1))`
(`direct_description_joint_descent.py:2296,2307,2436-2440`; loss `:2397` carries the exact pose
sqrt term; realized `d_seg` computed `:2445` as diagnostic ONLY). B = burn: `ce` →
`tau_softplus(τ=0.3)` event-switched; `margin_weighted_loss=off`, `margin_weight_temp=1.0`,
focal/fisher/natural-grad flags default-off (`pt2`-ported, verified consumed at `:2418-2436`).

| # | S property | leg A (correction) | leg B (burn) | consequence | status |
|---|---|---|---|---|---|
| 1 | counts sign of margin only | CE pushes at ALL margins — 87.93% of the plane sits at margin [4,8) with flip rate 0.000257% (`mg1` §2.2) and still receives CE gradient; hinge is exactly 0 there | post-switch `tau_softplus` ≈ 0 in the bulk | leg A's aggregate gradient is bulk-dominated; leg B's is not (post-switch) | MEASURED (field) + DERIVED (allocation); aggregate CE:hinge ratio UNMEASURED — needs candidate logits (a scorer pass; pu2's cache is argmax-only, verified this unit) |
| 2 | decided at the separatrix | hinge push `w=0.05` vs CE push `1−p ∈ [0.50,0.80]` at m=0 ⇒ **6–10% of parity** | `tau_softplus` push 0.5 at m=0 ⇒ **at parity** | the "too quiet" diagnosis applies to leg A, NOT leg B (conditional on the event) | DERIVED, re-verified this unit (§1.2) |
| 3 | uniform over pixels; under scarce DOF the S-optimal spend is cheapest-flip-first | no depth-awareness; hinge gradient FLAT on active sites (`mg1` §1.2, executed) | same | neither leg orders repair by cost; convex (squared) hinge would be ANTI-aligned (`mg1` §6.3, derivation) | DERIVED, untested |
| 4 | edge-blind, but the residual is one graph with one hub (Road in 87.8% of flips; Road↔Lane 49.2% — `pc2`) | no per-edge term | no per-edge term | matches S's form, mismatches the residual's geometry; NOTE `mg1`'s per-class-pair-scalar null is INSTANCE/formulation-scoped (§5.3 below), not a family kill | MEASURED (structure) |
| 5 | target = GT argmax | same (`seg_targets`) | same (`lstars`) | no diff | VERIFIED |
| 6 | realized d_seg | not in the loss — gradient identically zero (`rt2`, executed) | same | necessity, not a defect | MEASURED |
| 7 | — | `margin_targets` declared+assigned, never read; `derive_margin_floor` unwired on this path (8 callsites hardcode defaults) | n/a | wiring owed IN THE SAME CHANGE as the A/B (`mg1` §5.3) | MEASURED |

### §1.2 Independent verification of the `mg1` derivation (charter: "verify, don't inherit")

- Parity band: at m=0 the target logit ties the best competitor; softmax with C=5 ⇒
  `p_target ∈ [1/5, 1/2]` ⇒ CE push on the target logit `1−p ∈ [0.50, 0.80]`; hinge push = `w·1`
  (relu slope 1). Parity ⇒ `w* ∈ [0.50, 0.80]` = 10–16× shipped 0.05. **Arithmetic checks.**
- Limit identity: for `f → ∞`, `relu(f−m) = f−m` for all m ⇒ `mean = f − mean(m)` **exactly** —
  the floor lever monotonically approaches the bulk objective; raising it is refuted (`mg1` §1).
- Floor-raise cannot recruit a flip: every flip has `m < 0 < f` at any `f > 0` ⇒ coverage already
  100%. **Algebraic, field-independent. Holds.**
- NEW (this unit): the burn's stage-2 `tau_softplus` push at m=0 is `sigmoid(0)=0.5` — parity
  without any weight change; decays as `sigmoid(−m/τ)` ⇒ boundary-concentrated. The parity defect
  is therefore **specific to the correction solver's leg**, modulo the event-fired check.

---

## §2 THE MENU — every discrete menu on the live path, with status

Live counted state (source-verified by the cited arms): `tokens.dr7t` 600×24×32×4 uint8 (16
symbols, all used — `br1`) · `renderer.sec` 3,341 B · `selector.sec` 535 B · pose/photo stream =
11 knobs/pair (6 pose + `s_t` + selector + (a,b) + `beta_idx` — `dd1`, source-verified) ·
correction-solver θ = template colour deltas + realized secants.

| menu | occupancy | dead codewords | exhausted? | receipt |
|---|---|---|---|---|
| token alphabet {0..15} | all 16 used | 0 | coarsening measured monotone-harm | `br1`; `gc6` S2 |
| token units (768 cells × 4) | 384/768 cells fully dropped | — | live base IS `cell_drop50` = `gr1`'s knee; both directions REALIZED-dominated (§2.3) | `br1` §3; `ba31` |
| basis/transform of the lattice | — | — | CLOSED: 264 evaluations, every reversible re-expression within 0.83% of IDENT; falsifier FIRED | `br1` §2 |
| entropy coder | — | — | CLOSED: 1.109× vs order-0 once free-riding dead zeros removed; RAW wins 50/50 | `br1` M2; `oc1` |
| `s_t` codebook | fitted (`ms8`) | the ONLY menu of 5 swept with any dead codeword | format ceiling ≤0.056% stands (`mq1`); `gap_lattice ≡ 0` — exactly degenerate with the translation triple (4.539e-16, n600); the −0.049 S win was SEARCH | `dc1` |
| selector | 535 B, byte-exact | — | indexes NOTHING per-region; `token_cell_mask` build-time only | `dd1` #8 |
| pose menus (beta_idx, a,b, p6) | occupied | 0 (others "exactly 0%") | pose side, out of seg scope | `dc1` |
| correction-solver θ | small (templates+secants) | — | NOT exhausted — never re-aimed (§4) | this unit |

**§2.3 The #918 adjudication ("rate coding is closed") — CONFIRMED, one bounded reopening
condition; the charter's wf2-tension DISSOLVES.** `tw1`'s +13.1% (marginal saving grows with
depth, 52/53 cells) and +7.0% (superadditive) are corrections to **`wr1`'s price KEY** (priced at
k=0). `ba31`'s knee domination is a **REALIZED S measurement**: restore +0.047 S @ +80,615 B;
drop-more +0.052 S @ −81,406 B, i.e. realized 0.6498 B/flip vs W=1.2731 — dominated **1.96×**.
A 13–20% key correction cannot overturn a 1.96× realized domination; both claims are true about
different objects. **The one reopening condition:** `ba31`'s 0.106205 S distortion cost does not
split seg from pose (`na1`'s open P0); with `dS/d(d_pose) = 31.30`, if the pose share exceeds
~49%, deeper drops flip profitable once pose is re-solved (pu2's floor probe shows pose repair is
cheap: 6 pairs, −0.0354 S). Until `na1` splits it, the knee stands as live base. **Scope note
(`gt2r`):** "coder axis shut" was measured on EXPLICIT streams (the token lattice); `gt2r`'s
implicit-vs-explicit gap is a DIFFERENT axis and it is measured NOT shut (§2.2b) — the closure
claim is scoped to the shipped explicit stream, not to coding as such.

**What the menu axis leaves OPEN for seg (not menus — instruments):** (a) sub-cell amplitude
realization through the `clip(rint())` dead zone — every displacement priced by `dd1`/`mf1` is
≤0.32 of a 16-px cell, invisible to every linear key; (b) the four `pt2` seg forces —
`--seg-focal-gamma`, `--fisher-density-weight(/-source)`, `--head-natural-grad(-eps)`,
`--tau-softplus-tau` — reachable, DSL-held, NONE raced; (c) the description-priced separatrix
carriers, whose realization no arm has measured (`dd1` #2: "Neither arm measured realization at
all"). Description budget is still not the wall, but **re-anchored** (`gt2r`, real coders, n600):
the exact L\* answer's REAL-CODER cost is **410,584 B** (implicit whole-corpus) against the
647,553 B buy threshold — **1.58× under**, not `dd1` #9's 2.99× (its 216,395 B was not a
real-coder row; `sx1`'s 253,341 B is refuted as a coder cost, real coder 1.6207× higher).

**§2.2b THE ADDRESS LAW (`gt2r` `43a68c18ab`, n600, real lzma/brotli/zlib outputs) — a measured
constraint on every FUTURE seg menu entry.** Implicit whole-corpus coding of L\* (410,584 B)
beats static-explicit factoring (434,156 B, +5.7%) and temporal-explicit factoring (564,784 B,
**+37.6% — WORSE than per-frame-independent coding**). Mechanism: the ADDRESS is **76.4–78.5%**
of every explicit factoring's cost (the static lexicon is ~424 B; payloads are cheap), and it
cannot be amortized — only **28%** of residuals sit within r=1 of the STATIC boundary (vs `pc2`'s
93.89% near the TRUE separatrix): **the boundary language is DYNAMIC; the address must be
receiver-GENERATED, never transmitted.** Consequences: any menu entry that transmits explicit
location structure starts 76–78% behind before its first payload byte; the viable forms are
receiver-derived context (free) or tiny counted priors (`sx2`'s 49 B class precedent). Also
representation-level, not loss-level: **Lane per-held-pixel stays 25–26× costlier than Road under
either grammar**, every ε>0 lossy point is dominated at W (up to 27.9× against), lossless
polygons beat rasters for all classes (0.61–0.69×) ⇒ Lane's production must be temporally
GENERATED (tracked words) — no objective re-weighting changes that.

---

## §3 THE ORDER — the byte-level receipt, and what it does and does not prove

1. **NEW RECEIPT (this unit, $0):** six frontier archives, `state/tokens.dr7t` +
   `state/renderer.sec` + `state/selector.sec` sha256-identical across all six; only
   `pose_warp.stp` (8,290→8,720 B range) and `manifest.json` vary. The live best (`cx1_pj2ix2`,
   single `0.bin` container) shows the same `d_seg = 0.00431179` as this token base (`pu2`
   baseline receipt) — consistent with zero realized seg change through 08-03.
2. **The last d_seg change was a deliberate seg-for-rate SALE** — `gr1`'s knee selection
   (`cell_drop50`) between the burn end (ep399 `d_seg 0.0038892` FLAT, burn telemetry axis) and
   the v4d base (0.0043118). Attribution of the full +49.9k flips to the drop is **INFERRED**
   (the burn-side number is advisory-axis; no pre-drop byte-closed argmax is cached) — labeled,
   not asserted.
3. **Coupling per edge (from receipts):** `token_base → {pose_solve, photo_fit, selector}`
   invalidation ~1.5 h (`ja1`, measured); `pose_solve → photo_fit` ~30 min; pose/photo re-solves
   invalidate nothing upstream. State-dependent prices (`tw1`) bind WITHIN the drop surface;
   stale-partner (`sf1`) binds every solved stream to its base.
4. **One measured order INVERSION exists — inside the coding grammar, not the stage DAG**
   (`gt2r`, §2.2b): temporal-explicit "factor first, code second" lands +37.6% ABOVE the
   per-frame-independent baseline — for explicit seg carriers the right order is
   **predict-then-code against a receiver-generated context**, not factor-then-code. This binds
   every FUTURE explicit carrier design (the §4 routing target), while the live implicit
   token path is untouched by it.
5. **Verdict:** within-build stage order NOT shown suboptimal (`ja1`/`ea1`/`bo1`-08-02 honored;
   the one measured inversion is intra-coder, item 4). The campaign loop's seg starvation is real
   and now byte-receipted, but it is the **rational consequence of an asymmetric price structure
   in which no REALIZED seg mechanism exists** — fixing "order" without a realized seg mechanism
   changes nothing. Order is the transmission, not the cause. *(Scoped: this flips if `na1`'s
   split shows the knee itself was mis-chosen — then a genuine ordering error, the seg-for-rate
   sale, sits at the base.)*

---

## §4 THE ONE DECISIVE MEASUREMENT — sealed spec (NOT fired; needs the scorer slot; MAIN's GO)

**Adjudication of the charter's candidate: YES — the `mg1` hinge-weight A/B is the decisive
measurement**, because (i) it tests H1 at the only live seg-touching stage with a derived
no-free-parameter target; (ii) it is a SOLVE over ~dozens of θ DOF, not a burn — the cheapest
scored seg measurement available; (iii) BOTH outcomes route: a move confirms the objective was
binding; a properly-instrumented null (below) confirms the realization wall and redirects all seg
work to carrier realization. No $0 measurement can substitute: the aggregate CE:hinge gradient
ratio needs candidate logits (pu2's cache is argmax-only — checked), and the carrier-realization
alternative requires building an instrument that does not exist, so it is a build, not a
measurement.

**SPEC (amendments over `mg1` §7.1 in bold):**
- Surface: `DirectDescriptionJointDescentMLXModule(margin_hinge_weight=w)`. Arm A `w=0.05`
  (**may reuse the existing solver receipt iff config-hash-identical; else run**), arm B
  `w=0.65` (band centre), endpoints 0.50/0.80 only if the slot allows.
- Fixed: `margin_floor=0.1` (raising REFUTED — `mg1` §1), targets, pairs, seed, solver iterations,
  θ budget (identical parameter count ⇒ description bytes matched by construction; archive byte
  delta reported anyway through the REAL byte-close).
- **n600 always; a prefix is a different population** (`m88`: 5.1× d_pose skew flipped a −0.122 S
  "win" to +0.152 S). If any subset is ever forced, print subset-vs-population mean of the
  governing quantity.
- **Both arms pose-RE-SOLVED against their own decoded base** (`uv1` `resolve_base()`; `sf1`
  partner law) before any comparison; report d_seg AND post-resolve d_pose AND bytes (seg-only
  A/B forbidden — `uv1` measured 3,019× d_pose separation between bases).
- **Telemetry, score-neutral, default ON:** per-iteration gradient share CE vs hinge + θ
  displacement norm + per-edge flip delta (Road↔Lane / Road↔Undriv / the two Movable edges /
  rest, per `pc2`'s decomposition — "every lever has a sign per edge"). This is what makes a null
  interpretable: hinge-share↑ with θ moved but flips unmoved ⇒ realization dead zone (H2);
  hinge-share↑ with θ unmoved ⇒ menu cannot express the direction (H2-menu); flips moved ⇒ H1.
- **Pre-registered kills (each with denominator):**
  - K1: |Δd_seg| below a **measured ≥2-seed repeat floor in the same window**; absent that floor
    the outcome is `UNRESOLVED_NO_NOISE_FLOOR`, never a kill. Report Δd_seg as % of the SEG gap
    (1% = 4,737 flips = 0.0040152 S) and of the total gap (1% = 7,301 flips = 0.0061893 S).
  - K2: proxy margin improves, realized d_seg does not ⇒ realization wall; route to carrier
    realization (`mf1`/`dd1` Lane offsets), not to more weight.
  - K3: post-resolve d_pose regression priced at the **CURRENT** `dS/d(d_pose)` (never a shelf
    price — `wf2` §5: shelf prices stale ≥2.22×) exceeding the seg gain ⇒ net-negative; INSTANCE.
  - K4: L7 cross-hardware portability guard degrades ⇒ stop. Prediction: it should STRENGTHEN
    (higher w pushes margins past the drift band); the prediction is itself the test.
- Same change wires `margin_targets` + `derive_margin_floor` (`mg1` §5.3 — exercised the moment
  it lands, not before).
- Outcome routing: ≥1% of seg gap ⇒ sweep w, then race the four `pt2` forces (same solver, same
  protocol); null-with-telemetry ⇒ the P0 seg lane becomes the sub-cell realization instrument
  (dd1 Lane perpendicular offset through real R→uint8→argmax), and no further objective work is
  admissible on this vehicle until that instrument exists. **The instrument inherits `gt2r`'s
  address law as a design constraint: receiver-generated address + tiny counted payload only;
  Lane production temporally GENERATED (tracked words); no explicit location stream (76–78%
  address tax, measured).**

---

## §5 WHAT THIS UNIT REFUTES IN ITS OWN CHARTER (required)

1. **"first launch — a prior spawn attempt never started"** — REFUTED; the 08-02 `ddm_bo1` memo
   exists and is committed. Reconciled above; nothing duplicated.
2. **"the live TR1 trainer" (singular)** — UNDER-SPECIFIED: two trainers hold seg legs, and the
   diff differs materially between them (§1). The parity defect is the correction solver's; the
   burn's stage-2 form already sits at parity at m=0 (conditional on the event having fired —
   named OPEN receipt check: burn run event telemetry, not located in this time-box).
3. **"mg1 also killed the per-class-pair scalar family"** — OVER-STATED: `mg1`'s typed verdict is
   INSTANCE (the #766 waterfill on live cx1, grains 1–256 px) with a FORMULATION-scoped
   generalization to side-segregated regimes; the mixed-edge subpopulation is its own named
   un-collapsing measurement. Family ≠ killed.
4. **The wf2-vs-#918 "tension"** — DISSOLVED (§2.3): price-key correction vs realized S
   measurement; both true; the realized measurement governs until `na1` splits ba31's distortion.
5. **Order as a co-equal third explanation** — on current receipts, order is the transmission
   mechanism of seg starvation, not its cause (§3.4); scoped to flip on `na1`'s split.
6. Charter arithmetic verified clean: 64.9% seg share ✓; 7,301 flips and 9,295 B per 1% of gap ✓.

## §6 POINTER HONESTY + NEXT-IF-RESUMED

This unit moved the pointer by ZERO. It produced one new byte-level receipt (six-archive
tokens/renderer/selector identity), one independent verification of the `mg1` derivation plus the
burn-side parity fact that bounds it, a per-menu closure table, and one sealed spec. Means, not
ends. NEXT: (1) MAIN fires the §4 A/B when the slot frees (arm A possibly free via receipt
reuse); (2) locate the burn event telemetry and close the `ce`→`tau_softplus` fired-or-not check
— it decides whether H1 applies to the burn at all; (3) `na1`'s ba31 seg/pose split — it is the
single number that can reopen BOTH the drop knee (§2.3) and the order verdict (§3.4);
(4) after the A/B: race `pt2`'s four forces or build the sub-cell realization instrument, per §4
routing.
