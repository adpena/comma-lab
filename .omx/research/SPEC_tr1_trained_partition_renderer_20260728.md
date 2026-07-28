# SPEC — tr1: the TRAINED partition→pixel renderer (token-grid + scorer-in-loop conv renderer)

**Design-only build-ready SPEC. No training, no scorer jobs, no launches ran in this arm.**
The BUILD fires only on MAIN's GO after fd2's FORMAL typed verdict lands (§S5 fork-conditional).

**Pointer honesty first.** `0.1910828242 [contest-CPU]` — our SUBMITTABLE original-work frontier —
is **UNMOVED**; this arm moved no exact score (it is a design memo). Full honest pointer picture,
verified from `.omx/state/canonical_frontier_pointer.json` this session:
- **0.1910828242 [contest-CPU]** = our submittable original-work frontier (the honesty anchor the
  whole 07-28 arc cites; corroborated by the fd2 receipt's `"pointer"` field).
- **0.172** = `effective_frontier` = the official leaderboard best (PR130 `semantic-pose-HPAC_CPR1`,
  rank 1) = the competitive score-to-beat. The GOAL bar is `min(0.15, 0.172)`.
- **0.18804** = `our_local_frontier_contest_cpu` = a NON-SUBMISSION borrowed-substrate Modal bank
  (PR128-on-PR110); harvest-signal only, not submittable, not our progress.

`evidence_axis: design-only SPEC · research_only=true · score_claim=false · paid_dispatch=false`.

---

## STORES CONSULTED (recall-first, multi-pass; every cited number verified from its committed artifact this session)

- **CLAUDE.md + AGENTS.md** (full read): NO-FAKE supreme rule (#6 search-as-solver, #7 borrowed-
  substrate-as-original, #8 surrogate-not-exact); THE GOAL sub-0.15; NO-OLD-LINEAGE ban (HNeRV/PR95/
  110/128 lessons-only, never vehicles/carriers/calibration); the evaluator-equivalent witness-compiler
  paradigm; rule-118 (GENERIC generator FREE in inflate.py, VIDEO-DERIVED/LEARNED weights COUNTED in
  archive.zip); eval_roundtrip + EMA + QAT non-negotiables; the capstone θ* trainer canonical entry
  point; serializer + post-edit sha; `.py` review gate.
- **Charter** `scratchpad/tr1_charter.md` — S1..S6, the fork-conditional discipline, the band-lemma
  no-correction operating point, the fd2 lesson, the fd1r wall-clock law.
- **fd2 receipt** `/Volumes/VertigoDataTier/pact/ddm_fd2_20260728/fd2_disambiguation_receipt.json`
  (schema `ddm_fd2_posenull_gn_disambiguation.v1`): `canary_pass=true`; Q2 ladder ×1.0 =
  `BLOCK_MOVED_NOT_IMPROVED` (489 block flips, block d_seg WORSE 0.08539835 vs baseline 0.08539199;
  offblock 757 flips WORSE); ×0.5 and ×0.25 = `REALIZATION_GAP_NO_BLOCK_ARGMAX_FLIP` (`description_changed=true`
  yet ZERO flips, delta EXACTLY 0.0); GN step1 `active_parameter_count=344`, `rayleigh_curvature=3.74e-5`,
  `step_norm=648.8`; `pointer_moved=false`. **NO top-level typed verdict field ⇒ FORMAL verdict PENDING.**
- **pp1 band-lemma receipt** `.omx/research/ddm_pp1_band_lemma_receipt_20260728.json` +
  **REGISTERED equation** `src/tac/canonical_equations/ddm_pp1_correction_stream_position_band_20260728.py`
  (`ddm_pp1_correction_stream_position_band_v1`): water `1.2731 B/flip`; measured coherent crossing
  `ρ_c = 5.02e-4`; derived uniform crossing `ρ_u = 8.59e-4`; rational-correction band `[5e-4, ~1e-2]`;
  `lemma_confirmed=true`. Docstring design spec (verbatim): *"a carrier must be natively ≤ ~5e-4
  (ideally ≤ 3e-4, PR130's rail) to ship NO correction stream; sub-ρ_c correction machinery is
  permanently pointless."*
- **pp1 pricing memo** `.omx/research/ddm_pp1_direct_partition_pricing_20260728.md`: direct-EXPLICIT
  partition = **173.6 KB** lossless (context-arith temporal o8, KT) → composed explicit route **~0.189**
  (above the 0.172 bar, +57 KB vs learned tokens); per-class attribution Road 62.4 / **Lane 62.3 (36%
  of cost at 0.72 b/px, 0.59% of pixels)** / Undriv 20.9 / Movable 13.2 / MyCar 5.6 KB; boundary 2,436
  px/frame; temporal disagreement 1.246%.
- **ee1 memo** `.omx/research/ddm_ee1_einstein_fresh_eyes_capstone_20260728.md`: C5 (paint-159× loss
  measured on a **ZERO-parameter** palette, native d_seg 0.0086; the trained renderer is the solved
  form); C10 (convergence theorem — family-d GN at partition-grade capacity ≡ token-grid + trained
  renderer in different coordinates); R2 (the build: ≤64 KB renderer, falsifier ≤1e-3); R6 (composed
  arithmetic ≈0.176 @196KB / 0.145–0.16 @150KB); R7 (control-token re-solve / dirty-paper).
- **fd1 memo** `.omx/research/ddm_fd1_family_d_gn_description_engine_20260728.md`: the **fd1r wall-clock
  law** — a step ≈ 1,547 s of which the GN propose is ~6 s and **99.6% is realized-acceptance pricing**
  (3 chunked n600 CPU verdicts ≈ 514 s each); the **two-rung ladder** whose **Rung 2 IS this object**
  ("re-parametrize as token-grid + small trained partition→pixel renderer, ≤64 KB counted, scorer-in-
  loop through R + uint8-STE … dissolves binding dimension 1 by construction (the renderer trains
  against ALL pairs simultaneously) … rung 2 is the plausible ≤5e-4 route"); the #383 terminal-pose
  law (`pose_objective_weight=0` on the seg leg); binding dims = cross-pair transfer (primary) +
  pose-collateral (co-primary); `verdict_scope: INSTANCE, NOT family`.
- **fc1 receipts** (via pp1 recall) `/Volumes/VertigoDataTier/pact/ddm_fc1_20260728/entropy_n600.json`:
  total-archive byte budgets **0.172 bar → 187,727 B**, **0.15 bar → 154,522 B**; 1.2731 water.
- **E4/WS1 exporter grammar** `.omx/research/codex_findings_ddm_e5_e4_ws1_exporter_adapter_20260724_codex.md`:
  `DDME4WS1RuntimeExporterConfigV1`, grammar `ddm_ws1_receiver_closed_warm_start.v1` (ordered/contiguous/
  gap-free stream partition; per-stream byte-count+offset+SHA-256+**named receiver consumer**; exact
  archive SHA + parser re-emit; Brotli-Q11 packets — raw-LZMA1 fallback BLOCKED).
- **sh1 memo** `.omx/research/ddm_sh1_compose_and_local_exact_findings_20260727.md`: the byte-close →
  FULL local exact protocol → advisory-S decomposition chain.
- **Canonical surfaces confirmed present** (paths cited in §S1/S4/S6): `src/tac/witness_dsl/`,
  `experiments/train_levelset_witness_realized_through_R_mlx.py`, `src/tac/witness_autoconfig.py`,
  `tools/launch_witness_run.py`, `tools/witness_memory_preflight.py`, `src/tac/quantization.py`
  (`FakeQuantSTE:142`, `Uint8STE:190`, `apply_uint8_ste:219`), `src/tac/canonical_equations/ema_decay_run_geometry_20260717.py`,
  `src/tac/boundary_math/{lever_b_generator,hood_static_component,lane_sdf_component,amortized_luma_carrier}.py`,
  `src/tac/packetir_exact_closure.py` (receiver-consumption / `unconsumed_trailing_bytes`).

**Value-provenance ladder tags used below:** `[MEASURED:<path>]` (our committed artifact) ·
`[REGISTERED-LAW:<eq_id>]` · `[DERIVED]` (computed here, derivation shown) ·
`[EXTERNAL-LESSONS-ONLY]` (PR130/pi1 — existence proof; never adopted as bytes/architecture constants).

**Three recall discrepancies I must report (RECALL-FIRST binding — reported, not silently proceeded):**
1. **The charter's "706/344-param lift" is two distinct objects.** fd1's *description space* is **706**
   counted params `[MEASURED: fd1 memo L208]`; the fd2 GN *lift* has `active_parameter_count=344`
   `[MEASURED: fd2 receipt]`. This SPEC treats them as distinct and cites each with its own object.
2. **The charter's "#402" for parse-back/exact-consumption is a mis-cite.** CLAUDE.md #402 =
   *telemetry-rows-carry-liveness* (a confound gate). The receiver-consumption bijection
   ("counted-but-inert = FAKE") is **#417** `[MEASURED: MEMORY row + src/tac/packetir_exact_closure.py]`.
   §S6 uses #417 (+ the E4/WS1 named-receiver-consumer grammar). Reported; not proceeded on the wrong number.
3. **The charter/fc1 "212 KB" budget was NOT located** in the fc1 entropy receipt this session
   (fc1 gives 187,727 B @0.172 and 154,522 B @0.15). §S3 uses the two verified budgets and flags 212 KB
   as unverified.

---

## S1 — THE OBJECT (derived from OUR laws; PR130 is lessons-only existence, never adopted bytes/constants)

**The object = a token-grid latent field + a small trained partition→pixel conv renderer, decoded
scorer-in-loop.** Two counted payloads: (a) the per-frame **token grid** (video-derived, COUNTED),
(b) the **renderer weights** (learned, COUNTED, int-quantized). The renderer forward-pass CODE and the
token-grid *interpreter* are GENERIC ⇒ FREE in inflate.py (rule-118). The SPEC derives every geometry
choice from OUR measured scorer facts; PR130's grid/renderer constants are never adopted.

### S1.1 The scored object (what the renderer must satisfy)
The SegNet-scored partition is the **600 last-frames** at **512×384** (scorer input plane), 5 classes
`{Road, Lane, Undrivable, Movable, MyCar}` `[MEASURED: pp1 memo — modules.py:108; gt_n600.npz lstars]`.
Class fractions `[Road 0.232, Lane 0.006, Undrivable 0.495, Movable 0.012, MyCar 0.254]`; boundary
**2,436 px/frame**; consecutive-frame partition disagreement **1.246%** (highly temporally coherent)
`[MEASURED: pp1 memo]`. The renderer's ONLY binding job is **native d_seg** (argmax on painted frames
vs GT partition through the frozen scorer); pose is TERMINAL (§S2); chroma is a d_seg lever because
SegNet reads RGB `[MEASURED: CLAUDE.md scorer facts]`.

### S1.2 Token-grid geometry — DERIVED from the scorer's spatial acuity (NOT PR130's grid constants)
Three OUR-law inputs fix the grid, each `[MEASURED]` from our scorer characterization:
- **Scorer plane 512×384** — the grid is a spatial lattice over this plane (argmax decided here after
  the stride-2 stem) `[MEASURED: CLAUDE.md exact scorer arch]`.
- **ERF r50 ≈ 85 px / r90 ≈ 300 px** `[MEASURED: MEMORY segnet_recursive_fractal_factorization]` — a
  single token influences an ERF-sized decoded region, so a token pitch `p` satisfies `p ≲ ERF_r50 / k`
  (k≈4–8, raced) to place a boundary within one token's zone of control. `[DERIVED]` starting lattice:
  a **downsample factor D over 512×384**, `D ∈ {8, 12, 16}` raced (→ 64×48 … 32×24 grids), because at
  D=16 the pitch is 16 px ≪ r50 85 px (boundary placement well inside control) and the grid is
  `32×24 = 768` cells/frame — the coarse end that keeps token bytes at the learned plateau end.
- **Temporal coherence 1.246%** `[MEASURED: pp1]` ⇒ tokens are **temporally delta/context-coded** (the
  frame-to-frame token field is nearly static), matching pp1's 33 KB temporal-as-context saving. `[DERIVED]`
- **Per-cell code width** `c ∈ {2, 4, 6}` raced; entropy-coded with a small learned prior (HPAC-style
  factorized categorical over the quantized code lattice — GENERIC coder, FREE). The **learned end** of
  the plateau is reached when the renderer absorbs the shared partition structure so the residual
  per-frame token entropy is minimal (pp1's finding: learned tokens land ~117 KB vs explicit context-
  arith 173.6 KB — a **+57 KB** structure the renderer is designed to absorb) `[MEASURED: pp1 R1 + ee1 R6]`.

**Falsifier for the grid derivation:** if at the coarsest raced grid (D=16, c=2) the renderer cannot
reach native d_seg ≤ 1e-3, the grid is capacity-starved and must refine (D↓ or c↑) — but refinement
raises token bytes, so the grid race is a d_seg(D,c)/bytes(D,c) Pareto sweep, not a fixed constant.

### S1.3 The renderer — DERIVED capacity + form (SPADE/CLADE-family, self-detecting per-class)
- **Form:** a small **partition-conditioned conv stack** with spatially-adaptive normalization
  (SPADE/CLADE-style) modulated by the token grid + the class field. `[EXTERNAL-LESSONS-ONLY]` RF-7
  (PR86) derives that **class-boundary placement needs ~3 conv layers** of receptive field; SPADE/CLADE
  literature is the family. We adopt the *form family* (3-ish conv layers, spatially-adaptive
  conditioning), never PR130's weights/constants.
- **Capacity sizing — DERIVED from the measured per-class hardness, not a global default:** partition
  cost attribution `[MEASURED: pp1]` says **Lane = 36% of the cost** (thin dashes, 0.72 b/px), while
  Road (bulk), Undrivable (sky), MyCar (static hood #139) are cheap. ⇒ the renderer's capacity budget
  is **concentrated on the Lane/boundary manifold** (the codim-1 separatrix, per the unified level-set
  flow), with the self-detecting per-class components as priors: `lane_sdf_component.py` (Lane,
  spatial/thin self-detect), `hood_static_component.py` (MyCar static core), leaving Road/Undrivable to
  the cheap bulk head. `[DERIVED + reuses OUR substrate]`
- **Chroma active:** the renderer emits full RGB (not luma-only); chroma channels carry argmax-relevant
  signal at the boundary annulus (SegNet reads RGB) `[MEASURED: CLAUDE.md "Chroma is a d_seg lever"]`.
- **Weights COUNTED + int-quantized:** QAT per the canonical discipline (`src/tac/quantization.py`
  `FakeQuantSTE`/`Uint8STE`; per-channel FP4/int fake-quant, freeze BN, fine-tune at 0.1× LR).
  **Bit-depth is RACED (int4/int5/int8), never cargo-culted int8** (bit-depth-DOF law). `[EXTERNAL-LESSONS-ONLY]`
  existence: PR130's **40,252 B int4** renderer reaches native **2.97e-4** — a proven ≤64 KB-class
  capacity point that shows the target is reachable; we neither load it nor copy its layer constants.

### S1.4 Why this object and not the explicit context-arith partition (the convergence, measured)
pp1 R1 measured the explicit route: context-arith partition = 173.6 KB → composed **~0.189** (above
bar). The token+renderer route absorbs the shared structure so the token leg lands at the **~117 KB
learned end** — the +57 KB the renderer is built to eat `[MEASURED: pp1 R1, ee1 C10]`. ee1's C10
convergence theorem: family-d GN at partition-grade capacity ≡ this object in different coordinates;
fd1's Rung-2 routing names it as the plausible ≤5e-4 route because **it trains against ALL pairs
simultaneously**, dissolving fd1's measured primary binding dimension (cross-pair transfer) by
construction `[MEASURED: fd1 memo capacity verdict]`.

---

## S2 — TRAINING PHYSICS (the fd2 lesson encoded: descend THROUGH the quantization, not propose-then-quantize)

**The fd2 lesson, stated exactly `[MEASURED: fd2 receipt]`:** a continuous GN step in description
coords either (×1.0) moves the block argmax but DEGRADES d_seg (489 flips, worse), or (×0.5, ×0.25)
changes the description yet realizes ZERO argmax flips (`description_changed=true`, delta EXACTLY 0.0)
— i.e. **the GN validity radius is below the uint8 quantum**: the improving small steps cannot realize
through the uint8/coder staircase, and the only step that realizes overshoots. **A propose-continuous-
then-quantize optimizer cannot cross this wall.** The trained renderer with uint8-STE + R in-loop
**optimizes THROUGH the quantization** — the straight-through estimator gives a gradient across the
uint8 round-trip so descent lands on quantized states that actually realize. This is the sole measured
mechanism that crosses the realization wall (weight-level eval_roundtrip since April; R1 pose descent;
mc1 admitted −0.0516) `[MEASURED: CLAUDE.md eval_roundtrip law + fd2 receipt sister_corroboration]`.

### S2.1 The loss (scorer-in-loop, description-level eval_roundtrip)
- **Full R in-loop, BOTH levels:** (a) weight-level uint8-STE on the renderer weights (`Uint8STE.apply`);
  (b) **description-level** eval_roundtrip — the token grid is quantized + entropy-round-tripped in the
  forward pass, and the rendered frame passes the full R operator (bicubic↑384→874 → uint8-STE →
  bilinear↓→512×384) before the frozen scorer. This is eval_roundtrip lifted from the weight to the
  *description* level — the fd2 wall lives at the description/uint8 staircase, so the STE must span it.
  `[DERIVED from fd2 lesson + CLAUDE.md eval_roundtrip]`
- **Seg loss:** boundary-weighted, computed on the frozen-scorer exact factorization — the SegNet head
  is EXACT rank-4 linear, flip distance `d = |margin| / ‖Δw‖`, so the loss weights the codim-1
  separatrix (margin field = Fisher surrogate, Pearson 0.978) `[MEASURED: MEMORY segnet fractal +
  frozen-scorer exact factorization]`. `pose_objective_weight = 0` during the seg trunk (§S2.2).
- **Train against ALL n600 pairs** (the whole point of Rung 2 — dissolves cross-pair transfer). Local
  MPS-gradient training is legal (MPS is a valid gradient device, NEVER a score); authority is the
  frozen CPU-torch scorer `[MEASURED: CLAUDE.md MPS train/authority split]`.

### S2.2 Pose staging — the #383 TERMINAL law (never priced mid-descent)
- Seg trunk trains FIRST with `pose_objective_weight=0` `[MEASURED: fd1 memo L99, #383]`. Pose descent
  engages LAST, once the seg trunk is conditioned — a **6-equation GN solve on the frozen composed
  frames** (per the terminal-pose law: bank `t_p = P(orig)`, `b_p = P(painted)`, GN-solve the seg-null
  low-frequency chroma steering; per-pair + normalize + grid-bicubic) `[MEASURED: MEMORY pose_is_a_
  terminal_six_equation_solve]`. **Watch pose collateral throughout the seg trunk (fd1 measured every
  seg step perturbs pose +2.8…+13.1%), but NEVER price it mid-descent** — the terminal solve absorbs it.
- **Why terminal, decisively (the dominant composed-S lever):** `[DERIVED]` the pose leg swings S by
  **0.11** — terminal-solved d_pose ~2.33e-5 → contribution √(10·2.33e-5) = **0.0153**; the banked R1
  dxi fallback (d_pose 0.001610, 7.2 KB) → contribution **0.1269**. With the fallback the whole renderer
  path is 0.31 (fails); with the terminal solve it is 0.144–0.176. **Pose-terminal is the single most
  decisive design decision.** frame_0 is structurally seg-free (d_seg obligation 8.5e-9) — the cheaper
  place for the joint-trained pose output `[MEASURED: CLAUDE.md Unit C]`.

### S2.3 Compute + the wall-clock verdict cadence (the fd1r pricing law)
- **1-thread MLX training** (2.96× standard) `[MEASURED: MEMORY operator_1thread_training_standard]`.
- **fd1r wall-clock law `[MEASURED: fd1 memo L162]`:** a training step is ~1,547 s of which the
  propose/solve is ~6 s and **99.6% is the realized-through-R verdict** (3 chunked n600 CPU verdicts ≈
  514 s each). **Verdict pricing DOMINATES.** ⇒ design the verdict cadence, not the step: **cheap
  hard-subset inner gates + sparse full-n600 confirms.** `[DERIVED]`
  - **Inner gate (frequent):** advisory d_seg on a **hard-subset** (the g3 flip-prone / small-margin
    pairs — the ~0.6% Lane + boundary long-tail that carries the flip mass) through the frozen scorer,
    chunked (`--verdict-batch` chunking, bit-identical because eval-mode BN uses running stats) `[MEASURED:
    CLAUDE.md verdict-batch OOM law]`.
  - **Full-n600 confirm (sparse):** the authority verdict, run only at stage boundaries / candidate-
    admission, `_verdict_dseg_dpose_chunked` at `--verdict-batch 32`.
  - **Admission:** a candidate is admitted only on a strict full-n600 realized decrease (the v19-style
    realized joint advisory action gate), never on the inner-gate surrogate `[MEASURED: fd2 acceptance_rule]`.

### S2.4 Control-token re-solve (ee1 R7 — zero-byte realization repair, optional post-convergence)
After the renderer converges, re-optimize the coded token grid THROUGH the frozen renderer+SegNet so
composed output matches GT (tokens become control variables; Gelfand-Pinsker / dirty-paper flavored).
Falsifier: no d_seg improvement at equal bytes ⇒ skip `[DERIVED: ee1 R7]`. Joint renderer+token training
gets this implicitly; the explicit re-solve is a cheap post-hoc top-up only if joint training under-uses it.

---

## S3 — TARGETS + GATES (pre-registered; each with its falsifier)

All byte figures are COUNTED archive.zip payload (token grid + renderer weights + pose). Composed S
uses `S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489`. Budgets `[MEASURED: fc1]`: **≤187,727 B for
0.172**, **≤154,522 B for 0.15** total archive (assuming seg~3e-4 + pose terminal — leaves 0.047 for
seg+pose at each bar). (Charter "212 KB" not located in fc1 this session — flagged.)

| # | Target | Value | Provenance | Pre-registered FALSIFIER |
|---|---|---|---|---|
| G1 | **Native d_seg (no-correction operating point)** | **≤ 5e-4, push to ≤ 3e-4** | `[REGISTERED-LAW: ddm_pp1_correction_stream_position_band_v1]` ρ_c=5.02e-4; PR130 rail 2.97e-4 `[EXTERNAL-LESSONS-ONLY]` | renderer cannot reach ≤1e-3 at ≤64 KB ⇒ learned-amortized realization **falsified at this scale** (ee1 R2); fd1 implicit-GN remains only live realization family |
| G2 | **Correction stream** | **NONE if G1≤ρ_c; else priced band [5e-4,1e-2]** | `[REGISTERED-LAW]` band lemma | if native lands in (5e-4,1e-2] a correction stream is rational at ≤1.2731 B/flip; below ρ_c any correction machinery is dead |
| G3 | **Renderer weights (counted)** | **≤ ~64 KB; bit-depth int4/int5/int8 RACED** | `[DERIVED]` sizing; PR130 40,252 B int4 `[EXTERNAL-LESSONS-ONLY]` | int8 default with no bit-depth race is a cargo-cult violation; a chosen bit-depth that fails G1 falsifies that bit-depth, not the family |
| G4 | **Token stream (counted)** | **≤ ~130 KB, target the ~117 KB learned end** | `[MEASURED: ee1 R6, pp1 R1]` plateau 117–177 KB | token leg ≥ 173.6 KB (explicit-arith level) ⇒ renderer absorbed NO shared structure ⇒ no advantage over explicit route |
| G5 | **Pose (TERMINAL 6-eq solve)** | **d_pose ~2.33e-5-class, ~2 KB, contrib ~0.0153** | `[MEASURED: MEMORY terminal-pose; ee1 R6]` | banked R1 dxi (7.2 KB, contrib 0.1269) is the FALLBACK; if terminal solve cannot beat ~1e-4 d_pose the path is 0.31 (fails) — pose is the decisive lever |
| G6 | **Composed archive total** | **≤ 187,727 B (→0.172); ≤154,522 B (→sub-0.15)** | `[MEASURED: fc1 budgets]` | composed byte-closed exact row > 0.172 ⇒ path does not beat the incumbent target; > 0.19108 ⇒ does not even beat our own submittable |

### S3.1 Pre-registered composed-S arithmetic `[DERIVED, arithmetic shown; ONLY a byte-closed evaluate.py row is authority]`

| config | tokens | renderer | pose | total | d_seg | d_pose | rate | **S** | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A** optimistic corner | 117 KB | 30 KB int4 | 2 KB | 149 KB | 2.97e-4 | 2.33e-5 | 0.099 | **0.144** | sub-0.15 corner |
| **B** spec mid | 130 KB | 64 KB | 2 KB | 196 KB | 3e-4 | 2.33e-5 | 0.131 | **0.176** | matches ee1 R6; just over 0.172 |
| **B'** mid @ d_seg=5e-4 | 130 KB | 64 KB | 2 KB | 196 KB | 5e-4 | 2.33e-5 | 0.131 | **0.196** | at the no-corr EDGE the seg term pushes over — G1 must push to ~3e-4 |
| **C** banked-pose FALLBACK | 130 KB | 64 KB | 7.2 KB | 201 KB | 5e-4 | 1.61e-3 | 0.134 | **0.311** | FAILS — pose leg dominates; proves G5 |

**Honest reading:** the renderer path lands **sub-bar only with (i) pose terminal-solved AND (ii) total
bytes toward the 150 KB corner AND (iii) native d_seg toward 3e-4.** Sub-0.15 is the optimistic corner
(A/PR130-shape 159 KB → 0.151), not the mid case. The decisive levers in order: **pose-terminal (0.11
swing) ≫ total bytes (0.03) > native d_seg (0.02).** Every S row above is DERIVED arithmetic — the
pointer moves ONLY through a byte-closed `upstream/evaluate.py` row (§S6).

---

## S4 — RESUMABILITY + LAUNCH CONTRACT (P0)

- **P0 resumability (non-negotiable):** crash-resumable-from-disk (`--resume-from`) + **per-stage
  checkpoints** at every curriculum boundary (seg-trunk stages, QAT boundary, terminal-pose finish),
  distinct stage-encoded filenames (never overwrite), atomic tmp+rename, save the **EMA shadow** (not
  live weights; decay via `[REGISTERED-LAW: ema_decay_run_geometry_v1]` — from run geometry, NOT the
  flat 0.997 constant), plus periodic intra-stage saves for the long seg trunk `[MEASURED: CLAUDE.md
  resumability non-negotiable]`.
- **Config is DSL-compiled, never ad-hoc argv:** every lever is a `Lever` factory in
  `src/tac/witness_dsl/` composing a `WitnessProgram` whose `.compile_trainer_argv()` emits the launch
  config and `.validate()` fail-closes on any invented flag. New tr1 levers (token-grid geometry
  `D`,`c`; renderer depth/width/bit-depth; description-level roundtrip toggle; hard-subset gate cadence;
  terminal-pose stage) land as `Lever` factories — never hand-added trainer flags. `[MEASURED: CLAUDE.md
  DSL-as-SoT + config_orphan_confound]`
- **Governed launcher route:** `tools/launch_witness_run.py` (a governor REFUSE is information, not an
  obstacle); trainer entry `experiments/train_levelset_witness_realized_through_R_mlx.py`; base config
  `witness_autoconfig.proven_base`. Heavy launches ONLY via the governed launcher (CONTAINMENT: never
  auto-fire paid/GPU) `[MEASURED: CLAUDE.md operating contract].`
- **Memory preflight at the REAL config:** `tools/witness_memory_preflight.py` projects peak RSS from
  the emitted launch (fixed + cf_mx_cache + gt + verdict-batch spike) and REFUSES (rc=4) if projected
  peak > 0.70×RAM; the n600 verdict MUST chunk (`--verdict-batch 32`) per the OOM law. Ceiling 116 GiB
  (op0721). `[MEASURED: CLAUDE.md #205 verdict-batch law + witness_memory_preflight]`
- **Derived event-driven schedule (sched1 — never PR95 8-stage skeletons):** stage transitions fire on
  measured events (seg-trunk convergence knee, QAT-admission, terminal-pose engage), not hardcoded epoch
  counts; l7 is a measured DEFECT and the smooth stage RAISES d_seg — both demoted. `[MEASURED: CLAUDE.md
  v9 event-driven + curriculum caveat]`
- **Warm-start / resume discipline:** on resume, re-anchor the event schedule to the resume epoch +
  geometry (#517/#518/#270) — never replay a stale absolute schedule `[MEASURED: MEMORY warm_start_resume]`.

### S4.1 DSL config skeleton (design target — NOT wired code; builds when the gate fires)

```python
# DESIGN SKELETON ONLY — build target for the BUILD arm; not landed as a wired Lever this arm.
# Lands in src/tac/witness_dsl/ as real Lever factories when MAIN's GO fires (§S5). NO-FAKE: this is
# a documented spec block, not a claim that these levers exist yet.
from tac.witness_dsl.curriculum_dsl import WitnessProgram, Lever  # existing surface

tr1 = WitnessProgram(
    name="tr1_trained_partition_renderer",
    base="proven_base",
    levers=[
        Lever("token_grid_downsample",   choices=[8, 12, 16], intent="grid pitch vs bytes; ERF-bounded"),
        Lever("token_code_width",        choices=[2, 4, 6],   intent="per-cell latent width raced"),
        Lever("renderer_conv_depth",     choices=[3, 4],      intent="RF-7: ~3 layers for placement"),
        Lever("renderer_bit_depth",      choices=["int4", "int5", "int8"], intent="G3 bit-depth-DOF race"),
        Lever("desc_level_eval_roundtrip", default=True,      intent="S2.1 fd2-wall STE across uint8/coder"),
        Lever("chroma_active",           default=True,        intent="SegNet reads RGB — d_seg lever"),
        Lever("pose_objective_weight",   default=0.0,         intent="#383: seg trunk first"),
        Lever("terminal_pose_solve",     default="gn_6eq",    intent="G5 terminal solve on frozen frames"),
        Lever("hard_subset_gate",        default="g3_flipprone", intent="fd1r cheap inner verdict"),
        Lever("full_n600_confirm_cadence", default="stage_boundary", intent="fd1r sparse authority verdict"),
    ],
)
argv = tr1.compile_trainer_argv()   # -> experiments/train_levelset_witness_realized_through_R_mlx.py flags
tr1.validate()                      # fail-closed on any invented flag
```

---

## S5 — FORK-CONDITIONAL SECTIONS (fd2's FORMAL typed verdict is PENDING; NO build inside this arm)

The fd2 receipt has NO top-level typed verdict field — the disambiguation MEASUREMENT is landed
(canary PASS; ×1.0 BLOCK_MOVED_NOT_IMPROVED; ×0.5/×0.25 REALIZATION_GAP), but MAIN owns the formal
typed verdict. The BUILD gate cites that verdict; this arm builds nothing under any branch.

- **(a) verdict = REALIZATION_GAP (EXPECTED — the measured ×0.5/×0.25 rows already read this):** the
  trained renderer **IS the cure** — the realization wall is exactly the uint8/coder staircase the
  renderer's uint8-STE + description-level roundtrip is designed to descend through, and it dissolves
  fd1's measured primary binding dimension (cross-pair transfer) by training on ALL pairs. **BUILD fires
  immediately on MAIN's GO**, SPEC as written, HIGH priority (it is fd1's own Rung-2 routing).
- **(b) verdict = MIXED / block-locality component:** the renderer with **all-pairs in-loop** is STILL
  the route (all-pairs training is the locality cure by construction). What changes: add ee1-R7
  control-token re-solve as a first-class stage (not optional) to repair any residual per-block locality
  the amortized renderer under-fits; keep the hard-subset gate weighted toward the localized blocks.
- **(c) verdict = a late rung-1 surprise (grown-706 shared-DOF descends to ρ_c):** the renderer becomes
  the **capacity rung** on fd1's ladder rather than the immediate cure — SPEC UNCHANGED, priority DROPS
  (fd1 Rung-1 is the cheaper intermediate; the renderer stays queued as the plausible ≤5e-4 route and as
  the object that creates the masks sp1's READY-GATED better-base contract waits on).

Under all three branches the object and its §S1–S4 physics are unchanged; only the fire-order/priority
moves. **NO build in this arm.**

---

## S6 — RECEIVER + R6 CHAIN (decode path, rule-118 boundary, byte-close → exact chain)

- **Decode path through the proven E4/WS1 exporter grammar** `[MEASURED: E4/WS1 codex findings]`:
  `DDME4WS1RuntimeExporterConfigV1` / grammar `ddm_ws1_receiver_closed_warm_start.v1` — an ordered,
  contiguous, gap-free stream partition with per-stream byte-count + offset + SHA-256 + **named receiver
  consumer**, exact archive SHA + parser re-emit, Brotli-Q11 packets (raw-LZMA1 fallback BLOCKED because
  the WS1 grammar consumes Brotli). tr1's two counted streams (token grid, renderer weights) + the pose
  stream each get a named receiver consumer.
- **Exact-consumption bijection = #417** (NOT the charter's #402, which is telemetry-liveness): every
  counted byte must be READ by an actual receiver consumer at decode; counted-but-inert bytes = INERT =
  FAKE `[MEASURED: MEMORY #417 + src/tac/packetir_exact_closure.py unconsumed_trailing_bytes +
  ddm_costate_organ.py unconsumed_*_counted_inert]`. The E4/WS1 "named receiver consumer" per stream IS
  this bijection at the grammar level.
- **rule-118 boundary (stated):** the **renderer weights (learned, video-derived) are COUNTED** in
  archive.zip; the **renderer forward-pass CODE and the token-grid interpreter are GENERIC ⇒ FREE** in
  inflate.py. No video-derived per-frame table may be smuggled into inflate.py as "code" (hide-data-in-
  code fake, guarded by the #417 receiver-consumption bijection + payload-cleanliness) `[MEASURED:
  CLAUDE.md rule-118]`.
- **Byte-close → exact chain (named with existing tools):** compose the archive → **FULL local exact
  protocol** (real evaluator, real bytes, `[macOS-CPU advisory]`, advisory-S decomposition) per the sh1
  chain (`ddm_sh1_compose_and_local_exact_findings`) → **staged Modal contest-CPU row** (the exact-row
  the <$5 budget buys). Byte-close packet builders exist (`tools/build_*_exact_eval_packet.py` family);
  the exact-CPU/CUDA replay is the ONLY authority for a pointer move. `[MEASURED: CLAUDE.md submission
  auth-eval BOTH CPU AND CUDA]`

---

## HONEST BOUNDARIES

- **Design-only.** No training, no scorer jobs, no launches ran. Every S row in §S3.1 is DERIVED
  arithmetic; every byte target is a pre-registered gate. **NOTHING here is a score.** The pointer
  `0.1910828242 [contest-CPU]` is UNMOVED and moves only through a byte-closed `evaluate.py` row.
- **The renderer is a CONJECTURE with a MEASURED-external existence proof** (PR130 int4 → 2.97e-4,
  lessons-only). Its native d_seg at ≤64 KB on OUR vehicle is UNMEASURED — G1's falsifier (cannot reach
  ≤1e-3 ⇒ learned-amortized realization falsified at this scale) is real and pre-registered.
- **Composed sub-bar requires all three legs at their good corner** (pose terminal-solved, bytes toward
  150 KB, d_seg toward 3e-4). Sub-0.15 is the optimistic corner, not the mid case (§S3.1).
- **fd2's formal typed verdict is PENDING** — the BUILD gate cites it; this arm carries fork-conditional
  sections and builds nothing.
- **Three recall discrepancies reported** (charter 706/344 = two distinct objects; charter #402 →
  correct #417; charter "212 KB" not located in fc1). No number consumed on an unverified value.
