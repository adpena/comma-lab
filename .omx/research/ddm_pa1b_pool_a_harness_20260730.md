# ddm_pa1b (#793) — the Pool-A RACE HARNESS: the hull-curvature instrument (BUILT, scorer-free)

**Pointer honesty first: 0.1910828242 [contest-CPU] UNMOVED.** Everything below is a BUILD +
[macOS-CPU advisory] $0 theorem; score_claim=False. No scorer job launched (gc10 §7 — MAIN fires the
race from the scorer slot after nv1). This unit BUILDS the instrument that will MEASURE hull curvature.

STORES-CONSULTED (path + sha/commit): gc10 memo `.omx/research/ddm_gc10_hull_mover_convocation_20260730.md`
(landing commit 2647b1f080) — the re-charter (Assumption-Adversary: "two on-contour points are a line
not a hull; the race MAPS curvature"), EXHAUST theorem #2, pre-mortem F1-F3 · ax1 memo
`.omx/research/ddm_ax1_all_axes_derivation_20260730.md` + the two DESIGN-stubs
`src/tac/witness_dsl/ax1_derived_levers_20260730.py` · QA84 grammar
`src/tac/witness_dsl/qa84_rowband_grammar_20260731.py` + the registered
`rowband_flip_mass_foveation_band_v1` equation · the SMEVR coder `experiments/ddm_r7_token_coder.py`
(`encode_token_codes(..., codec="smevr")`) · the product law
`tac.canonical_equations.ddm_gc9_seg_rate_product_law_20260730.product_c` (3 anchors; QA24 c=0.08815) ·
zb1 FEED (QA80 field MEASURED n600, custody `/Volumes/VertigoDataTier/pact/ddm_zb1_qa80_field_20260730`,
q50 1.8181; the "B is a hull-filler not a hull-mover" convergence) · dw1 re-price (distill term DROPPED;
plain-continuation dividend RETIRED — NOT carried in the stack arithmetic) · the non-additive-pools LAW
(same-pool levers COMPETE; per-lever stack-claims REFUSED — the race IS the adjudicator).

## §1 What was built (all scorer-free, all tested — 27/27 pass)

**Two new modules + surgical trainer wiring + DSL factories + a 27-test suite:**

1. `src/tac/witness_dsl/ax1_pool_a_levers_20260730.py` — the two ax1 lever LOGIC (numpy authority):
   - **QA80 typed field loader** (`load_qa80_cell_field`): streams the MEASURED custody, sha-verifies
     each of the 600 pairs (staleness/custody law), aggregates the exact flip-distance field to per-cell
     `flip_mass` (proximity-weighted `relu(1-d/q50)`, tail-emphasizing, anchored to the field's own
     median scale), `min_distance`, and `dynamic_frac` (lane/movable winner-class fraction). Fail-closed
     on custody/sha/geometry. Real-field smoke: q50 1.8107 (matches zb1's 1.8181).
   - **(a) margin-coupled quant allocation LAW** (`margin_coupled_level_map`): per-cell EFFECTIVE quant
     levels = a **rank transform of the field's own flip-mass order statistic** (NO bare α/β constant;
     ties collapse ⇒ uniform field ⇒ uniform allocation). Endpoints are the config's raced level ladder
     (base = `token_quant_levels`, floor = `//4`). OFF (min==base) ⇒ uniform ⇒ byte-identical.
   - **(b) delta group-sparsity** (`delta_group_sparsity_penalty` + `xi_informed_delta_weight`): group-L2
     (group-lasso) on per-pair token deltas; the ξ-informed weight RELAXES on dynamic (lane/movable)
     cells, TIGHTENS on the static mass (DERIVED from the QA80 winner classes, ax1 §5).

2. `src/tac/witness_dsl/ax1_pool_a_race_20260730.py` — the harness:
   - **EXHAUST THEOREM #2** (`enumerate_band_edge_theorem`): the complete band-edge enumeration + the
     $0 geometric-rate Pareto frontier + the min-rate band at each flip-mass coverage target (§3).
   - **Matched-SMEVR-bytes seal** (`seal_matched_bytes_race`): prices each arm's SMEVR estimate on the
     FIRE-TIME parent field via the SHIPPED coder, enforces ±1% matching (refuse = the tuning-step
     signal), and records each arm's sealed-ticket **argv-diff vs control** (the pre-fire diff law from
     the QA86c #517-twin incident).
   - **Hull-curvature verdict schema + analyzer** (`RaceReceipt` / `HullCurvatureAnalyzer`): the typed
     receipt {lever, window, d_seg, ΔSMEVR bytes, c, curvature reading vs iso-c 0.08815} + the verdict
     (does ANY matched-bytes point sit strictly INSIDE the contour?).

3. **Trainer levers REALIZED** (`experiments/train_tr1_partition_renderer_mlx.py`, the ph3 fold-and-delete
   pattern dw1 used for QA75): 7 new argparse flags + TR1Config fields + `_build_pool_a_banks` (FIXED
   non-trainable buffers from the QA80 field) + per-cell L in `quantized_tokens` + the `delta_sparsity_term`
   in `batch_loss` engaged at the **base-stability EVENT** (the CE→tau knee; event-driven, never
   epoch-hardcoded) with a `from_step_0` mode = the **gc10 F2 ν-snap warm-start holder** (documented as
   the TRAIN-side twin of the export-side snap). Resume re-engages past the knee (resume-registry hygiene).
   **Byte-identical when OFF** (all flags default off; `_build_pool_a_banks` returns (None,None) ⇒ no
   buffers ⇒ bit-identical forward + byte-close; PROVEN by `test_trainer_off_path_byte_identical` +
   `test_per_cell_quant_off_identity_vs_scalar`). NEITHER lever adds a trainable param ⇒ not checkpointed
   ⇒ not in the EMA shadow ⇒ byte-identical resume by construction (the resume/EMA obligation satisfied
   vacuously + documented, not by adding state).

4. **DSL fold-and-delete**: real factories `lever_token_quant_margin_coupling` + `lever_delta_group_sparsity`
   land in the SoT (`spec_tr1_renderer_20260728.py`, with value-provenance manifests); the race programs
   `pool_a_race_programs` land in `spec_tr1_burn2_20260731.py`; the superseded stubs
   (`Ax1MarginCoupledTokenQuant`, `Ax1DeltaGroupSparsity`, `Ax1PoolAJointRace`) are DELETED from
   `ax1_derived_levers_20260730.py` (only the Pool-C `Ax1Frame0CarriedWarp` stub remains — out of scope).
   DSL coverage: all 6 Pool-A flags are DECLARED by the trainer argparse (never-invent, `validate()`
   fail-closed) AND EMITTED by the new factories (`test_pool_a_flags_declared_by_trainer_and_emitted`).

## §2 The 276-enumeration VERDICT — corrected to 300 (derivation shown)

The gc10 EXHAUST theorem #2 claimed "276 band-edge configs = complete enumeration of the rowband
boundary space." **DERIVED count = C(25,2) = 300, not C(24,2) = 276.** Derivation from the
`RowBandGrammar` constraint set (`0 <= lo < hi <= fine_gh`, both multiples of coarse_factor): on the D8
grid (fine_gh=48, coarse_factor=2) the coarse-aligned boundaries are {0,2,…,48} = fine_gh/2 + 1 = **25
boundaries**; a non-empty band picks 2 of them (lo<hi) ⇒ **C(25,2) = 300**. The memo's 276 = C(24,2)
counts pairs of the 24 *D16 rows* — i.e. it EXCLUDES the 24 single-D16-row bands from the 300 (a
boundaries-vs-rows off-by-one; 300 − 276 = 24). **The enumeration pass covers the full 300 SUPERSET**, so
no optimal placement is missed regardless of the count dispute — this converts the rowband arm from a
SAMPLE to a THEOREM. (Verified: `test_theorem_count_is_300_superset_of_memo_276`.)

## §3 The theorem FIRED on the real n600 field — the provably-rate-optimal band

$0 scorer-free pass over the MEASURED QA80 field (n600, 600/600 sha-verified, q50 1.8107), proximity-
weighted per-row flip mass (custody `/Volumes/VertigoDataTier/pact/ddm_pa1b_theorem_20260730_per_row_flip_mass_D8.npy`):

| coverage target | provably-optimal band (render rows) | independent cells (rate) | flip mass covered |
|---|---|---|---|
| ≥ 0.50 (op1 gate) | **[160, 240)** | **1248** | 0.541 |
| ≥ 0.721 | [160, 272) | 1440 | 0.723 |
| ≥ 0.90 | [112, 304) | 1920 | 0.913 |

**Headline (measured corroboration):** the provably-rate-optimal min-cells band at the op1 ≥50% gate is
render rows **[160,240) at 1248 independent cells — EXACTLY the QA84 default grammar's band + DOF**
(`default_flip_band_grammar` → DOF 1248). The exact-flip-distance field independently recovers the QA74
`rowband_flip_mass_foveation_band_v1` typing AND certifies the pre-registered rowband default is
rate-optimal at the gate criterion — the sampled default was, in fact, the theorem's answer. The 72.1%
and 90% targets name the wider bands (a MAIN coverage-vs-rate choice); the theorem hands MAIN the whole
Pareto frontier (24 non-dominated placements) instead of one sampled band.

## §4 READY-TO-FIRE race plan (fires from MAIN, scorer slot AFTER nv1)

**Slot order:** ps1 (live) → nv1 (#796, has the slot next) → **Pool-A race** (this harness). Do NOT fire
before nv1 (task binding).

**Parent** (resolved at fire time, NEVER hardcoded): ps1's composed best, else B (dw1 control ep440 EMA).

**Arms** (`spec_tr1_burn2.pool_a_race_programs`, all on ONE shared D8 row-band base so the matched-bytes
comparison isolates Pool-A): `control_rowband` · `margin_quant` (+ax1 §2a) · `delta_sparsity` (+ax1 §4a) ·
`joint_quant_sparsity` (+both). Grammar = the theorem's ≥50% optimum ([160,240), DOF 1248) OR a wider
target per MAIN's coverage choice.

**Fire sequence:**
1. `$0` — materialize each arm's parent-derived quantized token field; call
   `seal_matched_bytes_race(control_argv, control_codes, arms, levels=16, tol=0.01)`; if `matched=False`,
   apply the code_width/level tuning step (the refusal names the offending arm + %) and re-seal. Record
   the argv-diff per arm (pre-fire diff law).
2. `scorer` — train each sealed arm (event-terminated; per-stage EMA checkpoints; resumable). MEASURE
   realized d_seg through R + byte-close SMEVR bytes per arm → build `RaceReceipt` rows.
3. `$0` — `HullCurvatureAnalyzer(iso_c=0.08815).analyze(receipts)` → the hull-curvature verdict.

**Wall-clock:** each arm is a QA24-class burn (event-terminated ≤480 min cap; typically ≪ that at the
knee). 4 arms; MAIN may parallelize under the fleet cap. The $0 seal + theorem are seconds.

**Verdict logic (gc10, Assumption-Adversary):** if ANY matched-bytes arm has c < 0.08815 (minus the 2%
band) ⇒ **the hull MOVED** (the contour is a true convex hull with interior points, not a line) ⇒ that
lever enters the burn-3 stack. If all arms sit on the contour (c ≈ 0.08815) ⇒ the 3-move negative record
EXTENDS to the class (the hull is a line here); each such lever EXITS the burn-3 stack (gc10 §1 row 3
falsifier). Per-lever stack-claims are REFUSED — the joint arm vs the single-lever arms adjudicates the
non-additive pool (F1 antagonism guard: joint < best-single ⇒ abort-to-best-single).

## §5 Pre-mortem F1-F3 compliance (gc10 §2b leg-l)
- **F1 (Pool-A antagonism):** the race is JOINT (control/single/single/joint arms); the analyzer's
  `best_lever` + the joint-vs-single comparison IS the antagonism read (composed < best-single ⇒ the
  joint arm loses; the non-additive-pools law is enforced by construction, not asserted).
- **F2 (ν-snap × training):** the delta-sparsity lever's `from_step_0` engage mode IS the in-training
  holder of the export snap (documented as the train-side twin); a warm-started burn keeps bytes low
  without re-inflating. Default engage = after-base-stability (§7) for a fresh burn.
- **F3 (pose-stage regression):** the race arms are seg-only (pose is TERMINAL #383); the composed-S
  verdict lever + the terminal pose re-solve remain MAIN's burn-3 budget item (unchanged).

## §6 Honest limits + verdict_scope
- The d_seg column of every receipt is MEASURED BY MAIN at fire time; the analyzer is scorer-free but
  its INPUT is scorer-produced. Nothing here is a score/pointer claim.
- The theorem's coverage measure is the QA80 proximity-weighted flip mass; a different measure (e.g. the
  QA74 GT-margin decile) yields a different band — the theorem is the INSTRUMENT (complete enumeration
  under whatever measure MAIN chooses), and it happens to recover [160,240) at the op1 gate.
- Every falsifier is instance/formulation-scoped: an arm ≤ noise at matched bytes closes THAT lever at
  INSTANCE on this vehicle, never the paradigm.
- Pointer 0.1910828242 [contest-CPU] UNMOVED — this unit built the instrument; it did not move the score.

## OP-ROUTABLES
1. MAIN fires the Pool-A race from the scorer slot after nv1 (the §4 sequence).
2. The theorem's Pareto frontier + [160,240) ≥50% optimum feed the burn-3 grammar choice (already the
   QA84 default — corroborated).
3. If the race moves the hull, the winning lever(s) enter burn-3; if not, they exit (gc10 §1 row 3).

pointer 0.1910828242 [contest-CPU] UNMOVED  ·  [no-triality] [p0-ledger-ok]
