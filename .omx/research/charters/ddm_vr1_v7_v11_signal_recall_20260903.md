# ddm_vr1_v7_v11_signal_recall — the operator's steer "we have a lot of useful signal, code and md from the v7–v11 era and since": mine that era's SPECs, kernels, laws, DSL levers, and measured rows for what plugs into the TWO live doors (born-field optimization; the fold-back training program) — a ranked fold table with code paths, receipts, and S-arithmetic, never a re-derivation

## MANDATE

Operator 2026-09-03: *"We have a lot of useful signal, code and md, from the v7-v11 era and since"* + *"all of the
stuff we have discovered … points to improvements to the training and other steps themselves"* + standing GO. The
fold-back program (`ddm_fb1_foldback_program_20260903.md` ff44a90ad) mapped TODAY's post-hoc laws to training levers;
the v7–v11 era (2026-07-08 → 07-22: SPEC_v75 optimal single trunk, SPEC_v8 per-class carriers, the v9 Bregman
all-surfaces stack, the v10 compiler/receiver spine "q1 = scorer CONTROL; plane-family RATE-DEAD; crux = REALIZATION",
the v11 obligation-vocabulary solve) built the deep-math substrate the current vehicle inherited only in pieces:
the CE1 expected-flip margin law, the level-set/Morse→persistence flow (F1b), the Fisher-metric/UNIWARD margin
surrogate (cosine sign-flips: m65/m108), the KKT capacity-routing waterfill, the AA-SDF and curvelet/shearlet
kernels, the realization ladder (lr2/hr1), the costate organ, the lever families in `src/tac/witness_dsl/`, the
custom Metal kernels. Recall-before-decide is the law (CLAUDE.md OPERATOR PRIORITY §1); "re-deriving or ignoring
our own work is the cardinal sin." This arm pays that debt for the fold-back: nothing gets built again that
exists, and nothing measured then gets re-measured now.

## SCOPE (read-only research; the deliverable is a table that other arms consume)

1. **Inventory the era at source** (code paths + commit pins + the memo/receipt that measured each):
   `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`, `SPEC_v8_perclass_decomposition_20260708.md`,
   `bregman_v9_all_surfaces_{build_spec,measurement,binding}_20260714.*`, `BUILD_SPEC_v10_compiler_receiver_20260718.md`,
   `campaign_meta_adversarial_review_v9c2_to_v10_20260718.md`, `codex_findings_ddm_v10_fisher_g2cs1_event_solve_20260722_codex.md`,
   `codex_findings_ddm_v11_obligation_vocabulary_solve_20260722_codex.md`, `codex_findings_ddm_v9_carrier_compose_byteclose_20260722_codex.md`,
   `docs/triality_dag_dsl_equations_deepmath.md`, `docs/vehicle_operating_system.md`, the DAG
   `.omx/research/sub015_DAG_*`, `src/tac/witness_dsl/{ax1,bi1,cw1,fh1,hg1,p4x,ph3,pt2,rb1,spec_c1}*.py` (lever
   factories; use `tac.witness_dsl.lever_registry.completeness()` — never hand-grep levers),
   `src/tac/boundary_math/*` (aa_sdf_observation_render, boundary_solver 7237d3eee, curvelet/shearlet bindings,
   chroma_boundary_match, contour_codec, context_partition_codec, …), `src/tac/ddm_costate_organ.py` (cd678f402),
   `src/tac/lie/`, the Metal kernels, and the canonical equations registry (465 rows: `tools/list_canonical_equations.py --json`;
   filter the v-era prefixes and the `EmpiricalAnchor` rows).
2. **For each artifact** answer with receipts: what it measured or built (n, axis, vehicle); whether it is LANDED CODE
   (path + commit), a MEASURED LAW (equation id + anchor), or MEMO-ONLY; and WHERE it plugs into the live doors:
   (a) the born trainer `experiments/ddm_qbt1_qbflow_trainer.py` / QBR1 successor configs (optimization door),
   (b) ft1's renderer fine-tune loss / the fpc3 population trainer (fold-back), (c) the DSL lever registry (a lever
   is not built until it is a `Lever` factory), (d) the equations leg. Price its expected effect in S-arithmetic
   where the receipt allows; otherwise UNPRICED.
3. **Ranked fold table** (≤ 25 rows): artifact · era · status · plugs-into · expected effect (derivation) · cost ·
   verdict {FOLD-NOW (a concrete lever/loss/kernel with a path) · FOLD-AFTER-BURN · ALREADY-IN (cite where) ·
   SUPERSEDED (cite what superseded it, with the receipt) · MEMO-ONLY-NEEDS-CODE}. Explicitly cover: the CE1 margin
   law's ladder; the Fisher/UNIWARD margin surrogate as a hard-site weight (fb1 item 3); the KKT capacity routing
   (Lane gets the bits); the AA-SDF raster + curvelet basis for a class-matched born form; the realization ladder;
   the costate organ's duty-to-measure queue; the v10 "q1 = scorer CONTROL" spine; the persistence (F1b) warrant.
4. **Update `ddm_fb1`'s map** — append-only section in your memo listing the rows fb1 missed; MAIN folds them.
5. **Gestalt line:** does anything from the era change where sub-0.12 lives (`ddm_gs3` 91ebf77a4 + addenda)?

## HARD CONSTRAINTS

- Read-only: NO builds, NO scorer/Modal/Metal, NO edits to live-arm files (ft1, ql2), the packet, or upstream.
- Every claim carries its receipt path; a row without one is UNPRICED/UNVERIFIED, not ranked. Never quote a
  number from memory — open the file (m44: never recall from working memory alone).
- Serializer commit of the memo with post-edit `--expected-content-sha256`.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- CLAUDE.md "SUPERSEDED PRODUCTION ROUTING FOR ITEMS 1–2 (2026-07-27)": the −48% directional-basis figure was a
  PROXY; self-orient default OFF; directional bases re-enter only via a matched n600 A/B — cite, don't re-litigate.
- CLAUDE.md "Morse–Smale reading RETIRED (mf1)": persistence survives (F1b), Morse–Smale does not.
- `ddm_ww1_walls_that_werent_20260902.md` — which v-era walls dissolved (config, toy-scale, stale constants).
- `ddm_rn1_n600_reopen_sweep_20260903.md` — closure roster (4,027 UNSTATED-N): a v-era verdict without n is not a closure.
- memory `m34` (NO OLD LINEAGE: HNeRV/PR95/110/128 = lessons only) — the v-era's OWN kernels/laws are ours and
  in scope; borrowed-lineage vehicles are not.

## OPTIMAL FORM

- Family exemplar (reference): the recall discipline of `ddm_rn1_n600_reopen_sweep_20260903.md` (commit
  `git log -1 --` it; receipts on APDataStore) and the era's own index `docs/triality_dag_dsl_equations_deepmath.md`;
  provenance pins: fb1 ff44a90ad, gs3 91ebf77a4, costate organ cd678f402, boundary solver 7237d3eee.
- SCOPE reductions: ≤ 25 ranked rows; the era window is 2026-07-08 → today. MECHANISM reductions FORBIDDEN: no
  row without a receipt; no "should work" without an anchor.
- **PRIOR-LAW PREDICTION (falsifiable):** ≥ 5 v-era artifacts are FOLD-NOW with landed code paths that the
  current born trainer / fold-back does not yet consume (the lever registry's `.unmapped` will show them).
  FALSIFIER: fewer than 3 — count it plainly; it means the era was already folded.

## DELIVERABLE

`.omx/research/ddm_vr1_v7_v11_signal_recall_20260903.md` — the inventory, the ranked fold table, the fb1
append-only additions, the gestalt line, RECALL EVIDENCE, NEXT_IF_RESUMED, DEAD-ENDS. Commit via the serializer.
Cite `docs/operating_manual_craft_handoff.md`. End with the own-vehicle frontier line.
