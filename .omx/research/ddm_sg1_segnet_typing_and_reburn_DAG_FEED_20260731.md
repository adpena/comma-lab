# FEED — ddm_sg1 (QA74 typing · QA24 config derived, NOT fired · QA75 scaffold) — 2026-07-31

POINTER `0.1910828242 [contest-CPU]` UNMOVED. `[macOS-CPU advisory]`; score_claim=false. The burn did
NOT fire (§7 of the memo).

- **QA74 FIRED-MEASURED (the decisive $0 typing):** renderer endpoint d_seg 0.00388778 (n600, reproduces
  gr1). **≥96.1% of the residual is AMORTIZATION GAP** (attackable) vs the EXACT-solve floor (17,927 err
  = 1.52e-4); gap 25.58× (the operator-corrected exact-solve reference, not the box 3.35×). **Lane is
  renderer-reach-limited (69.5× over its exact-solve floor), NOT SegNet-stride-limited** — and is the
  dominant error class (38.7% of flips, 25.7% in-class rate). **100% of flips at small GT-margin (the
  boundary annulus).** 80% of flip mass in SegNet rows 160-240. Hood well-handled (1.0%).
  Harness `experiments/ddm_sg1_residual_typing.py` (6648db9a6e). Receipt `ddm_sg1_20260731/
  sg1_typing_receipt.json`.
- **QA24 GRID DERIVED + VALIDATED:** keep the 384 cell_drop50-kept cells (grid rows 5-19; drop sky+hood
  from birth) → captures **99.61%** of flip mass. Pose caveat: sky/hood freeze costs pose (co9) → REQUIRES
  the QA77-lite composed-S verdicts. Mask `qa24_grid_keep_mask_50.npy` (SSD).
- **QA24 CONFIG = 5 COMPOSED PIECES (derived, provenance+falsifier each):** (1) coarse-grid cell-mask
  [new trainer feature], (2) margin-weighted loss [#1 form fix — burn is margin_weighted=False vs 100%
  flips at small margin], (3) QAT-dynamics as sched events [lattice-annealing + redistribution + dither],
  (4) rate-in-loss [MAIN §9.1, stl1 row-8 LAW, CONFIRMED], (5) QA77-lite composed-S verdicts [MAIN §9.2,
  CONFIRMED — kills the Knee-A externality]. tt1 CONSUMED (§9.3): pose polish reduced, twin analytic
  gradient adopted for the re-solve.
- **QA24 DID NOT FIRE (honest):** the binding operator corrections + MAIN §9 gate the fire behind an
  ATOMIC 5-piece composed build (multi-surface, resume-sensitive); firing the un-composed T3 config = a
  weaker state (correction-2 forbids) + the dispatch-at-lifted-form trap. DERIVED + staged; handed to MAIN
  with the governed-chain fire path (launcher READY, `--dry-run` validates all gates). Pointer UNMOVED.
- **QA75 SCAFFOLDED (research_only):** solve-distillation stage config
  `ddm_sg1_qa75_solve_distill_stage_20260731.json`; BLOCKED on exact-solve frame/label materialization
  (409MB inflate, owed). Falsifier: distilled ≤ CE at matched budget → gap distillation-curable.
- **APPARATUS:** fixed the shared-venv tac-hijack (eg1 worktree editable-install → restored to main src).
  OWED guard (pth-scan for codex_worktrees) still un-built → MAIN.
