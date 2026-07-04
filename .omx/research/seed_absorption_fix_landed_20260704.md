# BUILD #300 — SEED-ABSORPTION FIX LANDED (2026-07-04)

**Pointer 0.19110 UNMOVED** (this is a trainer fix + $0 diagnosis-confirmation; the pointer moves only
through a byte-closed n600 exact row from the fresh relaunch). Operator GO'd the fix+relaunch.

## What was broken (measured root cause)
The CE plateau of the seeded witness (`levelset_n600_witness_20260704T174257Z`, pid 5873) is
**SEED-COMPOSE ISLAND-GRADIENT STARVATION** (memo `plateau_disambiguator_results_20260704.md`, commit
`75a1038a0`; memory `seed_compose_island_gradient_starvation_the_crutch_that_blocks_learning`).
`_compose_chain` adds the deploy-EXCLUDED island seed (lane=1, movable=3) to the SegNet-scored frame1;
EVERY realized-through-R seg lever read that seed-composed frame, so once the seed satisfied the loss
on the island, `∂L/∂witness ≈ 0` there → the witness never learned to FORM lane/movable → deploy
(witness-alone) has ~0 island mass. MEASURED: 71% of the plateau = exactly the 2 seeded classes at 100%
within-class flip. #205 (no seed) FORMED islands → it is NOT capacity/mod-dim.

## The fix (two coupled, DEFAULT-OFF mechanisms in the LEVELSET trainer)
File: `experiments/train_levelset_witness_realized_through_R_mlx.py`.

**(a) `--witness-alone-island-loss` [CORE absorption pathway].** The island-FORMATION levers (`#224`
island **amplify** + **persistence**) now read a **witness-alone** realized margin/logits (`_signed_wa` /
`_slog_wa`) computed from a seed-EXCLUDED render (`_render_R_wa` → `_compose_chain_noseed`), instead of
the seed-composed `_signed`/`_slog`. This is the `eval_roundtrip` discipline applied to the island: the
levers now push the WITNESS to reproduce lane/movable itself (the deploy surface). The seed still
composes for the OTHER levers (base CE etc.) + nucleation-init; the seed is ABSENT from
`_compose_chain_noseed` ⇒ `∂(_signed_wa)/∂seed = 0` ⇒ the seed correctly gets NO gradient from these
levers. `total_loss_fn` lines ~2644–2799 (split-gating + wa render), amplify ~2825, persistence ~2839.

**(b) `--seed-anneal-epochs N [--seed-anneal-shape {linear,cosine}]` [TRANSFER schedule].** The island
seed's compose weight ramps full(1.0)→0.0 over epochs `[1, N]` (module fn `seed_compose_weight_at_epoch`,
~line 969; read LIVE in `_compose_chain`, ~line 2166; set per-epoch in the loop, ~line 4400). By the
anneal end the composed frame == the witness render (deploy surface). Nucleation preserved early;
crutch removed before the tau/MCF stage erodes sub-critical island structure.

Serial-path only (mechanism a): fails CLOSED under `--micro-batch-pairs` (NotImplementedError) rather
than silently ignoring the routing. Mechanism (b) composes with micro-batch (it lives in
`_compose_chain`). New levers recorded in the resume-drift guard (`_resume_lever_divergences`) so a
`--resume-from` that drops/changes them fails closed (deterministic-repro). `math` import added.

## Byte-identity when OFF (both flags default OFF)
`--no-witness-alone-island-loss` (default) ⇒ `_wa_route=False` ⇒ `_signed_wa`/`_slog_wa` **alias** the
seed-composed `_signed`/`_slog` (SAME objects) ⇒ the island levers consume identical tensors, no 2nd
forward; the split-gating's `_need_composed` reduces EXACTLY to the pre-#300 `_seg_levers_on` OR.
`--seed-anneal-epochs 0` (default) ⇒ compose weight stays 1.0 ⇒ the `if _cw != 1.0` guard is False ⇒
`_compose_chain` emits `rgb + res*mask` verbatim (bit-identical). ruff: 27 findings before AND after
(0 new; 0 F821). Empirical corroboration: the ep0 witness-alone verdict is IDENTICAL ON vs OFF
(d_seg 0.507463) — the new flags do not perturb the deploy render.

## Tests (`experiments/test_seed_absorption_fix.py`, all $0; MLX CPU authority)
- **8 fast unit tests (PASS):** `seed_compose_weight_at_epoch` off=constant-1.0 / linear full→0 / cosine
  full→0 / monotone+bounded / edge anneal_epochs=1 (no div-by-zero); compose-weight byte-identity
  (weight 1.0 == `rgb + res*mask` bit-exact) / removes seed at 0.0 / scales at 0.5.
- **1 real-scorer confirmation (PASS, slow, $0):** the memo's pending **D1**. On real gt_n6 + the real
  frozen MLX SegNet + the real `render_through_R_mlx`, the island-birth gradient w.r.t. the witness is
  **STARVED through the seed-composed margin (|grad| = 2.1e-3, L=2e-5)** but **NONZERO through the
  witness-alone margin (|grad| = 45.3, L=3.15)** — ratio **~21000×**. This DIRECTLY confirms the
  absorption pathway the fix restores (and the diagnosis mechanism).
- Sibling regressions clean: `test_batched_seed_cograd.py` (6), `test_fixall_wave_a_trainer.py` (17).

## Runnability (measured on CPU; GPU step deferred to relaunch)
Trainer LAUNCHES with the new flags: island_seed / island_amplify / persistence_loss all build, ep0
verdict runs (forward path). The full training-STEP backward could NOT be exercised on CPU: the witness
backward hits `src/tac/local_acceleration/metal_grouped_conv_backward.py:268` (`[metal_kernel] Only
supports the GPU`) — a **PRE-EXISTING** GPU-only kernel that fires IDENTICALLY with the new flags OFF
(exonerating this edit; the backward machinery is UNTOUCHED). A GPU smoke was NOT run: the live 77 GiB
run holds the GPU and concurrent >128 GiB is the P0 machine-crash gate. **Residual (honest):** the full
GPU value_and_grad step with wa-routing is confirmed only at the relaunch; the wa-routing MATH
(render→SegNet→island_birth→grad) is confirmed on CPU by the D1 test, and my edit only adds a 2nd
forward + reroutes which tensor the island levers read (SDF/conv backward untouched).

## Recommended relaunch config (the parent fires it; do NOT auto-launch)
Base = the live `fresh_seeded` launch (`levelset_n600_witness_20260704T174257Z/launch.sh`), unchanged,
PLUS the two absorption flags:
```
--witness-alone-island-loss \
--seed-anneal-epochs 300 --seed-anneal-shape cosine
```
Rationale: `--seed-anneal-epochs 300` == `--tau-softplus-start-epoch 300` so the seed is fully
transferred to the witness BEFORE the tau/MCF stage; cosine holds the seed longer through early
nucleation then drops. (a)+(b) are complementary: (a) gives the witness the absorption gradient
throughout; (b) guarantees the training surface == the deploy surface by tau. The verdict watch stays
the witness-alone per-class island flip (should now FALL through CE, not stay ~100%). Fresh run
(resumable, per-stage checkpoints), CPU authority verdict, GROUPED_BACKWARD=1 on GPU. Watch Movable
(50% of d_seg, NO analytic fallback) — verify its witness-alone flip falls.

## 6-hook wire-in declaration (per Catalog #125)
1. **Sensitivity-map:** N/A — this is a training-time gradient-routing fix (which render the island
   levers read), not a per-axis byte-sensitivity contribution.
2. **Pareto constraint:** N/A — no new archive bytes (the seed is 0-byte accelerant; the fix reroutes
   gradients only).
3. **Bit-allocator hook:** N/A — no per-tensor byte allocation change.
4. **Cathedral autopilot dispatch:** N/A — no archive-deployable artifact; feeds the manual relaunch.
5. **Continual-learning posterior:** ACTIVE via memory
   `seed_compose_island_gradient_starvation_the_crutch_that_blocks_learning` + this memo (the
   compose-time-crutch-starves-the-gradient meta-pattern is the durable learning).
6. **Probe-disambiguator:** the D1 gradient-flow test IS the disambiguator (composed-starved vs
   witness-alone-flows), confirming the mechanism over the capacity/bandwidth/convergence alternatives.

## Triality (DAG ↔ DSL ↔ equations)
DAG: append a FEED block for BUILD #300 (root cause → fix → D1 confirmation → relaunch config). DSL:
the `--witness-alone-island-loss` + `--seed-anneal-epochs/shape` flags are the new argv surface. Equation:
`seed_compose_weight_at_epoch` (the transfer schedule) + the absorption invariant (any compose-time
assist on a scored forward MUST anneal→0 OR route the absorption gradient through the deploy surface).

Commit shas: (recorded at serializer commit). Sisters:
`plateau_disambiguator_results_20260704.md` · `lane_nucleation_failure_seed_above_critical_nucleus` ·
`council_grand_symposium_ce_plateau_20260704.md`.
