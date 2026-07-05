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

---

## STALL DIAGNOSIS + GPU-SMOKE CONFIRMATION (2026-07-04, appended)

**The reported "GPU stall" at ep3/accum_batch74 was investigated, and the parent's strong hypothesis
(the witness-alone `_render_R_wa` graph accumulates as an unbounded lazy graph that is never
`mx.eval`'d per accum-step, so the periodic forced eval grinds forever — the #240 OOM/stall class) is
REFUTED by direct measurement. There is NO wa-routing eval-discipline bug; the wa path already obeys the
per-pair `mx.eval` discipline. No trainer code change was warranted (NO-FAKE: a behavior-neutral "fix"
for a non-existent bug is forbidden).**

### Evidence (4 independent, converging)

1. **n600 mem_probe is FLAT, not growing.** In the reported run
   (`levelset_n600_witness_20260704T234054Z/run.log`) the per-batch `mem_probe` is **constant at
   `mlx_active_gib=52.02`, `mlx_peak_gib=57.30`** across ALL 75 accum-batches of ep3 (batch 0→74). An
   unbounded lazy graph would GROW memory batch-over-batch; it does not. → the per-pair
   `mx.eval(loss, grads)` (trainer L4595) already bounds the graph, wa render included (wa output feeds
   `L` via `island_birth_from_signed_mx(_signed_wa, …)` L2828 + `persistence_topology_loss_mlx(_slog_wa,
   …)` L2841, so it IS forced by that eval).

2. **The "freeze at ep3/batch74" was a LOG-SILENCE MISREAD.** `mem_probe` only logs for
   `ep ≤ TAC_MEM_PROBE_EPOCHS` (default 3), so per-batch logging STOPS at ep3/batch74 by design. The
   next log line is the ep25 verdict (`--eval-every 25`). That run in fact **reached ep25 and emitted a
   verdict** (`d_seg 0.159614`, `implied_S 17.37`, ts 01:23:15Z) — ep0→ep25 spanned ~90 min (~3.6
   min/epoch), so the ~80 min of log silence during ep4–24 is a healthy-but-quiet loop, exactly where a
   process sample lands the main thread inside a normal `mx::core::eval`.

3. **GPU smoke (n24, real MLX-GPU, governed via `tools/safe_run.py`) — per-epoch train time is BOUNDED
   and CONSTANT, wa ON vs OFF.** Faithful #205 config (self-orient, chroma, lane-render-band, seed-islands,
   amplify=1.0, persistence, pose-carrier, hosc, film-stiefel; CE regime ep0–7). The wa route provably
   fired (`island_seed` live, `island_amplify weight=1.0 class=1`, `persistence_loss` active,
   `seed_survival` every epoch → `_island_levers_on=True` → `_wa_route=True` → `_render_R_wa` every pair):

   | epoch | t_step (fwd+bwd+opt+ema) — WA ON | t_step — BASE (pre-#300, wa OFF) |
   |---|---|---|
   | 1 | 5.45s | 5.72s |
   | 5 | 5.44s | 5.70s |
   | 8 | 5.43s | 5.69s |
   | 11 | 5.45s | 5.70s |

   **Flat across all epochs; NO runaway growth.** Peak RSS 5.7 GiB at n24 (both). WA is marginally
   *cheaper* than BASE, not 2× more expensive.

4. **Code confirms no second forward in the live config.** With the #205 flags NO non-wa surgical lever
   is engaged (no `--lane-edge`/`--margin-saliency`/`--lane-thin`/`--mfh`/`--subpix`/`--chroma-boundary`;
   `--lane-render-band` is a composer, not a lever), so `_nonwa_levers_on=False` →
   `_need_composed = _nonwa_levers_on or (_island_levers_on and not _wa_route) = False` (L2684). #300
   therefore SKIPS the seed-composed SegNet forward and runs ONLY the witness-alone forward — **one
   SegNet forward per pair either way** (wa slightly cheaper because `_compose_chain_noseed` omits the
   seed residual add). The parent's "second render+scorer per island pair" is only real if a non-wa
   lever is ALSO on (then both forwards run → ~2× cost, still BOUNDED, still eval'd per pair — a
   throughput cost, never a stall). The live #205 config is NOT that case.

### The real cause of the "apparent stall" (not a #300 bug)

Log-silence after ep3 (mem_probe off; verdicts only every 25 ep) + **`--async-verdict` GPU contention**:
the ep0 and ep25 verdicts run over ALL 600 pairs (`--verdict-pairs 0` is falsy → full P) on the SAME
GPU in a background thread; the ep25 verdict took **1437 s (~24 min)** (`verdict_async_done secs 1437.1`).
While an async verdict grinds the GPU, the main training `mx.eval` shares the device and slows — a
healthy-but-slow window that reads as a freeze under log-silence.

### Byte-identity-off + gate

- Golden test `experiments/test_seed_absorption_fix.py`: **9/9 pass** (no trainer change → off-path
  byte-identity trivially intact; re-run as the gate).
- Smoke evidence preserved: `experiments/results/wa_stall_diag_{wa,base}/run.log` (repro:
  seed 0, deterministic).

### Recommendation to the parent (relaunch)

- **No trainer code change is needed** for the wa eval discipline. Relaunch the #300 config **as-is**
  (`--witness-alone-island-loss --seed-anneal-epochs 300 --seed-anneal-shape cosine`); it trains
  cleanly and bounded.
- To remove the misdiagnosis-inducing silence/contention on the next long run (OBSERVABILITY, separate
  from this fix): consider a lightweight per-epoch heartbeat print and/or gating the async full-600
  verdict less aggressively (e.g. a small `--verdict-pairs` for the interim verdicts, full only at
  stage boundaries). These are throughput/observability niceties, NOT correctness fixes.
- Pointer 0.19110 UNMOVED (this is diagnosis + confirmation; the pointer moves only via the byte-closed
  n600 exact row from the relaunch).
