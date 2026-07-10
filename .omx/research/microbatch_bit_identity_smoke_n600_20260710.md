# Micro-batch twin bit-identity smoke — verdict for the v7.5.2 pointer relaunch (2026-07-10)

**Task #410** (operator-GO 2026-07-10 "clear the residual items first" + T2 council Contrarian
revision). **Question:** is the `--micro-batch-pairs` batched twin *bit-identical* to the sealed
serial `--accum-pairs 8` path at n600, so the free speed lever can ride the **baseline v7.5.2
pointer relaunch**? Score-correctness > speed: if NOT bit-identical, the pointer run must NOT use
micro-batch.

Pointer `0.19108282 [contest-CPU]` — MEANS. This unit moves no pointer; it clears a
config-correctness gate before a pointer run.

---

## VERDICT: NOT bit-identical → micro-batch speedup **NO-GO** for the v7.5.2 pointer relaunch

Two **independent, MEASURED** reasons, either of which alone is decisive:

### (1) STRUCTURAL — the sealed config is *un-runnable* with micro-batch (the decisive blocker)

The sealed v7.5.2 launch (`experiments/results/__v752_drystart_final__/launch.sh`) sets
`--seg-chroma-boundary-weight 0.1`. That lever is one of the **genuinely-unrouted** legs vs the
batched twin (trainer L1117-1119, L5491-5493). The trainer **hard-raises** when the two are
combined:

```
experiments/train_levelset_witness_realized_through_R_mlx.py:5892
    if chroma_bnd_w > 0.0:
        if _use_micro_batch:
            raise ValueError(
                "--seg-chroma-boundary-weight>0 is not supported with --micro-batch-pairs>1 "
                "(the batched twin does not consume the chroma-sharpening lever yet); "
                "run this arm at --micro-batch-pairs 1.")
```

**MEASURED (this session):** ran the sealed flag-set (`--seg-chroma-boundary-weight 0.1` +
`--micro-batch-pairs 2`, all other sealed levers, `--fused-r-kernel`) and it raised **exactly**
that `ValueError` before any training step. This guard is a pure config-compatibility check
(`chroma_bnd_w>0 AND micro_batch>1`) — **n-independent by construction**, so the verdict is
identical at n6 and n600 (no toy-vs-scale gap for a boolean config guard). Demonstrated at gt_n6
(reaches the guard fast); the sealed config would raise the same at n600.

Consequence: you **cannot** run the sealed v7.5.2 pointer-relaunch config with the micro-batch
speedup at all. To use micro-batch you would have to **drop/replace** `--seg-chroma-boundary-weight`
(a sealed lever) — i.e. change the sealed config, which is no longer "the baseline relaunch."

### (2) NUMERICAL — the batched twin is fp-*equivalent*, not bit-*identical* (trajectory-affecting)

Even setting the chroma-boundary blocker aside, the batched twin is **not** bit-identical to the
serial accum path. The batched path replaces the serial per-pair `value_and_grad` +
accumulate-then-divide (the accum loop's mean-over-chunk) with a single batched forward + one
`mx.mean` over the stacked (B,…) axis. The per-pair math is identical, but the **floating-point
reduction order differs** (float addition is non-associative). The code documents this in its own
words:

- Module `tac.boundary_math.levelset_micro_batch_loss` docstring (L16-19): `batched_realized_loss(B
  pairs) == mean_b single_realized_loss(pair_b)` **within fp tolerance** — "the mean-over-B is the
  only reduction re-order."
- Trainer L5388-5389: "**NOT bit-identical to the serial path (batched fp reduction order): a
  trajectory-affecting opt-in** validated by a short A/B."
- Test-suite headers (the authors' own position):
  - `src/tac/tests/test_levelset_micro_batch_loss.py` L20-25: on **MLX CPU** the reductions are
    batch-independent → batched == serial to fp32 machine precision (isolates the *math*); the
    "**GPU batched-fp-reduction noise (~1e-3, from GPU matmul kernel tiling on the pose path) is
    the acknowledged non-bit-identity** of the `--micro-batch-pairs` opt-in, validated end-to-end
    by the trajectory A/B smoke, not here."
  - `experiments/test_batched_seed_cograd.py` L18-27: real gt + real frozen MLX scorer, MLX-CPU:
    loss rel `9.4e-08`, per-group grad rel `≤5.9e-05`; "batched re-run bit-identical (0.0) → the
    deviation class is the pre-existing batched-scorer fp reduction noise … **(trajectory-affecting,
    GPU ~1e-3)**."

**MEASURED (this session, live, real frozen scorer, MLX-CPU, this machine):**
- `test_levelset_micro_batch_loss.py`: **70 passed** (1.48 s) — pins batched-grad == mean-of-per-pair
  to `<1e-4` rel (tolerance, NOT exact).
- `test_batched_seed_cograd.py -m ""` (slow real-path proof): loss rel **1.015e-07**; group
  (0,1,2,3) grad rel **2.2e-05** (witness) / **4.3e-05** (seed); **tail group (4,5) grad rel 1.32e-04
  (witness) / 5.85e-04 (seed)**. The tail exceeded the test's own `2e-4` seed headroom — a
  **pre-existing test-tolerance flakiness** (denominator cancellation in the small-||mean|| tail,
  per the test's own docstring), **not** a mechanism defect and **not** introduced here (git shows
  these files untouched). It *reinforces* the verdict: the batched-vs-serial deviation is non-zero
  and draw-dependent.

Key substrate point: the **pointer relaunch trains on MLX-GPU** (M5 Max; the sealed launch uses
`TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`, `--fused-r-kernel`, GPU). On GPU the batched pose-path matmul
tiling injects **~1e-3** fp noise vs the serial path — **not** bit-identical on the very substrate
the pointer run uses. Note `--fused-r-kernel` (L70/#348) fixes the *render* VJP scatter order; it
does **not** address the *batched-scorer* pose-path reduction, which is the micro-batch drift
source. So "fused-r makes it bit-identity-safe" does not transfer to this drift.

Under the measured chaotic-amplification of this optimizer stack (Muon + weakly-driven pose;
EMA-shadow lag), a ~1e-3 per-step grad delta compounds → the trained weights, archive bytes, and
exact score of a micro-batch run **diverge** from the serial-accum run the sealed config was
validated as. That is a *different run*, not a free speedup.

---

## What "bit-identical" means here, and the smoke design

"Bit-identical for the relaunch" = *would swapping serial-accum for micro-batch leave the trained
weights (hence archive bytes, hence exact score) unchanged?* That requires per-step grads to match
**bit-for-bit** (exact array equality), because any per-step delta compounds over the multi-thousand
epoch curriculum. It does **not** — the equivalence is a `<1e-4` fp tolerance, and the trainer/tests
explicitly label it "trajectory-affecting."

**Smoke executed:**
1. Live-ran the two existing equivalence suites (real frozen MLX SegNet/PoseNet via the cograd
   real-path; `--fused-r-kernel` semantics) — they pin **tolerance, not exact equality**, and
   measure the real-scorer deviation (above). These ARE the numerical measurement; the drift
   mechanism (per-chunk reduction re-order + GPU matmul tiling) is **per-chunk, pool-size(P)-
   independent** — the accum loop runs 600/B such chunks, each structurally identical to the tested
   chunk, so the per-chunk non-bit-identity is faithful at n600. This is a numerical-reduction
   property, not a score/d_seg claim, so there is no toy-vs-n600 evidence gap.
2. MEASURED the sealed-config structural fail-close directly (the `ValueError` above) with the
   sealed flag-set + `--fused-r-kernel`.

**Containment honored:** no heavy/multi-day run; the fail-close demo is setup-only (raises before
any training; gt_n6, <5 GB) via the governor's sanctioned reviewed-exception envs
(`TAC_LAUNCH_GUARD_OK=1` + `TAC_ADMISSION_BYPASS_OK=<rationale>`); scratch dir deleted after. Live
run-dirs untouched; sibling #409 (kill_pgrp / `tools/memory_guard.py`) surface untouched.

## Is the drift a bug or expected? — EXPECTED reduction-order, not a bug

The deviation is the **expected** non-associativity of fp reduction (serial accumulate-then-divide
vs one batched `mx.mean`) plus GPU matmul-kernel tiling on the batched pose path. The per-pair math
is op-for-op identical (verified bit-exact on CPU where reductions are batch-independent). It is a
correct-but-different reduction, i.e. a legitimate **trajectory-affecting opt-in**, NOT a wrong
gradient. So micro-batch is safe to use **as its own A/B arm** (it descends correctly), but it is
**not** a drop-in bit-identical accelerator for a config validated on the serial path.

## Consequence for the relaunch

- **Use micro-batch speedup for the baseline v7.5.2 pointer relaunch: NO.** Run the pointer relaunch
  on the sealed serial `--accum-pairs 8` path (micro-batch unset / `1`), exactly as sealed. This is
  forced twice over: (a) the sealed config won't even run with micro-batch (chroma-boundary), and
  (b) micro-batch is not bit-identical on the GPU training substrate.
- The micro-batch lever remains a **valid trajectory-affecting opt-in** for *non-baseline* arms that
  don't set the 4 unrouted levers (chroma-boundary / margin-saliency-reachability /
  seg-spike-reweight / seg-subpix), validated by its own end-to-end GPU trajectory A/B — never
  presented as a free speedup on a serial-validated pointer config.

## Follow-ups (not this task's surface)

- `test_batched_seed_cograd.py::test_equivalence_real_gt_real_scorer_seed_islands` tail-group seed
  tolerance (`2e-4`) is marginal — today's draw hit `5.85e-04` (denominator-cancellation class, well
  under the acknowledged GPU `1e-3`). The test owner should either widen the tail-group headroom to
  the documented reduction-noise class or switch the tail assertion to an absolute-deviation metric
  (the docstring already explains the small-||mean|| denominator effect). Flagged, not patched
  (sibling test surface; the failure reinforces this verdict).
- If the council ever wants micro-batch to accelerate a *future* pointer config, route
  chroma-boundary (+ the other 3 legs) into the batched twin first, then run a GPU trajectory A/B —
  that is a build, not a "free speedup."

## STORES CONSULTED

- CLAUDE.md §NO-FAKE (surrogate≠authority), §THE GOAL (score-correctness > speed), §Recursive
  adversarial review protocol axis-9 (measured-runnability + measured-scored-quantity), §Confound
  self-protection, §OPERATOR PRIORITY (n600-or-not-evidence, allergic-to-toys).
- `docs/operating_manual_craft_handoff.md` (re-derive from primaries; label MEASURED/DERIVED;
  point-fix ≠ class-fix; attack your own conclusion).
- MEMORY: L70/#348 (fused-r-kernel bit-identity localized to the render VJP scatter — does NOT cover
  the batched-scorer pose reduction), L53 (MPS never a score), verdict-scope ladder (this is a
  disposition, not a family-kill).
- Code: `experiments/train_levelset_witness_realized_through_R_mlx.py` (L1108-1120 logit-adjust
  compat, L4308-4314 + L5892-5897 chroma fail-close, L5370-5519 batched-twin wiring),
  `tac.boundary_math.levelset_micro_batch_loss` (L16-25 equivalence contract),
  `src/tac/tests/test_levelset_micro_batch_loss.py` (L20-25), `experiments/test_batched_seed_cograd.py`
  (L18-27 measured deltas), `experiments/results/__v752_drystart_final__/launch.sh` (sealed config).
- DAG residual line (FEED-fleet #173 tail): "Contrarian's micro-batch-twin bit-identity n600 smoke
  before the A/B rides a pointer run (config-correctness, orthogonal)" — this closes it.

## Triality

- **DAG:** `### FEED-410-microbatch-bitid` appended.
- **DSL:** N/A — no new/changed lever; `--micro-batch-pairs` already lives as a lever/flag. This unit
  is a config-compatibility + numerical-property **disposition**, not a lever change.
- **equations:** N/A-with-rationale — the batched≈serial fp-tolerance identity and the GPU ~1e-3
  reduction-noise class are already pinned by the two unit-test suites (not a score-term S_τ law);
  the chroma fail-close is a config guard, not an equation. No new `canonical_equations` row
  warranted (per the verdict-scope ladder: disposition, not truth).
