# owed-15 — Class-A trunk-lever isolation LADDER — RUNBOOK — 2026-07-09

**SYNTHESIS_v3_v752 §C item 15.** `[macOS-MLX advisory]` NON-PROMOTABLE. **CONFIGS ONLY — do NOT train until the
operator's which-to-run GO.** NON-blocking (launch-1 still ships if net-positive). Regenerate the configs with:

```bash
.venv/bin/python tools/build_v752_isolation_arms.py \
    --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600
# -> experiments/results/v752_isolation_arms/{arm1_basis_only,arm2_plus_taper,arm3_plus_aa_ipe}/launch.sh
#    + manifest.json  (all argparse-validated, never-invent-a-flag; NOT executed)
```

## Why (the load-bearing evidence gap)
Launch-1 composes THREE Class-A levers at ep0; two ride load-bearing in the trunk on sub-n600 evidence:
- **#121 d_seg-aware taper** — ESTIMATED ~0.03 RANK-1 from ONE under-converged run; "converged flips to −8%" is
  INFERRED, not measured.
- **AA render-IPE** — oracle-R lane-recall +0.38 @384; through-training ΔS ASSUMED.
- **self-orient directional basis** — MEASURED −48% (n600-class) = the STRONG anchor; needs no isolation.

## R8 LAW (CODE-GROUNDED — why FRESH training, never inference-toggle)
NEVER inference-toggle a trained-WITH render lever. #121 taper is a trained-WITH curvelet-amplitude reshape;
`--render-aa ipe` reshapes the render path (persisted as `__cfg_render_aa`). The weights ADAPT to the tapered/AA'd
render, so toggling either at inference on ONE composed checkpoint MISMATCHES the render = an INVALID/toy
isolation (the exact toy this owed exists to prevent). Each arm is therefore a FRESH incremental-TRAINING run to
the SAME floor, scored at n600 through-R d_seg.

## The three arms (run order; each FRESH to the same floor)
| arm | self-orient | #121 taper | --render-aa | isolates |
|---|---|---|---|---|
| **arm1_basis_only** | ON | OFF | none | the incremental baseline (directional basis alone) |
| **arm2_plus_taper** | ON | ON | none | the #121 taper delta (vs arm1) |
| **arm3_plus_aa_ipe** | ON | ON | ipe | the AA render-IPE delta (vs arm2) = the full launch-1 trunk |

## ROLLBACK sign-test (the verdict rule)
Keep a lever IFF its isolated n600 through-R d_seg IMPROVES over the arm below it; else ROLLBACK (drop the lever
from the trunk). Never ship a silently-hurting trunk lever on a single-run / oracle estimate. verdict_scope:
INSTANCE (taper) / FORMULATION (AA-ipe through-training ΔS).

## To run (AFTER operator GO — each is a multi-hour n600 training arm, governed)
Do NOT bypass the launcher. When approved, each arm rides the FULL `tools/launch_witness_run.py` gate chain
(memory preflight + admission + throughput + governed durable spawn). The emitted `launch.sh` under each arm dir
is the exact command; author each as a governed launch (e.g. via `--config crucible_v7 --extra-trainer-flags`
for arm3's pure-add taper, or the P7 typed `crucible_v752` variant for arms 1/2 which set `render_aa=none`).
Byte-close all arms through `tools/levelset_byte_close_and_eval.py` and read per-class d_seg through-R.

## Join point (P7)
When P7 lands the typed `crucible_v752` configs, register these three as typed `WitnessProgram` variants
(basis-only / +taper / +AA-ipe) so arms 1/2's `render_aa=none` is set at compile (not a launch.sh value rewrite)
and each carries a DSL-provenance manifest.
