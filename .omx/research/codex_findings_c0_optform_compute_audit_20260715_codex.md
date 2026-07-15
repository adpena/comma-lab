# C0 optimal-form compute/convergence audit — 2026-07-15

`research_only=true`; `$0`; no GPU launch; no score claim; pointer unchanged.  The live process and
run directory were not stopped or mutated.  The only live-run artifact read was
`experiments/results/levelset_n600_witness_20260715T095030Z/launch.sh` (SHA-256
`5329619a794f276efc41a070fed4231adfaf95a10cde93099f81d249836daffb`).

## Verdict

**RELAUNCH-RECOMMENDED: Y.** C0 is not optimal-form on the pose-blind compute axis.  Its launch has
neither `--pose-training-compute-gate` nor `--verdict-pose-gate`.  Before pose finish engages, the
effective training weight is exactly `w_pose=0`, so PoseNet contributes **zero training gradient**,
but the incumbent loss still constructs the frame-0/PoseNet graph.  The verdict path also runs live
PoseNet despite being unable to use pose for a pose-blind progress decision.  The CPU-torch n96
receipt measures the PoseNet forward as **13.421/59.615 s = 22.6% of verdict wall**; the analogous
training-step reduction is structurally real but **UNMEASURED** on C0.  This is enough to establish
wasted compute without inventing a C0 whole-run speedup.

The corrected delta is the typed, default-off `PoseBlindComputeGate` DSL lever.  It adds exactly:

```text
--pose-training-compute-gate
--verdict-pose-gate
```

It never substitutes a banked pose scalar.  The pre-loop baseline remains a full live verdict;
pose-blind in-loop rows carry live `d_seg`, `d_pose=null`, `implied_S=null`, and are explicitly
score/selection-ineligible; live PoseNet resumes at the actual pose-finish event/backstop.  The live
C0 has `--pose-finish-engage-on sigma_min_plateau` with epoch 726 as the fail-safe backstop, so the
precise blind interval is **until the conditioning event fires or epoch 726**, not unconditionally
“until Muon at 726” (`launch.sh:155-156,207`).

## Per-axis audit

| Axis | C0 state | Exact live surface | Re-derived consequence |
|---|---|---|---|
| 1. Pose forward/backward while pose-blind | **OFF/MISSING — MATERIAL** | `--w-pose 1.0` is present, but pose finish is armed at backstop 726 and engaged by `sigma_min_plateau`; both new compute-gate flags are absent (`launch.sh:26-27,143-156,207`). | The trainer resolves the effective weight to zero until the event/backstop (`train_levelset...:7636-7654,11105-11150`), so pose contributes **no training gradient**. With the gate default OFF, the incumbent loss still renders frame 0 and forwards PoseNet; the verdict helper's `compute_pose` default is also `True` (`train_witness...:1225-1337`; `train_levelset...:246-315`). `w_pose=1.0` is only the finish-phase value. |
| 2. CPU-torch one-thread standard | **ON** | No CLI/env override is needed or present. The trainer imports canonical `SELECTED_THREADS` and calls `torch.set_num_threads(1)` before loading the frozen scorers (`train_levelset...:4169-4190`); the law pins `SELECTED_THREADS=1` (`segnet_exact_forward_cpu_thread_law_20260713.py:32-39`). | Resolved intra-op threads = **1** unless the fail-soft import/set call itself errors. This applies to training teacher/verdict CPU-torch forwards, not the separate auth evaluator. The n600 static-process receipts measure 2.9562855x/2.9970427x versus six threads; this is training/advisory compute evidence, not a score claim. |
| 3. Fewer epochs / convergence | **FreSh MISSING, HELD; current budget ON/RIGHT-SIZED** | C0 has `--structured-init`, event curriculum, Muon event start, tail controller, Polyak finisher, and `--epochs 3000` (`launch.sh:66,77-79,89-100,149,162-187,213-218`). It has no `--fresh-init`. | FreSh is built/default-off, but its governed fixed-quality run was refused before execution; epochs reduction and wall reduction are **UNMEASURED** (`init_levers_fresh_metainit_20260712.md:1-16,79-92`). It also changes the matched initialization basis (`self-orient` ON and seed-island legs OFF), so it is not an additive speed flag for this C0 treatment (`curriculum_dsl.py:4700-4747,4793-4885`). No evidence authorizes a smaller production cap. Keep 3000 as the cap; event/tail stop may terminate earlier on measured convergence. |
| 4a. Fused R | **ON** | `--fused-r-kernel` (`launch.sh:80`). | Trainer runs a per-chip parity gate before activation (`train_levelset...:10558-10569`). |
| 4b. Grouped backward / persistence pool | **ON** | `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 TAC_MLX_CUSTOM_PERSISTENCE_POOL=1` (`launch.sh:10`). | Both fast paths are armed. No new grouped-backward flag is missing. |
| 4c. GT constant cache / async verdict / chunking | **ON** | `--cache-gt-skeleton`, `--async-verdict`, `--verdict-batch 32`, all pairs via `--verdict-pairs 0` (`launch.sh:13-16,92`). | Compatible surviving throughput apparatus is active. |
| 4d. Safe compile | **ON-ARMED / CERTIFICATE-CONDITIONAL** | `--safe-compile-regions hosc_activation` (`launch.sh:102`). | This is the only certified region allowed by the #356 result. The trainer intersects the request with the per-host manifest and otherwise fails closed to eager (`train_levelset...:10526-10557`). The isolated-worktree dry-run found its local manifest absent; a real governed launch will refuse until MAIN's host certificate is present/fresh (`launch_witness_run.py:1846-1867`). That is an evidence gate, not an unreasoned omission. |
| 4e. Micro-batch B>1 | **OFF WITH REASON** | `--micro-batch-pairs 1` (`launch.sh:145`). | C0 carries `--seg-subpix-boundary-weight 0.3` (`launch.sh:220-224`), whose batched consumer is still absent and fails closed for B>1 (`train_levelset...:6992-7000`). Independently, the current full-step evidence is n24/non-ABBA: B2 was only 1.036x epoch / 1.001x step; B4/B8 regressed (`throughput_fresh_eyes_20260713.md:111-117`). B1 is the correct pointer-treatment choice, not a missing material opt. |
| 4f. Whole-step megakernel #356 | **OFF/EXCLUDED WITH MEASURED REASON** | No legal flag exists; safe compile is scoped to `hosc_activation`. | Eager-vs-compiled gradients differ by 2.3e-7 to 2.3e-5; CPU is 0.79-0.83x, GPU closure 1.12-1.21x, about 5% end-to-end (`whole_step_megakernel_356_20260711.md:30-61`). Verdict scope is the MLX-fp32 whole-step formulation, not all explicit-order kernels. Shipping a flag would violate the identity wall. |
| 5. Muon warm-start / anneal | **ON** | `--muon-warm-start-momentum`, `--muon-lr-final-frac 0.1`, event `powerlaw_meat`, backstop 726, transition rewarmup 14/cosine/floor 0.1, and reset moments (`launch.sh:19-25,93-96,141,155`). | The outgoing AdamW first moment seeds Muon and the Muon LR cosine-decays to 10% (`curriculum_dsl.py:4395-4412`; `train_witness...:2375-2458`). `--stage-transition-reset-moments` does not cancel the dedicated AdamW-to-Muon warm seed. |

## Pose-gate implementation and authority boundary

- `make_loss_fn(..., compute_pose=True)` preserves every existing caller.  Only the explicit false
  branch omits frame 0 and PoseNet and returns a zero pose term, so the already-zero weighted pose
  contribution remains numerically zero (`train_witness...:1225-1337`).
- The level-set wrapper derives `compute_pose=False` only when the typed gate is ON and the **effective**
  per-epoch pose weight is exactly zero (`train_levelset...:5391-5400,6139-6155`).
- Verdict helpers already have a real `compute_pose=False` branch that does not call PoseNet
  (`train_levelset...:246-315,319-352`).  The routing now uses it on CPU/GPU/nucleus/annulus paths
  (`train_levelset...:7753-7940`).
- A missing pose can never reach implied score, checkpoint score selection, or a causal score row;
  the emitted progress row is explicitly labelled and controller history omits `d_pose/implied_S`
  (`train_levelset...:7761-7777,8546-8569`).
- Any configured numeric non-live pose is still refused.  This landing supersedes only the old
  **banked-scalar-substitution** negative; it does not claim a payload-bound pose cache now exists
  (`src/tac/witness_control/pose_verdict_gate.py`).

## Exact DSL comparison and dry-run

Freshly generated corrected `launch.sh` versus the live C0 `launch.sh`, parsed as trainer flags:

```json
{
  "live_flags": 224,
  "relaunch_flags": 226,
  "only_live": [],
  "only_relaunch": ["--pose-training-compute-gate", "--verdict-pose-gate"],
  "value_mismatches": {"--out-dir": ["live C0", "dry-run output"]}
}
```

Dry-run command:

```text
.venv/bin/python tools/launch_witness_run.py --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 --config v9_cgauge_ideal_mod19 --dsl-lever PoseBlindComputeGate --out-dir experiments/results/levelset_n600_witness_c0_optform_relaunch_dryrun_20260715 --purpose "C0 optimal-form compute-only relaunch; no launch from audit" --dry-run
```

**DRY-RUN: PASS (exit 0).** It composed 18 levers, retained config-sealed 3000 epochs, validated
226/226 trainer flags, passed the typed DSL and schedule-provenance gates, and wrote launch SHA-256
`5df1bd2877471f9ea9d9a29ba8ec28e273cb6fbad8cbf3964bbf24628e18fb99`.  No process spawned.  The
isolated worktree has no per-host safe-compile manifest, so the dry-run emitted the designed advisory;
without `--dry-run`, the launcher refuses until that host-local certificate is fresh.

Corrected governed launch command for MAIN review (do not run from this audit branch; do not reuse or
mutate the live C0 directory):

```text
.venv/bin/python tools/launch_witness_run.py --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 --config v9_cgauge_ideal_mod19 --dsl-lever PoseBlindComputeGate --out-dir experiments/results/levelset_n600_witness_c0_optform_relaunch_20260715 --purpose "C0 optimal-form compute-only relaunch: pose-blind PoseNet forwards removed; MEANS only"
```

## Verification

- `ruff check --select F` on every changed Python file: **PASS**.
- `python3 -m py_compile` on every changed runtime/DSL file: **PASS**.
- `pytest -q src/tac/witness_control/tests/test_pose_verdict_gate.py src/tac/tests/test_pose_verdict_gate_trainer_wirein.py`: **12 passed**.
- Canonical launcher dry-run: **PASS, 226/226 flags, NOT spawning**.
- No MLX/Metal training smoke was launched.  The managed worktree has no Metal device, so no runtime
  speed number is fabricated.

## Stores consulted

`docs/operating_manual_craft_handoff.md`; `CLAUDE.md`; `AGENTS.md`; live C0 `launch.sh` only;
`canonical_task_status.jsonl`; lane/deferral ledgers; graph-memory recall; task #356 megakernel memo;
micro-batch #410/#447 receipts; FreSh #448 build/blocker and steps-dimension memos; one-thread canonical
law and static-process receipts; Muon DSL/runtime; pose verdict wall-clock receipt; latest C0 confound
hunt and campaign DAG feeds.

## Triality and pointer delta

- **DSL:** `PoseBlindComputeGate` is typed, default OFF, zero-argument composable, and expands to the
  two exact compute-gate flags above.
- **DAG:** `.omx/research/c0_optform_compute_audit_DAG_FEED_20260715.md`.
- **Equations:** `# NO_EQUATION_NEEDED` — this removes a zero-weight scorer graph and an ineligible
  observer forward; it adds no energy term or control law.  The empirical wall anchor remains
  `.omx/research/frozen_scorer_verdict_wallclock_n96_20260714.json`.
- **Sensitivity/Pareto/bit allocator:** non-binding; this landing changes wall-clock MEANS only and no
  witness bytes, score, archive, or pointer.
- **Pointer delta:** **NONE**.
