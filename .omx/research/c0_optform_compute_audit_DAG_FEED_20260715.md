# DAG FEED-C0-optform-compute — 2026-07-15

`research_only=true`; `$0`; no GPU launch; no score claim; pointer unchanged.  Collision-safe FEED for
MAIN review/union.

## Graph

```text
C0 live launch v9_cgauge_ideal_mod19 (224 trainer flags)
  -> two-phase pose finish
       -> effective w_pose = 0 until sigma_min_plateau event OR ep726 backstop
       -> training pose gradient = 0 during blind phase
       -> incumbent still constructs frame0 + PoseNet graph
       -> incumbent verdict still runs PoseNet
       -> MEASURED n96 verdict pose share = 13.421 / 59.615 = 22.6%
  -> optimal-form audit
       -> one-thread CPU-torch = ON (canonical SELECTED_THREADS=1)
       -> structured init / event schedules / 3000-epoch cap = ON
       -> FreSh = HELD, epochs-to-target UNMEASURED
       -> fused R = ON
       -> grouped backward + persistence pool = ON
       -> cache GT + async verdict + chunk32/all-pairs = ON
       -> safe compile hosc_activation = ARMED, per-host certificate gated
       -> micro-batch B>1 = EXCLUDED for this treatment (subpix consumer absent + no admitted wall win)
       -> whole-step megakernel = EXCLUDED, measured fp-reorder formulation no-go
       -> Muon warm-start + LR anneal = ON
  -> exactly one material omission
       -> PoseBlindComputeGate
            -> --pose-training-compute-gate
            -> --verdict-pose-gate
            -> no banked pose scalar
            -> pre-loop and post-engagement PoseNet live
            -> blind progress d_pose=null, implied_S=null, selection_eligible=false
  -> typed launcher dry-run
       -> 226/226 flags valid
       -> only scientific argv additions are the two gate flags
       -> epochs remains 3000
       -> no process spawned
  -> RELAUNCH-RECOMMENDED (MAIN review + host safe-compile certificate required)
```

## Apparatus edges

- Producer: `src/tac/witness_dsl/curriculum_dsl.PoseBlindComputeGate`.
- Training consumer: `experiments/train_witness_realized_through_R_mlx.make_loss_fn(compute_pose)` via
  the level-set trainer's effective per-epoch pose weight.
- Verdict consumer: existing chunked CPU/GPU d_seg path with `compute_pose=False`.
- Authority guard: `src/tac/witness_control/pose_verdict_gate.py`; numeric non-live pose remains
  refused, and a pose-missing progress row cannot produce an implied score.
- Resume: the existing pose-finish event/backstop and persisted Muon/controller state re-derive the
  effective pose phase; the new flags add no mutable controller state.
- Launcher/autopilot: typed lever composition plus existing host-certificate, memory, system-admission,
  and governed-launch gates.  No direct trainer launch is authorized by this FEED.
- Sensitivity/Pareto/bit allocator: non-binding because the delta changes compute MEANS only.

## Scoped negatives and reopen triggers

- Old PoseVerdictGate negative remains valid only for **banked numeric d_pose substitution without
  payload/receiver custody**.  This FEED uses no substitution; it emits a missing, ineligible pose.
- FreSh is **NO-VERDICT / HELD**, not a convergence negative.  Reopen with the governed matched cold
  fixed-quality A/B and one-time sweep cost included.
- Micro-batch B>1 is excluded only for this C0 treatment/current formulation.  Reopen after every
  active loss consumer is routed and same-SHA uncontended B1/B2 current-V9 ABBA clears wall, memory,
  and descent parity.
- Whole-step compile negative is formulation-scoped to MLX-fp32 reorder-permitting fusion.  Explicit-
  order certified kernels remain open.
- Safe compile is host-certificate-scoped.  Recertify on the launch host; never transfer a manifest.

## Canonical evidence

- Live config: `experiments/results/levelset_n600_witness_20260715T095030Z/launch.sh`, SHA-256
  `5329619a794f276efc41a070fed4231adfaf95a10cde93099f81d249836daffb`.
- Pose wall: `.omx/research/frozen_scorer_verdict_wallclock_n96_20260714.json`,
  `[macOS-CPU-torch 1-thread advisory wall-clock]`, score claim false.
- Findings: `.omx/research/codex_findings_c0_optform_compute_audit_20260715_codex.md`.
- DSL equation disposition: `# NO_EQUATION_NEEDED` for compute-elision; no loss/control math changes.
