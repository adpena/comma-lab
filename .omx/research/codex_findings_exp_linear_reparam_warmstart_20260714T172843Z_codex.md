# Codex findings — exponential-linear warm-start from the divergence fork

UTC: 2026-07-14T17:28:43Z  
Contract: `PAPER_WARM_START_FROM_DIVERGENCE`  
Paper: Ethan Smith, [Learning in Curved Weight Space: Exponential-Linear Weight Reparameterization for Improved Optimization](https://arxiv.org/abs/2607.09967)  
Lane: `lane_exp_linear_reparam_warmstart_20260714`  
Authority: `[macOS-CPU advisory] NON-PROMOTABLE`  
`research_only=true`

## TIER-0 outcome

The deterministic actual-witness 2×2 harness is built, statically verified, resumable, and starts all arms from an exactly identical effective checkpoint. The required four MLX optimizer arms did **not** execute: this headless process cannot load an MLX Metal device even when MLX is asked for CPU. Therefore:

- SEL-over-Muon: **NO VERDICT — EXECUTION CUSTODY BLOCKER**.
- Fewer steps: **NO VERDICT — EXECUTION CUSTODY BLOCKER**.
- Terminal quantized-rate/distribution effect: **NO VERDICT — EXECUTION CUSTODY BLOCKER**.
- Pointer delta: **NONE**.
- No family-negative token is issued.

The blocker is durable in `exp_linear_reparam_warmstart_execution_blocker_20260714.json`. The common-start receipt is durable in `exp_linear_reparam_warmstart_common_start_20260714.json`.

Commit custody is separately blocked. The required serializer was invoked once with the seven owned non-contested paths and every post-edit SHA-256, but `git add` failed before staging because this sandbox cannot create temporary Git index objects (`Operation not permitted`). No serializer bypass was attempted; the shared lane, checkpoint, and review-state files were excluded. Receipt: `exp_linear_reparam_warmstart_commit_blocker_20260714.json`.

## The assumption-divergence fork

| Surface | Paper | This witness vehicle | Consequence for the smoke |
|---|---|---|---|
| Model/data | large transformer, OpenWebText, generalization | tiny 96×4 coordinate INR, one clip, overfit | warm-start the real ep650 basin; do not import a fresh-transformer claim |
| Optimizer | Adam-family additive raw updates | AdamW or Muon+AdamW, with Muon already orthogonalizing matrix updates | the decisive contrast is SEL-over-Muon, not SEL-over-AdamW alone |
| Objective | next-token cross entropy | frozen-scorer witness objective through uint8/resize R | every d_seg row must be recomputed through actual R and frozen CPU SegNet |
| Runtime | torch | MLX training; numpy-fp32/CPU verdict | train in MLX, adjudicate exact folded weights through the portable reference |
| Payload | weights serve prediction | weights are archive bytes | require matched-d_seg step count **and** exact int8+Brotli bytes/distribution |

The paper's reported 1.32–1.49× matched-validation-loss step reduction is a prior, not evidence on this fork.

## Paper-faithful method and the warm-start correction

The tested formulation is `mismatch_no_offset_anchored_state_preserving_fixed_scale_lr_v1`:

- symmetric-exponential and identity-like linear pathways are both present;
- sign-aware mismatch forward, structured row/column multipliers, linear weights and biases only;
- no exponential-only arm, because the paper reports that construction failed;
- fixed scale learning rate is explicitly the paper's fixed-multiplier-ablation formulation, not a claim to reproduce its annealed main default;
- `beta=2.5` is a width-96 formulation choice and stays inside the verdict scope.

Direct numerical inversion of the mismatch map was not sufficient for this incumbent. Its maximum effective-weight error was only `2.384185791015625e-07`, but that moved three razor-tie SegNet argmax pixels. The accepted conversion is instead

`W_eff = W_checkpoint + f(u, s) - f(u0, s0)`.

This preserves the SEL Jacobian while making activation exactly state preserving. Effective weights, int8+Brotli bytes, and through-R CPU d_seg are identical for all four arms at step 0.

## Bregman / #500 / #504 grounding

Let `W=f(u,s)` be the effective weight map. Euclidean descent in raw coordinates gives, to first order,

`ΔW = -η J_f J_fᵀ ∇_W L`.

Thus SEL constructs a diagonal/structured pullback preconditioner. Where integrable, the corresponding local primal metric is `(J_f J_fᵀ)⁻¹`, expressible as the Hessian of a Bregman potential. This is the constructive mirror-chart connection to the #500/#504 optimal-metric thread.

The limits matter: raw dual-coordinate Euclidean descent is the squared-Hessian geometry identified in the #504 correction, and SEL is **not automatically** the Fisher-natural or reachable-VJP metric `argmax_native_vjp_fidelity_v1`. Fisher cotangent transport still requires the inverse/solve, and no measured Hessian/Gram custody exists here. Muon and SEL therefore act on plausibly distinct structures—Muon on matrix-gradient singular geometry, SEL on elementwise/row-column chart curvature—but “distinct” is not “additive.” Only the SEL-over-Muon contrast can settle that.

## MEASURED common start

Checkpoint: actual `levelset_witness_ema_BEST.npz`, epoch 650, SHA-256 `6dd28a6e295d007ef0e53ae3e0e792a517a5708394a17d2185870e44920dedca`.

CPU/numpy verdict uses exact int8 dequantization, numpy-fp32 level-set forward, torch-authority camera uint8 R, frozen CPU SegNet, and deterministic padding to canonical batch32 geometry.

| Metric | MEASURED value |
|---|---:|
| mean d_seg, pairs 0–3 | `0.0032145182291666665` |
| pair 0 | `0.0035654703776041665` |
| pair 1 | `0.00323486328125` |
| pair 2 | `0.003021240234375` |
| pair 3 | `0.0030364990234375` |
| exact state identity | `true` |
| int8+Brotli bytes | `62,087` |
| parameter count after pair slice | `72,951` |
| weight std | `0.21997597` |
| abs p99 / p999 / max | `0.6148828 / 1.209574 / 3.608866` |
| Pearson kurtosis | `11.0403` |
| fraction `abs(w)>4σ` | `0.0021247` |

## Required 2×2

Preregistered target rule: for each optimizer family, the non-SEL control must first improve strictly from the common start. Then measure the first treatment step at or below the terminal d_seg of that control; SEL lands only if it reaches that target in strictly fewer steps and finishes equal-or-better. Rate is compared between the first control and treatment rows that reach the matched basin, with terminal byte delta retained as secondary telemetry.

| Arm | MLX steps | terminal d_seg | matched-target steps | blob bytes | Status |
|---|---:|---:|---:|---:|---|
| AdamW | — | — | — | — | `NOT_RUN_EXECUTION_CUSTODY_BLOCKER` |
| Muon+AdamW | — | — | — | — | `NOT_RUN_EXECUTION_CUSTODY_BLOCKER` |
| SEL+AdamW | — | — | — | — | `NOT_RUN_EXECUTION_CUSTODY_BLOCKER` |
| SEL+Muon+AdamW | — | — | — | — | `NOT_RUN_EXECUTION_CUSTODY_BLOCKER` |

There is no honest additive-or-redundant verdict until these rows exist.

## Rate-side verdict

At activation, SEL is exactly rate neutral because the effective weights are bit-identical. The incumbent small-n slice is 62,087 int8+Brotli bytes and has a heavy-tailed distribution (Pearson kurtosis 11.0403). Whether SEL makes terminal weights more quantizable or more entropy-expensive remains **NO VERDICT** until the four traces produce exact byte deltas and distribution changes at matched d_seg.

## Held DSL lever specification

Provisional typed lever name: `ExpLinearWarmStart`.

| Field | Typed contract |
|---|---|
| `enabled` | boolean, default `false` |
| `activation_boundary` | optimizer stage boundary; never mid-step |
| `parameter_scope` | enum `linear_weights_and_biases` |
| `forward_mode` | enum `mismatch_no_offset` for this formulation |
| `state_preserving_anchor` | required `true` for checkpoint warm-start |
| `beta` | positive finite float, provenance required |
| `scale_structure` | enum `row_col_mult` |
| `scale_lr_schedule` | tagged union `fixed` or `annealed`; this smoke is `fixed` |
| `optimizer_cross` | enum `adamw` or `muon_adamw` |
| `fold_on_export` | required `true`; decoder sees only effective weights |
| `receiver_byte_delta` | structurally zero before quantization; exact blob delta still measured |
| `admission_receipts` | common-start identity, four traces, matched-step contrasts, exact byte/distribution deltas |

Born-through-DSL rule: no CLI flag or trainer branch may precede the typed node, compile validation, provenance record, equation registration, and resume schema. The source legs under `src/tac/witness_dsl/` and `src/tac/canonical_equations/` are **HELD** for the V9 provenance owner / shared-worktree drain. This advisory arm touched neither tree.

Admission rule after drain: wire the lever only if the Muon control strictly improves from step 0, `SEL+Muon` reaches Muon's terminal-control d_seg in fewer steps, ends equal-or-better, has no matched-basin rate regression that outweighs saved training time under the declared objective, and preserves exact fold/parse-back behavior. Otherwise retain the trace as a scoped negative and queue the closest paper-faithful reformulation rather than killing the family.

## Reformulation queue if the fixed-scale treatment is negative

1. Paper-faithful annealed scale learning rate on the same anchored warm-start.
2. Fresh mismatch initialization, explicitly separated from warm-start identity.
3. Small preregistered `beta` grid around the width-96 heuristic.
4. Offset/low-rank path only after the simpler family is adjudicated.

Each is a new formulation token; none may retroactively broaden this smoke's verdict.

## Reproducible execution handoff

Run in an ordinary macOS Terminal process with Metal custody:

```bash
cd /Users/adpena/Projects/pact
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python tools/probe_exp_linear_reparam_warmstart_mlx.py --steps 24 --eval-every 2 --pairs 0,1,2,3
```

The command is `$0`, foreground, deterministic seed 0, storage-preflighted, and crash-resumable from each evaluation boundary. It does not dispatch a provider or launch a heavy n600 run.

## Triality and stores consulted

- DSL: specification above; code leg **HELD**.
- Equations: pullback/Bregman equation above; canonical-equation code leg **HELD**.
- DAG: `sub015_DAG_exp_linear_reparam_warmstart_20260714.md`.
- Stores consulted: `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `tac.subagent_contract` entry for `PAPER_WARM_START_FROM_DIVERGENCE`, latest Codex findings/session summaries, latest design/council memos, `reports/latest.md`, lane registry, subagent progress, master gradient anchors, metric canon, paper source, actual checkpoint/config/GT/feature-state artifacts, and both live inboxes through `2026-07-14T17:00:15Z`.

## Verdict scope

No performance verdict was reached. The only negative is an execution-environment classification: `NO_VERDICT_EXECUTION_CUSTODY_BLOCKER`. It does not apply to SEL, Muon, the SEL+Muon composition, a future scale schedule, full n600, contest CPU/CUDA, or V9.
