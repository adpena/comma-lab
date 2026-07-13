# DAG FEED — SPS gradient-role separation dig (2026-07-13)

## FEED-SPS-GRADIENT-ROLE-SEPARATION-20260713

```yaml
feed_id: FEED-SPS-GRADIENT-ROLE-SEPARATION-20260713
lane_id: lane_sps_gradient_separation_probe_20260713
research_only: true
status: NO_GO_FORMULATION_INSTANCE_PROBE
authority: macOS-CPU local Torch-NumPy parity probe; NON-PROMOTABLE
pointer_delta: zero
source_run_mutated: false
```

### Question

Does the V9 CGauge shared coordinate-INR trunk exhibit the SPS-style gradient-role conflict between
pair-local scorer prediction and temporal state-consistency strongly enough to justify a split
parameter stream?

### Inputs

- SPS paper/project/code: `arXiv:2607.01218`, `lil-lab.github.io/sps`, `github.com/lil-lab/sps`.
- Read-only epoch-275 deploy EMA:
  `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_mlx.npz`, SHA-256
  `1676e4d45e180c7a28ec2ecce2b932d0e5087a2cfec2636ff2efe1673dbbcbf0`.
- Exact cached scorer targets:
  `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (mmap, no mutation).
- Probe tool: `tools/probe_sps_gradient_role_conflict.py`.
- Receipt: `experiments/results/sps_gradient_separation_probe_20260713/receipt.json`.

### Evidence

```yaml
live_epoch_275:
  temporal_gradient_norm: 0.0
  cosine: undefined
  reason: screw starts 450; phase and pose start 726
counterfactual_fully_armed_n4:
  pairs: [75, 225, 375, 525]
  seg_vs_temporal:
    cosine: 0.10718932747840881
    negative_cosine_tensor_weight_fraction: 0.0
    negative_product_scalar_weight_fraction: 0.4541577825159915
  pose_vs_temporal:
    cosine: -0.0000069293791966629215
    negative_cosine_tensor_weight_fraction: 0.4157782515991471
  seg_plus_pose_vs_temporal:
    cosine: 0.09969637542963028
    negative_cosine_tensor_weight_fraction: 0.0
  heterogeneous_pair:
    pair: 225
    seg_vs_temporal_cosine: -0.2788417637348175
    negative_cosine_tensor_weight_fraction: 1.0
apparatus:
  torch_numpy_phi_cosine: 1.000000138144482
  argmax_equal: true
  pose_yuv6_forward_max_abs_delta: 0.000003814697265625
  mlx_direct: blocked_no_metal
```

### Decision edges

1. Aggregate pair-local-versus-temporal trunk gradients are aligned -> architectural SPS split is
   `NO_GO` at formulation-instance scope.
2. One of four pairs is anti-aligned -> preserve a stratified n600/engaged-checkpoint reformulation;
   do not kill the gradient-conflict family.
3. No token/KV/activation-state analogue exists -> any weight-space intervention is routed to the
   PCGrad/MGDA/CAGrad/Nash-MTL comparison family, with unitary scalarization as control.
4. L87 `d_cov+d_gauge` is output-debt taxonomy, not a parameter loss mask -> do not route by label.
5. Conditional steps break-even is `r > k_T f_T + k_W(1-f_T)`; `r>1.05` only under the unmeasured
   `f_T=.95,k_T=1,k_W=2` assumptions. Extra teacher VJPs invalidate the 1.05 claim.

### Downstream consumers

- `tac.witness_dsl.curriculum_candidate_pool`: candidate
  `sps_weight_space_gradient_role_separation`, status `reformulation-queue`.
- STEPS-dimension controller: may consume only after an engaged n600 conflict measurement and a
  measured component-cost split.
- V9 trainer: no change.
- Canonical equations: no registration; evidence bar not met.

### Reactivation gate

```text
engaged checkpoint AND n600 gradient distribution
AND (aggregate cosine <= -0.05 OR repeatable harmful negative-cosine stratum)
-> equal-step scalarization/stratified-batching control
-> PCGrad control with scorer-VJP cost measured
-> only then consider a byte-costed adapter split
```

### Triality

- DAG: this FEED.
- DSL: design-only candidate pool row; no trainer lever claimed built.
- equations: N/A-with-rationale; conflict law and 95/5 runtime premise do not close.

### Verdict scope and reformulation queue

`NO_GO` applies only to SPS-like architectural stream duplication for the epoch-275/n4 instance.
Temporal consistency, phase advection, and weight-space conflict methods remain live only behind the
reactivation gate above. n600 is explicitly owed for any family-level verdict.
