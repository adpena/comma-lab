# Apple Local Compute Readiness

Schema: `apple_local_compute_readiness.v1`
Authority: `false_authority_local_dev_velocity_only`
Recommended backend: `mlx_metal`

| backend | available | notes |
|---|---:|---|
| mlx | True | MLX/Metal local research-signal backend; not contest authority |
| torch_mps | True | Torch MPS is local research-signal only; scorer drift must be measured |
| numpy_accelerate | True | Apple Accelerate can help CPU NumPy kernels but is not a scorer authority |
| hf_accelerate | False | Only needed for source-faithful HiNeRV OSS Accelerate launches |

## Blockers

- `macos_local_acceleration_false_authority`
- `contest_cpu_cuda_exact_eval_required_for_promotion`
- `python_accelerate_package_missing_for_oss_hinerv_launch`
