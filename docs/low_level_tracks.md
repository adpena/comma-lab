# low-level and acceleration tracks

## Rust

Primary compiled lane.

Target uses:
- sparse residual decode
- ROI patch application
- raw frame writer
- SIMD-fixed filters
- a future CPU inflator core

## CUDA

Selective lane.

Target uses:
- local residual optimization
- optional inflator plugin
- micro-kernels that are clearly worth the complexity

## JAX

Research-time acceleration lane.

Target uses:
- surrogate evaluator prototypes
- batched differentiable search
- analysis code that benefits from `jit` and `vmap`

Do not make JAX a shipped runtime dependency unless the evidence is overwhelming.

## Mojo

Experimental lane only.

Use Mojo if it wins a narrow benchmark cleanly.
Do not make it a required dependency for the mainline repo unless it proves itself.

## Assembly

Last resort.
Only write assembly after profiling proves one tiny kernel dominates end-to-end runtime.
