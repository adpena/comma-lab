# JAX lane

Use this lane for research-time acceleration, not as a default shipped runtime dependency.

Best uses here:

- surrogate evaluator prototypes
- batched differentiable searches
- vectorized loss scans over preprocess/postfilter knobs
- analysis scripts that benefit from `jit` and `vmap`

Avoid promoting JAX into the shipped inflator unless the evidence is overwhelming.
