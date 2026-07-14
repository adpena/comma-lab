# Deterministic-reduction source → #348/L70 MLX-GPU bit-identity wall (deep-dive, operator 2026-07-14)

**Provenance:** operator shared aleksagordic.com/blog/collective-operations ("how can we leverage"). Honest
verdict on THAT article: cluster-scale distributed-collective infra (all-reduce/all-gather/reduce-scatter,
ring/tree, NVLink/InfiniBand/NCCL) — NOT applicable to our n=1 single-device witness; ZERO determinism
content. Banked as a reference for the #297 EXO/JACCL multi-machine path IF activated (see ledger pointer).
The article did surface the RIGHT adjacent question — deterministic reductions — pursued here as the
operator directed ("pursue that other source").

## The wall (L70 / #348, MEASURED)
MLX-GPU is NON-bit-identical to the numpy-fp32/CPU authority; the non-identity LOCALIZES to ONE op class:
**dup-index atomic scatter-add** in the R-up backward (scatter/gather-VJP). Atomic FP add is
order-nondeterministic (threads arrive in arbitrary order; FP add is non-associative). This is why MLX-GPU
is ADVISORY-only, never a score authority. `--fused-r-kernel` already got 0/28 cross-proc + ~8% faster.

## The source (deep-read, not abstract) — the 3-level determinism ladder
Primary: NVIDIA CCCL floating-point-determinism blog (CUB 3.1) + arXiv 2408.05148 (ORNL/UT-Battelle,
"Impacts of FP non-associativity on reproducibility for HPC and DL") + Collange/Iakymchuk "Reproducible
floating-point atomic addition in data-parallel environment" (RFA/superaccumulator lineage) + PyTorch
deterministic-algorithms docs.

| Level | Mechanism | Guarantee | Cost | OUR mapping |
|---|---|---|---|---|
| **not_guaranteed** | unordered atomicAdd, single kernel | none | fastest | **= our atomic-scatter wall (L70)** |
| **run_to_run** | FIXED hierarchical reduction TREE (per-thread→warp-shuffle→block-shmem→2nd-kernel aggregate; NO atomics; predetermined order) | same-GPU, same-config bit-identical | ~baseline (can beat 2-phase atomic at our sizes) | **= our `--fused-r-kernel` (0/28 cross-proc, +8%) — literature CONFIRMS the mechanism we built** |
| **gpu_to_gpu** | **RFA (Reproducible FP Accumulator)**: bin inputs into fixed exponent ranges (default 3 bins) → summation order architecture-INDEPENDENT | cross-GPU AND (properly configured) cross-CPU bit-identical | 20-30% slower on large N | **THE MISSING RUNG → MLX-GPU as a bit-exact AUTHORITY** |

**int64 fixed-point superaccumulator** (Kulisch/Collange; NVIDIA blog omits it, ReproBLAS/annals-csis
confirm): accumulate in WIDE INTEGERS (associative + exact → order/device-independent) then convert. This
is #348's OWN second instinct ("fixed-point int64"). A simpler, exact alternative to RFA — likely the
better fit for a Metal kernel (no exponent-bin bookkeeping, exact by construction).

## Transfer to #348 (DERIVED)
1. Our atomic-scatter wall = textbook `not_guaranteed`. CONFIRMED.
2. `--fused-r-kernel` = the `run_to_run` fixed-tree fix (same-GPU bit-identity). CONFIRMED + named.
3. **The unlock (new):** to promote MLX-GPU from ADVISORY → BIT-EXACT AUTHORITY (match numpy-fp32/CPU
   exactly, not just self-consistent), implement a **`gpu_to_gpu`-class scatter-add**: either (a) RFA
   exponent-binning, or (b) **int64 fixed-point superaccumulator** (recommended — exact by construction,
   #348's instinct), as a Metal kernel at the remaining scatter/gather-VJP sites (the "re-poison, probe
   first" sites L70 flagged). Match the numpy-fp32 CPU reduction ORDER bit-for-bit.
4. Why it matters (MEANS, not a direct pointer-mover — honest): a bit-exact MLX-GPU makes the ~104× MLX
   scorer speedup usable for AUTHORITATIVE verdicts (today CPU-only), and strengthens the
   deterministic-reproducibility spine (MLX-GPU could join CPU/CUDA as an authority axis). Compute-facet
   win (#252 standing program), verdict-speed win. NOT a score row by itself.

## OSS harvest (patterns, not links)
- **CUB `cub::DeviceReduce::Sum(in,out,N, env=determinism::gpu_to_gpu)`** (CCCL 3.1) — reference RFA
  impl; the 3-exponent-bin split + fixed accumulation ALGORITHM ports to a Metal kernel (CUDA API itself
  is unusable on MLX/Metal — take the algorithm, not the code).
- **ReproBLAS / Collange-Iakymchuk RFA** (annals-csis Vol.5 pdf; arXiv reproducible-atomic-addition) —
  original RFA + superaccumulator reference; the int64/Kulisch-accumulator variant is the exact one.
- **PyTorch deterministic `scatter_add`/`index_add`** (sort-then-segment-reduce, `use_deterministic_
  algorithms(True)`) — the CPU/torch reference reduction ORDER our numpy-fp32 authority uses; MATCH it so
  MLX-GPU == CPU bit-for-bit (the actual authority target, not just cross-GPU).

## Next step (queued, $0-design → Metal build; NOT built now — consolidation in progress, no-rush)
#348 follow-on: build an int64-fixed-point (or RFA) scatter-add Metal kernel at the remaining VJP scatter
sites → verify MLX-GPU bit-identical to numpy-fp32 CPU on n600 → promote MLX-GPU verdict from advisory to
authority (parity ≥ bit-exact, not just ≥0.9997). Sister: #348 (completed run_to_run), #252 (MLX/Metal
standing program), #212/#356 (kernel suite/megakernel), the deterministic-reproducibility non-negotiable,
L70/L53 (MPS-never-authority — MLX-GPU-as-authority is the legitimate opposite when bit-exact).
