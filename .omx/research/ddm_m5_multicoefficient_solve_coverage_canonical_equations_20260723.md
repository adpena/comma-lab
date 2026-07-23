# DDM M5 multicoefficient solve-coverage equations

Date: 2026-07-23  
Lane: `lane_ddm_m5_multicoefficient_solve_coverage_20260723`  
Evidence: `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`.

## Per-stratum receiver transition law

Let `y_i` be the frozen target label at frame-1 scorer site `i`, `b_i` the
receiver-closed control argmax, `c_i(p)` the receiver-closed argmax for counted
integer-lattice program `p`, and `k` one class detected from the canonical
receiver class contract.

```text
E_k(b) = sum_i 1[y_i=k] 1[b_i != y_i]
H_k(p) = sum_i 1[y_i=k] 1[b_i != y_i] 1[c_i(p) = y_i]
C_k(p) = sum_i 1[y_i=k] 1[b_i = y_i] 1[c_i(p) != y_i]
E_k(c(p)) = E_k(b) - H_k(p) + C_k(p)
N_k(p) = H_k(p) - C_k(p)
```

`H_k` is helpful reach, `C_k` is harmful collateral, and `N_k` is only a net
effect. A positive `N_k` is not a zero-collateral solve.

## Byte-box reach and certification law

For a named, finite receiver program set `P` and exact archive byte box `B`,
define:

```text
P_B = {p in P : exact_archive_bytes(p) <= B}
P_B^0 = {p in P_B : sum_k C_k(p) = 0}
R_k(P,B) = max_{p in P_B^0} H_k(p)
I_k(P,B) = E_k(b) - R_k(P,B)
```

`I_k` is a certified-infeasible residual only if all of the following are
proved:

1. `P` has a finite, content-addressed completeness manifest;
2. every member of `P_B` is exhaustively enumerated, or a math-complete
   optimality certificate covers the omitted members;
3. every candidate is replayed through its real receiver, uint8, exact `R`, and
   frozen argmax;
4. class-isolated zero-collateral accounting is present.

A solver stall, greedy stop, or one measured `p` cannot establish `I_k`.

For the C1 residual bucket `K_res={Road,Undrivable,MyCar}` and integer target
allowance `T=136,839`, the #366 scope requested by M3 would be:

```text
X_366(P,B) = max(0, sum_{k in K_res} I_k(P,B) - T)
```

The M5 input has `B=200,000` B and exact receiver replay, but lacks the finite
reachable-set manifest, exhaustive certificate, and isolated per-stratum
solutions. Therefore `I_k` and `X_366` are undefined, not zero and not the
observed candidate residual.

## Measured v19b instance

The exact measured program is the v19b sequential joint stack:

```text
control archive = 133,941 B
candidate archive = 137,825 B
shared delta = 3,884 B
sum_k H_k = 232,540
sum_k C_k = 129,218
sum_k N_k = 103,322
```

All five `C_k` are nonzero. This instance proves nonzero Road and Lane reach,
but it is not in `P_B^0` and cannot certify a residual.

`verdict_scope=INSTANCE:V19B_GREEDY_INTEGER_LATTICE_STACK_AT_C1_200000_BYTE_BOX_N600`

