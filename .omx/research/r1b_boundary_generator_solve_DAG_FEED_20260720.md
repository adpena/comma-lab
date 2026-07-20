# DAG FEED — R1b boundary-generator joint solve

`feed_id=FEED-R1B-BOUNDARY-GENERATOR-20260720` · `lane_id=r1b_boundary_generator_solve_20260720T161946Z` · `research_only=true` · pointer unchanged

## Typed nodes

| node | evidence type | state |
|---|---|---|
| `A_C2` | exact existing production archive | MEASURED n600 control: `94,344 B`, SHA-bound |
| `R_C2` | receiver + scorer-input parse-back | MEASURED exact factor-2 equality over `600` pairs |
| `M_C2` | hard CPU-Torch through-R row | MEASURED `[macOS-CPU advisory]`: `d_seg=.003515794640406966`, `d_pose=127.36588287353516`, `S=36.10275630841103` |
| `P_bnd` | counted localized boundary packet | BUILT and tested; absent from `A_C2` |
| `J_mod` | batch-16 first-order + realized-secant Jacobian | OPEN; moderate-margin production custody owed |
| `Q_joint` | deterministic corrected active-set QP | BUILT and tested |
| `U_exact` | bounded uint8 exact resize preimage | BUILT, mandatory, exact-verified |
| `K_full` | full separable resize kernel / MDL selector | CONSUMED from `da64a5bc8e` as `9b50eb4aeb`; mandatory before hard oracle |
| `C_replay` | compact counted full-kernel replay grammar | OPEN; offline selector output is not yet receiver-bound |
| `H_cpu` | fresh hard-oracle candidate predicate | BUILT as a required callback inside admission; production candidate evaluation owed |
| `E_erm` | contained nonlinear fallback | BUILT: unknown-only, `4x16`, hard terminal, degenerate=no |
| `X_0` | pose `xi[0]` receiver and counted payload | OPEN; no sidecar/manifest in `A_C2` |
| `A_R1b` | strict counted production candidate archive | OPEN |
| `G_task` | task admission | FAIL on current control Seg gate; cannot evaluate R1b candidate while `A_R1b` is open |

## Authority-preserving edges

```text
A_C2 --exact decode--> R_C2 --hard CPU Torch through R--> M_C2
M_C2 --bytes pass; d_seg fails 10.371x--> G_task = false [control only]

generic localized frame + video-derived counted selections/coefficients
    --> P_bnd

moderate-margin winner-rival debt --first order + realized secant--> J_mod [OWED]
{P_bnd, J_mod, Fisher diagonal} --> Q_joint
Q_joint --> U_exact --> K_full --second exact projection check--> H_cpu

only {STALLED,CYCLE,BUDGET}_UNKNOWN
    --> E_erm --4 hard terminals; degenerate=no--> H_cpu

K_full --selected null coordinates--> C_replay [OWED; decoder <=1800 s]
{H_cpu accepted packet, C_replay, X_0, base payload, runtime, manifest, container}
    --compile and count every video-derived byte--> A_R1b [OWED]
A_R1b --full n600 receiver + hard scorer--> task and fixed-C1 gates [OWED]
```

No edge permits `P_bnd`, `K_full`, a target label plane, caller JSON, proxy loss, or a local frame-coder size to assert archive admission. PDW2 remains conditioning only. `K_full` chooses an exact camera preimage by a bounded offline frame-level MDL heuristic and never proves the final packet/container byte count or receiver runtime. Its measured constant-preference timing was `34.7690 s` for one frame; `600x=20,861.4 s` is only a derived diagnostic, but it is enough to forbid naive per-frame search inside the receiver.

## Measured acquisition update

The sibling R2b result changes the acquisition surface:

- independent cell repair realized `1,585/16,751 = 9.462%`; interaction is dominant;
- the `[1e-3,1)` margin band holds `16,319/17,926` gap flips and `0.01383 S`, versus `1,607` flips / `0.00136 S` in `<1e-3` ties;
- `93.4%` of flip cells have bounded-uint8 realizations, so selection/interaction precedes lattice expansion;
- `xi[0]` dominates the Pose residual by more than `1000x` over each other dimension;
- the current realization fraction implies a `<=1,852 B` carrier break-even target; a changed measured realization fraction requires recomputation.

## Solver-stack hooks

- **Sensitivity map:** consume batch-16 winner-rival first-order plus realized secants in the moderate-margin boundary band. Do not schedule cells independently.
- **Pareto constraint:** require `archive<=286,680 B` and `d_seg<=3.39e-4`; separately retain fixed-C1 `archive<=216,223 B`. All payload, `xi`, residual, runtime, and container terms are charged.
- **Bit allocator:** radius-one boundary coordinates first; widen only when measured marginal value is strictly above `25/37,545,489 = 6.6585895312e-7 S/B`. Use `1,852 B` only with the measured `9.462%` realization premise. Charge the compact kernel replay payload and its runtime.
- **Cathedral/autopilot:** no dispatch or promotion. Reactivation requires `A_R1b`, durable custody, and full n600 local advisory replay before any separately governed contest-axis replay.
- **Continual learning:** the machine receipt stores control metrics, exact section bytes, gate margins, live dependency facts, and the next acquisition coordinate.
- **Probe disambiguator:** compare at least the inherited mask candidate and requested full-kernel preferences through the dual-coder selector and hard oracle; no preferred kernel family is assumed. The final archive resolves whether frame-level MDL survives container accounting.

## Reactivation predicate

Reactivate only when production batch-16 moderate-margin Jacobian/secant custody exists and a counted `P_bnd + C_replay + xi[0]` payload compiles into a deterministic strict archive whose receiver finishes within `1,800 s`. Re-run full n600 through the same receiver/hard scorer and record archive hash/bytes, decoded hash, parse-back hash, runtime, axis, `d_seg`, `d_pose`, and action components. Do not reactivate for packet unit tests, a target-space improvement, a frame-coder proxy, or the existing C2 control row.
