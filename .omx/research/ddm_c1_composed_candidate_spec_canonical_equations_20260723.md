---
title: Canonical equations for DDM C1 composed candidate
date_utc: 2026-07-23
lane_id: lane_ddm_c1_composed_candidate_spec_603_613_20260723
research_only: true
execution_allowed: false
---

# Canonical equations

Let `N=117,964,800`, control errors `E0=3,240,528`, target distortion
`tau=0.00116`, final archive bytes `B`, and official-YUV6 Pose distortion `P`.

The integer Seg feasibility gate is

`E* = floor(N tau) = 136,839`

and therefore

`Delta E_required = E0 - E* = 3,103,689`.

The measured role ceiling is

`E_LM = E_Lane + E_Movable = 300,563 + 425,853 = 726,416`,

so the unavoidable residual after perfect Lane+Movable repair is

`E_residual = E0 - E_LM - E* = 2,377,273`.

For exact sequential receiver states `A0, A1, ..., Ak`, component credit is defined only by

`credit_i = errors(A_(i-1)) - errors(A_i)`.

Thus the telescope

`sum_i credit_i = errors(A0) - errors(Ak)`

prevents double counting. Deltas measured from different controls are not additive evidence.

Within a frame, SegNet squeeze-excite gives a global channel gate

`g_c = sigmoid(phi_c(mean_(h,w) x))`,

so a local correction can change `g_c` and rescale the full frame. Consequently

`effect(delta_1 + delta_2) != effect(delta_1) + effect(delta_2)`

in general. Same-frame component effects require joint replay; eval-BN, YUV6, R, and the
rank-4 head being linear does not linearize the upstream squeeze-excite response.

For the live arms,

`E_v17 + E_v18b + E_j3 >= 3,103,689`,

`0 <= E_v17 <= 726,416`,

and

`E_j3 >= 3,103,689 - E_v17 - E_v18b`.

The composed-box predicate is

`B <= 200,000`

`errors(A_final) <= 136,839`

`Pose6_or_xi_stream_present(A_final) = true`

`P <= 0.00161`

with every quantity measured on the same exact archive and receiver runtime.

The contest action is

`S(A) = 100 d_seg(A) + sqrt(10 d_pose(A)) + 25 B(A)/37,545,489`.

The byte dual is

`lambda_B = partial S / partial B = 25/37,545,489`

`= 6.658589531221713e-7 score units/byte`.

Let `a` name correction application stage:
`a in {high_resolution_FP_pre_uint8, post_quantization_int8_lattice}`. For a measured
receiver-closed stream-and-stage curve `D_(i,a)(b)`, its interior waterfill condition is

`-partial D_i/partial b = lambda_B`.

At boundaries, allocate while `-Delta D_(i,a)/Delta b > lambda_B`; stop or change streams when
it does not. If two mechanisms act on the same debt pool, or two application stages encode the
same correction, replace their curves with the jointly measured lower convex envelope before
solving the KKT condition.

For each pair `p`, let its G3 atlas row define exact current debt `e_p` and let the waterfill
assign integer allowance `t_p` and byte budget `b_p`. The pair thresholds obey

`t_p = T(G3_p, lambda_B, {D_(i,a)}, B_remaining)`,

`sum_p t_p = 136,839`,

so no pair inherits a global constant. The recursive transition is

`A_(p,k+1) = Replay_R(Repair(Route_G4(Diff_R(Solve(A_(p,k))))))`.

It stops only when `errors_p(A_(p,k)) <= t_p` (`threshold-met`), the exact admitted family has
no feasible receiver-closed correction (`infeasible-certified`), or `b_p`/the global remaining
budget is exhausted (`budget-exhausted`). Every transition appends one convergence-ledger row.

For shared component `s` first owned by pair `p0`,

`incremental_bytes(s,p) = bytes(s)` if `p=p0`, otherwise `0`,

while every later pair references the same content hash. Pair-local repair bytes remain charged
to their owner. This is a rate-ownership identity only; exact Seg/Pose effects still require
joint replay because of squeeze-excite.

The 200,000-byte accounting equality in the target reservation is

`133,941 + 270 + 25,789 + 16,384 + 16,384 + 7,232 = 200,000`.

Only the first two terms are measured exact spends. The remaining terms are preregistered
reservations whose exact effects are computable-not-yet-computed; they do not imply a feasible
KKT solution.

The admissible computational-status set is

`{EXACT_COMPUTED, COMPUTABLE_NOT_YET_COMPUTED, INFEASIBILITY_CERTIFIED}`.

There is no epistemic “unknown efficiency” state in the frozen evaluator space. A pending term
must identify the exact adjoint, lattice solve, projector, and real coder used to compute it.

The pointer is not a variable in this research-only system:

`pointer = 0.1910828242 [contest-CPU]`, `pointer_moved = false`.
