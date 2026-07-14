# HELD equation specs — warm-start ORGAN / n=1 / RL (2026-07-14)

`research_only=true`. These are serial wire-in specifications, not registered equations. Exclusive
ownership remains with `provenance_canonicalize_fix_all_fakes` for
`canonical_equations/**`, `witness_dsl/**`, preflight, v9 autoconfig, and trainer consumers.

## 1. `costate_hierarchical_physics_residual_v1`

For interval `i`,

```text
y_i = (x_{i+1} - x_i) / delta_epoch_i
z_i = [1, x_i, Phi^T u_i]
C_0(q) = fixed Q_iso or P_aniso physics-response coefficient prior
P(alpha,g) = alpha * mean(diag(Z^T Z)) * diag(g)
beta_hat = (Z^T Z + P)^-1 (Z^T Y + P C_0)
Sigma_beta,c = sigma_c^2 (Z^T Z + P)^-1
Var[y_*c | z_*] = z_* Sigma_beta,c z_*^T + sigma_c^2
```

Typed payload:

- `prior_modes = [Q_priormean_iso, P_priormean_aniso]`
- `precision_grid = [0.01, 0.1, 1.0, 10.0]`
- `block_multipliers = [0.25, 8.0, 1.0, 16.0]` (SPECIFIED, not optimized)
- `min_inner_selection_folds = 4` (POST-HOC repair; must be labeled adaptive)
- `selection = prefix_only; tie = Q_then_lowest_precision`
- `reference_backend = numpy-fp32`; `parity_backend = mlx-fp32`
- receipt path: `U_hierarchical_physics_residual`

Held factory: `costate_hierarchical_physics_residual_v1_spec()`  
Consumer: organ tournament/dispatcher after telemetry extraction.  
Graduation: three independent trajectories and both aggregate + pc gates.

## 2. `costate_score_aggregate_projection_v1`

Given an organ vector `y`, diagonal predictive covariance `V`, analytic score weights `w`, and a
pre-existing aggregate forecast `a`, solve

```text
min_delta  0.5 delta^T V^-1 delta
subject to w^T (y + delta) = a

delta = V w (a - w^T y) / (w^T V w)
```

The aggregate supplier is #436 T/persistence. This equation cannot claim its aggregate score as a U
win. Held consumer: U per-class decomposition only.

## 3. `costate_requential_disagreement_curriculum_v1`

For each real past-prefix interval `i>=2`, fit the student on `[:i]`, then define the analytic score
disagreement and its shared-Gaussian KL proxy:

```text
delta_i = w^T (y_teacher,i - y_student,i)
v_i = w^T V_student,i w
k_i = delta_i^2 / (2 v_i ln 2)

r_i_raw = 0.5 + 0.5 n k_i / sum_j k_j
r = capped_simplex(r_raw, max=2, sum=n)
beta_R = posterior_solve(sqrt(r) Z, sqrt(r) Y, C_0)
```

`0.5` protected mass and cap `2` are SPECIFIED. `v_i < eps_fp32` is floored and counted. The law must
emit `NOT_PREFIX_FREE_REC`, `NOT_CAPACITY_FLOOR`, `NOT_PAC_BAYES` unless normalized proposal/teacher
distributions and a decodable prefix-free message are separately supplied.

Held factory: `costate_requential_disagreement_curriculum_v1_spec()`  
LawRef: `costate_requential_disagreement_curriculum_v1`  
Receipt path: `R_requential_disagreement_replay`.

## 4. `costate_support_identifiability_certificate_v1`

```text
O = rows(Phi^T u_i)
rank = rank(O - mean(O))
condition = sigma_max / sigma_min_nonzero
```

Required false-authority fields:

- `FORE = BLOCKED_DISTRIBUTION_CUSTODY` without behavior/target ratios;
- `TOFU = BLOCKED_PARTIAL_ACTION_CUSTODY` without random masks/offered slates;
- `HCM causal = false` without within-unit randomized treatments;
- `RL actor = disabled` without propensities/executed decisions.

Rank/support is predictive only and never implies causal identification.

## 5. `costate_mechanism_society_hse_v1`

For actor `m` and WF fold `i`, primary bounded behavior is

```text
b_mi = exp(-error_mi / error_persistence,i)
d(m,n) = 1 - cosine(b_m, b_n)
H(h) = -sum_c p_c(h) log2 p_c(h)       # single-linkage clusters at threshold h
HSE = integral_0^1 H(h) dh
normalized_HSE = HSE / log2(number_of_actors)
```

The dispatcher is excluded from the actor set. Reciprocal and fold-minmax behavior maps are mandatory
sensitivity rows. Diversity-only coresets cannot be promoted as score-optimal.

Held diagnostic factory: `costate_mechanism_society_diagnostic_v1_spec()`  
LawRef: `costate_mechanism_society_hse_v1`.

## 6. `costate_router_ulp_robustness_v1`

For every fold, generate the six one-ULP fp32 surface forms:

```text
recent +/- 1 ULP
median +/- 1 ULP
(recent-1 ULP, median+1 ULP)
(recent+1 ULP, median-1 ULP)

rho_i = mean_j 1[tool(variant_ij) == tool(original_i)]
rho = mean_i rho_i
```

This is local numeric representation robustness only, not state-estimation uncertainty. Current
receipt: `rho=36/42=0.8571428571`; two exact-zero-margin folds have `rho_i=0.5`.

Held next formulation: uncertainty/deadband or hysteretic tie policy, compiled as an A/B against the
current exact tie rule. No direct hot-file edit by this lane.

## Required provenance edges

```text
trajectory_log_sha256 -> interval_rows -> prefix_selector -> U posterior
U posterior -> requential KL proxy -> replay weights -> R posterior
all fixed-arm WF rows -> HSE actor society
router fp32 gate rows -> ULP variants -> robustness receipt
existing #436 aggregate -> score projection -> U decomposition
```

All edges terminate in `.omx/research/warmstart_organ_n1_rl_backtest_20260714.json`, schema
`costate_warmstart_cluster_backtest.v1`.
