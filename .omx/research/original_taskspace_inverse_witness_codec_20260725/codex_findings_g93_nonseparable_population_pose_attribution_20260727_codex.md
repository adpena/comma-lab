# G93 — nonseparable population pose attribution

Date: 2026-07-27  
Lane: `lane_g93_nonseparable_pose_attribution_20260727`  
Verdict scope: canonical pair-priority and XRay attribution surfaces  
Authority: exact evaluator mathematics; no candidate, eval, or score claim

## Finding

The canonical per-pair XRay primitive and the pair-component diagnostic placed
the pose square root inside the pair average:

```text
mean_i sqrt(10*d_pose_i)
```

The frozen evaluator instead computes:

```text
sqrt(10*mean_i d_pose_i)
```

These differ whenever pair pose errors are heterogeneous. Since the square root
is concave, the former is systematically smaller and changes pair ordering
relative to the true population operating point. The wrong rows fed
cathedral-autopilot, sensitivity-map, bit-allocation, hard-pair, and
multi-granularity consumers, so this was a real signal-suppression and
misallocation path rather than a reporting typo.

## Landed law

`tac.score_geometry.population_score_attribution` is now the canonical
torch-free law. For population pose distortion

```text
D = mean_i d_i
P = sqrt(10*D)
```

it assigns

```text
a_i = P*d_i/D,  D > 0
a_i = 0,        D = 0
```

and therefore

```text
mean_i a_i = P.
```

This is the proportional Euler-complete attribution. Equivalently,

```text
a_i = 2*N*(dS/dd_i)*d_i
dS/dd_i = 5/(N*sqrt(10*D)).
```

The factor two is required because the square-root term is homogeneous of
degree one half. The attribution is for prioritization; an individual pair is
not independently score-additive.

## Wire-in

- `src/tac/score_geometry.py`: one canonical exact population law.
- `src/tac/xray/per_pair_score_decomposition.py`: exact global recomposition,
  score-preserving per-pair attribution, and exact pair-MSE VJP scale.
- `src/tac/multi_granularity_sensitivity.py`: pair ranking uses the same law.
- `tools/xray_pair_component_errors.py`: future XRay rows carry explicit
  `GLOBAL_SQRT_PROPORTIONAL_EULER_COMPLETE_V1` custody and never use a
  pair-local square root.
- `tools/xray_hardpair_hitlist.py`: scorer-native rows are accepted; legacy
  rows are reattributed only when complete raw per-pair Seg/Pose distortions
  survive, otherwise they fail closed.

The exact nonlinear whole-object decision remains G83. G93 changes proposal
ordering and costate intelligence; it does not authorize local acceptance.

## Adversarial witness

For two pairs with pose distortions `[0,4]`:

```text
authoritative = sqrt(10*mean([0,4])) = sqrt(20)
old proxy      = mean([sqrt(0),sqrt(40)]) = sqrt(10)
```

The scorer-native pair attributions are `[0, 2*sqrt(20)]`; their mean is exactly
`sqrt(20)`. Regression tests seal this counterexample, heterogeneous Seg/Pose
recomposition, zero-pose singularity custody, invalid inputs, and legacy
artifact refusal.

## Verification

```text
94 passed
ruff check: passed
ruff format --check: passed
mypy --follow-imports=skip: passed
py_compile: passed
```

## Triality

DSL:

```text
PopulationAttribution(seg[0:N], pose[0:N])
  -> exact_global_components
  + costate_scale
  + score_preserving_pair_attribution[0:N]
```

DAG:

```text
frozen scorer pair distortions
  -> global mean operating point
  -> nonseparable square-root attribution
  -> XRay / sensitivity / bit-allocation ordering
  -> finite receiver-realized proposals
  -> exact archive states
  -> G83 nonlinear whole-state argmin
```

Equations:

```text
S_dist = 100*mean(seg_i) + sqrt(10*mean(pose_i))
mean_i [100*seg_i + sqrt(10*D_pose)*pose_i/D_pose] = S_dist
```

## Pointer honesty

No archive was built or evaluated in G93. Candidate claim: false. Score claim:
false. Frontier moved: false. The dynamic target remains whatever is fresher
and lower between our exact promoted pointer and the upstream authoritative
leaderboard; no literal target is compiled into this law.
