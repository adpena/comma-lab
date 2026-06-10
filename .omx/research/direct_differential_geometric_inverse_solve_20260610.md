# The DIRECT differential-geometric inverse-evaluator solve (operator, 2026-06-10)

Operator critique (binding redirect): we have NOT exhausted inverse-evaluate.py / inverse-steganalysis
/ adversarial / generative / QAT / Quantizr-fully-optimized-NOT-in-RGB; the interpreter/compiler is
"kind of a FAKE implementation" (it SEARCHES candidates, it does not SOLVE for the witness from the
evaluator's structure); and with the FROZEN evaluate.py + ONE video there are far more DIRECT, ELEGANT
attacks via derivatives / integrals / manifolds / PDEs / Taylor — combining everything. This memo is
the honest reframe + the real lever.

## The honest "fake implementation" admission
V6 as built = a FrozenEvaluatorContract + a frontier-synthesis loop that EVALUATES candidate archives.
That is candidate SEARCH, not COMPILATION. A real compiler SOLVES for the minimum-description witness
directly from the oracle's local geometry. We built the measurement organs (atlas, cone, flip-map,
invisibility basis) but consumed them as inputs to byte-shaving / training — never assembled them into
the direct analytic/variational solve they're the ingredients of. That is the gap.

## The direct problem (variational, not regression)
S = 100·d_seg(x) + sqrt(10·d_pose(x)) + 25·rate(A), x = inflate(A). Solve:
  minimize_A  rate(A)
  s.t.        m_c(x) >= 0 at every scored pixel        (argmax preserved => d_seg = 0)
              ||PoseNet(x) - PoseNet(src)||^2_{first-6} <= tau   (pose tube)
KKT: L = rate(A) + sum_i lambda_i max(0,-m_i(x)) + mu (pose_err - tau).

## The geometry that makes it DIRECT (not RGB, not trained)
- Margin field m(x)=l_c - max_{j!=c} l_j; ∇_x m via the SegNet Jacobian; m/||∇m|| = signed distance to
  the decision boundary IN PIXEL SPACE = per-pixel appearance freedom, ANALYTICALLY.
- J_pose (restricted to the 6 scored dims) null space = appearance directions with ZERO pose cost.
- Feasible cell per pixel = (margin-positive half-space) ∩ (pose-tube) ∩ (resize/YUV null space).
  The optimal frame = the MINIMUM-DESCRIPTION point in that intersection (S12 generalized from the
  resize null space to the FULL scorer-constraint manifold).
- Taylor: pose smooth -> 2nd-order tube; seg-margin locally linear -> half-space; combine for the
  local quadratic program whose solution is the cheapest feasible perturbation.
- SCORER-NATIVE coordinates (the "not RGB"): represent the argmax PARTITION (SegNet's scored quantity)
  + the 6-dim pose TRAJECTORY (PoseNet's scored quantity) + minimal margin-holding appearance. Derived
  from the variational structure, NOT hand-designed (vs Quantizr's grayscale-LUT in RGB).

## One solve subsumes the listed techniques
adversarial training = descent through J_scorer; generative = sampling the feasible manifold; QAT =
quantization as a CONSTRAINT in the variational problem (not post-hoc); inverse-steganalysis/STC/UNIWARD
= optimal coding under the per-pixel margin cost (the contest IS inverse steganalysis). They are
instances of ONE direct solve, not separate lanes.

## Why this can beat the learned renderer (the falsifiable thesis)
A NeRV spends bytes to reproduce APPEARANCE (RGB) and only incidentally lands in the cell. The direct
solve spends bytes ONLY on what the constraints require (the partition + trajectory + boundary-holding
appearance) and explicitly puts all other error in the free null spaces. At equal exact d_seg/d_pose it
should encode CHEAPER; and it lowers distortion + rate TOGETHER (the R-D-coupling reprioritization),
which is the only class-shift that dominates the curve.

## Relationship to running levers
- Sharpens the offensive-research lever (a1cf37e8): the #1 lever is THIS direct solve (quotient/score-
  native, derived), ranked above rate plays per the R-D reprioritization.
- Sharpens F (floor): T_floor IS the value of this variational program's optimum — the minimum rate
  over the feasible manifold. F should derive it as the program's optimum, not a separate byte count.
- Makes V6 REAL: the compiler becomes the KKT solver over the scorer manifold, not a candidate search.

## The $0 prototype (falsifiable, pre-registered)
On N>=8 pairs, locally (MLX/CPU exact scorers, NO MPS): for each scored pixel compute the feasible cell
(margin half-space from the real SegNet Jacobian ∩ pose null/tube from the real PoseNet Jacobian ∩
resize/YUV null space), find the minimum-coding-cost feasible frame, byte-estimate it, and compare its
coding cost to the current decoder's output AT EQUAL exact d_seg/d_pose. PREDICTION: the direct-solve
frame encodes measurably cheaper (or reaches lower d_seg at equal bytes); KILL if it cannot beat the
learned renderer at equal distortion (then the renderer is already near the manifold optimum and the
class shift must come from elsewhere). Materialize byte-closed + exact-eval if the smoke passes.
