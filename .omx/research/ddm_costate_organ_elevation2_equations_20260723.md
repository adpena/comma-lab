# DDM Costate Organ Elevation 2 — canonical equations

Date: 2026-07-23  
Code authority: `src/tac/ddm_costate_law.py`  
Canonical wrapper: `src/tac/canonical_equations/ddm_joint_recursion_costate_20260723.py`

## D2 costate

Equation ID: `ddm_joint_recursion_costate_d2_v1`

\[
|\lambda_b^{D2}| =
g_b\;v_b\;r_b\;p_b\;\tau_b
\]

where:

- \(g_b\) is the exact current score-debt gap on the block/pair/site;
- \(v_b\in[0,1]\) is measured visibility through usable evaluator support;
- \(r_b\in[0,1]\) is measured receiver/uint8 realizability;
- \(p_b=1/\max(B_b,1)\) is the measured or custody-safe derived byte price;
- \(\tau_b\in[0,1]\) is D2, a family-local dual tolerance or realized-validity factor.

Direction is stored on the proposed update; the ranker consumes the magnitude. Missing
receiver realization sets \(r_b=0\), not one.

For g3×v19 pair rows, v19 leaves the candidate's global archive delta unallocated. Therefore the
pair distortion target is:

\[
\Delta S_{\mathrm{pair}} =
100(d_{\mathrm{seg,after}}-d_{\mathrm{seg,before}})
+\sqrt{10d_{\mathrm{pose,after}}}
-\sqrt{10d_{\mathrm{pose,before}}}.
\]

The shared rate delta is excluded rather than charged once per pair. g3's existing pair allocation
prices the baseline pair lambda; the new candidate's shared-byte amortization remains owed.

Pair D2 is:

\[
\tau_p=\min(1,\max(0,-\Delta S_{\mathrm{pair}})/g_p).
\]

v19b block D2 is the measured sequential joint-survival fraction clipped to \([0,1]\). Neither is a
universal Hessian radius. If a future producer emits a J-of-J radius, that producer-specific radius
replaces the SLA proxy for its block only.

## Primitive chain

Each dv1/g4 primitive records:

\[
\text{measured bytes}\rightarrow\text{described fraction/cell reach}
\rightarrow\text{receiver-realized }\Delta S.
\]

The final term is currently null for these cell-space primitives. Their cell-space marginal is an
upper bound, while admitted \(\lambda^{D2}=0\). J_paint is the named measurement that closes the
chain.

## Scheduler

Equation ID: `ddm_topological_gauss_southwell_validity_v1`

\[
G_b=|\lambda_b^{D2}|\,\rho_b.
\]

Only blocks whose dependencies are complete enter the current frontier. Within it:

1. freeing blocks precede spending blocks;
2. coarse levels precede fine levels;
3. independent same-scale blocks descend by \(G_b\).

This is a deterministic lexicographic policy, not a scalar blend that can bypass topology.

## Rate floor

\[
p_{\mathrm{rate}} = 25/37{,}545{,}489
= 6.658589531221714\times 10^{-7}
\quad\text{score units/byte}.
\]

No candidate is admitted from a cell-space proxy alone; reverse waterfill stops when measured
realized marginal value falls below this rate floor.

## Epistemic scope

All current numeric anchors are `[macOS-CPU frozen-scorer advisory]`, score claim false,
promotion ineligible. Contest-CPU/CUDA equivalence is not inferred.

