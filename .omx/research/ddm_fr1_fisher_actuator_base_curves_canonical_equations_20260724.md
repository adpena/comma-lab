# Canonical-equations note — DDM FR1 Fisher actuator base curves

`research_only=true` · `[macOS-CPU frozen-scorer advisory]` ·
`score_claim=false` · `pointer=0.1910828242 [contest-CPU] UNMOVED`

No new canonical equation is registered because no actuator realization or
base-curve delta was measured.

## Consumed laws

1. Fisher/margin rank:

   \[
   \operatorname{tr}F(m)=\tfrac12\operatorname{sech}^2(m/2),\qquad
   d_{\mathrm{flip}}=\frac{|m|}{\|w_c-w_{c'}\|}.
   \]

   Consumers:
   `frozen_scorer_fisher_curvature_margin_colocation_v1`,
   `fisher_curvature_equals_categorical_fisher_trace_caustic_v1`, and
   `segnet_head_rank4_linear_flipdist_v1`.

2. Corrected realization:

   \[
   d^*=D^\top\nabla_x m|_{\mathrm{fp}},\qquad
   \alpha^*=\frac{\mu}{\|P_R\nabla m\|}(1+O(\kappa\mu)),\qquad
   \min_{\delta x}\|\delta x\|^2\;\text{s.t.}\;J\delta x\ge\mu+\epsilon .
   \]

   Consumer: `flip_margin_step_law_v1`. The #583 artifact instantiates only the
   first-order ordering term; its own status record says the realized secant
   and receiver-closed QP legs are absent.

3. Contest-weighted local objective:

   \[
   S=100d_{\rm seg}+\sqrt{10d_{\rm pose}}
     +25B/37{,}545{,}489 .
   \]

   For each base \(b\), the owed comparison is:

   \[
   \Delta S_b =
   100(d_{\rm seg,b}'-d_{\rm seg,b})
   +\sqrt{10d_{\rm pose,b}'}-\sqrt{10d_{\rm pose,b}}
   +25(B_b'-B_b)/37{,}545{,}489 .
   \]

   Every term remains `NOT_MEASURED`; substituting zero would be false evidence.

4. Reverse-waterfill admission:

   \[
   -\Delta S/\Delta B \ge 25/37{,}545{,}489
   \]

   at the marginal stop. Exact prefix-byte custody is absent, so the KKT
   admission edge remains closed.

## Euclidean control and basis

The ranker is Fisher/margin. Euclidean geometry is a labeled control only and
was not run. No Fourier residual, carrier, or proxy was used. Any future
residual remains curvelet/shearlet-only under the operator directive.

## Scope

This note records an equation-input custody gap, not an equation falsification.
`flip_margin_step_law_v1` remains open; its executable receiver-closed
instantiation is owed for this rank-1 candidate and each current base state.
