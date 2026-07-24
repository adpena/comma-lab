# Canonical equations — DDM MS6b receiver support

All measured quantities below are scoped to the V19C endpoint, a signed
one-receiver-quantum perturbation, and the
`[macOS-CPU frozen-scorer advisory]` axis.

For receiver actuator \(a\), sign
\(\sigma\in\{-1,+1\}\), and support metric \(m\), the measured support is

\[
s_m(a,\sigma) =
\operatorname{measure}_m\!\left(R(\theta+\sigma q_a e_a),R(\theta)\right).
\]

Here \(q_a\) is the receiver's exact quantization unit and \(R\) is the actual
decode/raster/uint8/frozen-scorer path. An infeasible receiver quantum is an
explicit state, not \(s_m=0\).

For two measured directions, sign asymmetry is the derived quantity

\[
A_m(a)=
\begin{cases}
\dfrac{s_m(a,+1)-s_m(a,-1)}
      {s_m(a,+1)+s_m(a,-1)},&
s_m(a,+1)+s_m(a,-1)>0,\\[6pt]
0,&\text{otherwise.}
\end{cases}
\]

For PF2 bucket \(b\), let \(M_b\) be its exact PF2 pair membership and let
\(J_b\) be the union of exact pair IDs reached by measured receiver assignments:

\[
J_b=\bigcup_{(a,\sigma)\in Q_b}J_{b,a,\sigma}.
\]

The bucket's probe-event incidence is

\[
I_b=\sum_{(a,\sigma)\in Q_b} n_{b,a,\sigma}.
\]

\(I_b\) is an incidence sum, not unique-event cardinality; the same raw event
may occur in more than one signed probe.

For preregistered G3 hard-pair set \(H_{24}\), complete coverage is the
conjunction

\[
C_{24} =
\bigwedge_{p\in H_{24}}
\bigwedge_{b:p\in M_b}
\left[p\in J_b\right].
\]

Measured result: \(C_{24}=\mathrm{false}\), with 106 false pair/bucket clauses;
only pair 21 has all of its required clauses true. Therefore the MS4 transition
predicate is false.

