# Task 701 typed Fisher/G4 waterfill equations

Authority is local `[macOS-CPU frozen-scorer advisory]` evidence:
`research_only=true`, `score_claim=false`, and the contest-CPU pointer is
unchanged.

The requested receiver-realized objective for a candidate \(x\) is

\[
S(x)=100D_{\rm seg}(x)+\sqrt{10D_{\rm pose}(x)}
     +\frac{25}{37{,}545{,}489}B(x),
\]

where \(D_{\rm seg}\) is exact argmax error fraction after receiver, uint8, and
resize; \(D_{\rm pose}\) is official batch-32 Pose6 MSE; and \(B\) is the
complete real-coded receiver archive size. The box is

\[
E(x)\le 136{,}839,\qquad
D_{\rm seg}(x)=\frac{E(x)}{117{,}964{,}800}
\le 0.001159998575846354.
\]

The local exchange rates required at every realized operating point are

\[
\frac{\partial S}{\partial D_{\rm pose}}
=\frac{5}{\sqrt{10D_{\rm pose}}},
\qquad
\frac{\partial S}{\partial B}
=\frac{25}{37{,}545{,}489}
=6.658589531221714\times10^{-7}\ {\rm S/byte}.
\]

Let the sealed exact endpoint be \(E_0=17{,}927\), the boundary
\(E_5=136{,}839\), and integer headroom \(H=E_5-E_0=118{,}912\). The base
ladder is preregistered without claiming measurements:

\[
(E_0,\ E_0+1,\ E_0+\lfloor H/8\rfloor,\
E_0+\lfloor H/2\rfloor,\ E_0+\lfloor7H/8\rfloor,\ E_5)
\]

\[
=(17{,}927,\ 17{,}928,\ 32{,}791,\ 77{,}383,\ 121{,}975,\ 136{,}839).
\]

After the full base ladder is measured, adaptive refinement bisects the
integer-error interval with the largest absolute change in adjacent marginal
score-per-error slopes. This does not authorize skipping base rungs.

Execution requires the conjunction

\[
\mathcal A =
C_{\rm RG3}\land T_{\rm pose}\land H_{\rm same\ object}
\land A_{\rm typed}\land Q_{\rm dim}\land\Lambda_{\rm RD1}.
\]

The sealed custody gives \(C_{\rm RG3}=0\) with 25 missing blocks,
\(T_{\rm pose}=0\), \(H_{\rm same\ object}=0\),
\(A_{\rm typed}=0\), \(Q_{\rm dim}=0\), and
\(\Lambda_{\rm RD1}=0\). Therefore \(\mathcal A=0\), and the lawful measured
rung set is empty:

\[
\mathcal R_{\rm Task701}^{\rm measured}=\varnothing.
\]

For every RD1 cell \(c\), a finite price requires same-object, adjacent,
receiver-realized rung deltas:

\[
\lambda_c =
\frac{\Delta B_c}{-\Delta D_c}.
\]

No such Task 701 deltas exist, so all 162 values remain
\(\lambda_c={\rm NULL}\). Cross-object endpoint ratios and the R3 q4/q8
finite-family control are not substitutes.

The preregistered formulation falsifier,
\(B\gtrsim200{,}000\) for every receiver-realized boundary rung, cannot be
evaluated until \(\mathcal A=1\) and the full ladder is run. Hence this landing
is a precondition refusal, not a formulation negative.
