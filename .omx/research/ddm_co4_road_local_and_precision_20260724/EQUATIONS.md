# DDM CO4 Road-local and pair-precision equations

Authority: `_dev`, local advisory, `research_only=true`,
`execution_allowed=false`, `actuation=NONE`, `score_claim=false`.

## Road-local ranker

For pair \(p\), the realized pair target remains the receiver-closed
distortion closure used by CO3:

\[
y_p=\max\left(0,-\Delta D_p^{\mathrm{realized}}\right).
\]

The target-derived EV1 dominant-closure stratum is used only to evaluate the
held-out Road slice. It is not a feature and is not an inference-time router.
The expert router is the target-free G3 dominant pre-outcome class-flip
stratum.

The Road bucket information block is

\[
G_p^{\mathrm{Road}}
=
\sum_{b:\,\mathrm{Road}\in b}
\frac{n_{bp}}{n_b}G_b ,
\]

where \(G_b\) is the measured MS4D rank-4 margin-Fisher Gram,
\(n_b\) is the bucket event count, and \(n_{bp}\) is the exact PF2
pair-support count. Log trace, log largest eigenvalue, effective rank,
boundary trace share, support, G4 temporal fractions, and SN1 boundary/cell
fractions form the frozen Road feature extension.

The two preregistered ridge candidates are selected by held-out Road NDCG@4,
then global NDCG@4, Road Spearman rho, and lower complexity. A CO4 candidate
may replace CO3 only when

\[
\mathrm{NDCG@4}_{\mathrm{Road}}\geq 0.60
\quad\land\quad
\mathrm{NDCG@4}_{\mathrm{global}}\geq 0.75.
\]

The measured winner of this frozen race did not meet the first inequality, so
the selected prediction vector is the exact sealed CO3 OOF vector.

## Propagated pair precision

For a pair without a positive direct MS4D block, the propagated information
block is

\[
G_p^{\mathrm{prop}}
=
\sum_{b}
\frac{n_{bp}}{n_b}G_b .
\]

For each contribution trace

\[
t_{bp}=\operatorname{tr}
\left(\frac{n_{bp}}{n_b}G_b\right),
\]

the observed heterogeneity penalty is

\[
\mathrm{DEFF}_p=1+\mathrm{CV}(\{t_{bp}\}_b)^2.
\]

With the selected OOF residual scale \(\hat{\sigma}\), the nominal and emitted
propagated standard errors are

\[
\mathrm{SE}^{\mathrm{nom}}_p
=
\frac{\hat{\sigma}}{\sqrt{\operatorname{tr}(G_p^{\mathrm{prop}})}},
\qquad
\mathrm{SE}^{\mathrm{emit}}_p
=
\sqrt{\mathrm{DEFF}_p}\,\mathrm{SE}^{\mathrm{nom}}_p.
\]

Direct positive pair-indexed MS4D blocks override propagation. A pair with
neither direct nor positive propagated information is `UNRANKED`. Adjacent
pair order is `TIED` when the 95% intervals overlap or either interval is
missing.

The propagation is explicitly assumption-bearing: within-bucket
exchangeability, additive independent bucket blocks, and no unmeasured
cross-bucket covariance. It is not promoted to direct measurement.

