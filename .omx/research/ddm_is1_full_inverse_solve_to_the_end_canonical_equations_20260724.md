# DDM IS1 — canonical-equation note

`research_only=true` · `score_claim=false` ·
`evidence_axis=[macOS-CPU frozen-scorer advisory]` ·
`main_review_required=true`

These are proposed laws for MAIN review. This isolated lane does not mutate
the shared canonical-equations registry.

## `ddm_is1_strict_rate_budget_at_fixed_distortion_v1`

Status: `DERIVED`.

\[
B_{\max}(s_\star,d_s,d_p)
=
\left\lfloor
\frac{37{,}545{,}489}{25}
\left(s_\star-100d_s-\sqrt{10d_p}\right)
\right\rfloor .
\]

At `s_star=0.15`, `d_s=0.0001519690619574653`, and
`d_p=0.00010184327939026322`, the continuous bound is
`154522.5148231086`, hence `B <= 154522`. At 154,522 bytes the derived total
is `0.14999965720042385`; at 154,524 it is `0.15000098891833010`. This is a
budget, not evidence that such an archive exists.

## `ddm_is1_pipeline_confound_non_telescope_v1`

Status: `STRUCTURAL FAIL-CLOSED LAW`.

Let \(e_k\) be the argmax error count after stage \(k\). A causal stage factor
\(\rho_k=e_{k+1}/e_k\) may be reported only when both endpoints are generated
from the same SHA-bound payload with exactly one stage changed. Otherwise:

\[
\rho_k=\mathrm{NULL},
\qquad
\prod_k \rho_k\ \text{must not be equated to}\
\frac{e_{\rm described}}{e_{\rm solve}}.
\]

The controlled total endpoint ratio is

\[
\frac{2{,}845{,}843}{17{,}927}
=158.746192893401\ldots .
\]

V14 and PT1 locate formulation-specific losses within this path, but they are
not a common-payload telescope. The solve’s exact uint8/real-\(R\) replay
proves quantization is not the binding stage family. The described-base gap is
therefore a pipeline-confound observation, not a correction-workload law.

## `ddm_is1_fivetype_layer_rehoming_conservation_v2`

Status: `STRUCTURAL FAIL-CLOSED LAW`.

Let
\(\mathcal T=\{\text{SKELETON},\text{CONNECTION},\text{FIBER},
\text{GAUGE},\text{RESIDUAL}\}\) and
\(\mathcal L=\{L1_{\rm program},L2_{\rm chart},L3_{\rm raster},
L4_{\rm scorer\ feature},L5_{\rm verdict}\}\), exactly as sealed by
`ddm_min_description_contract`.

\[
B_{\rm counted}
=\sum_{t\in\mathcal T}\sum_{\ell\in\mathcal L}B_{t,\ell}.
\]

Each video-derived byte has exactly one home. A structural zero is permitted
only where a specific quotient coordinate proves that stream absent; it is
not a measured exchange rate for other coordinate systems. RESIDUAL is
irreducible only after #669c has tested all admissible re-homings. Current
#669c status: queued, never run.

## `ddm_is1_oracle_diff_true_price_firewall_v1`

Status: `STRUCTURAL + ONE MEASURED ROW`.

For a proposed generator \(g_{t,\ell}\), a true price exists only if:

\[
P_{t,\ell}
=
B_{\rm exact}\!\left(
\operatorname{encode}_{t,\ell}
\left[\Delta_{\rm solve}\right]
\right)
\]

has deterministic parse-back, reproduces the claimed object through the real
receiver/uint8/\(R\) path, and binds all input and implementation custody.
Otherwise \(P_{t,\ell}=\mathrm{NULL}\).

The IS1 pass establishes one row:

- `RESIDUAL × L3_raster`: measured exact reversible n600 byte upper bound
  before re-homing.

The `GAUGE × L3_raster` zero is structural in the selected scorer-plane
coordinate system, not an empirical exchange rate. SKELETON, CONNECTION, and
FIBER remain `NULL`. Every inherited RD1/DR2B/C1/MENU1/V19C “price” is
`upper-bound, proposal-search-channel`; it cannot kill a path or support box
arithmetic.

## `ddm_is1_solution_set_min_S_objective_v1`

Status: `STRUCTURAL`.

\[
\min_{\substack{x\in\mathcal X_{\rm legal},\;f\in\mathcal F,\;z\\
                 R\,D(f,z)=x}}
\left[
100d_{\rm seg}(x,x_{\rm source})
+\sqrt{10d_{\rm pose}(x,x_{\rm source})}
+\frac{25}{37{,}545{,}489}B_{\rm real}(f,z)
\right],
\]

Here \(x_{\rm source}\) is fixed evaluator custody; \(x\) selects which legal
scorer-equivalent/tolerance-region decoded solution to emit. The constraint
requires the counted description \((f,z)\) and deterministic receiver \(D\)
to realize that solution through \(R\). \(f\) selects the description family
and parameters, while \(z\) carries counted video-derived
latents/exceptions. `B_real` is exact post-coder, post-parse-back counted
bytes. Distortion and the tolerance knee are outputs. The #613 box and the
exact 17,927-error solve are diagnostics/constraints, not the optimization
objective. This strictly contains fixed-distortion min-description and
box-steering as restricted problems.

For the score-quotient functional family, \(D(f,z)\) emits only the scorer
plane/at-risk margins and the at-most-six-dimensional Pose statistic. Camera
RGB and human fidelity are excluded unless demanded by the legal receiver.
The counted payload is

\[
B_{\rm functional}
=B(\theta_f)+B(z_{\rm temporal})+B(z_{\rm interface})
+B(z_{\rm residual}),
\]

with real coder price inside fitting. Its current exact minimum is `NULL`.

## `ddm_is1_class_interface_placement_is_counted_v1`

Status: `MEASURED INSTANCE LAW / STRUCTURAL INTERPRETATION`.

For the RG3 current-vocabulary testbed:

\[
\operatorname{closed}_{\rm class\ birth}=0/10,\quad
\operatorname{closed}_{\rm finer\ boundary}=3/9,\quad
\operatorname{closed}_{\rm Fisher\ cell}=8/17.
\]

An interface’s geometry may be receiver-generic, but its placement for this
video is information:

\[
B_{\rm interface\ placement}>0
\quad\text{unless another counted type/layer stream derives it exactly.}
\]

Solution-description reads the target placement from the solved object; it
does not ask which problem-space coordinate can cause it. The 25 remaining
rows are an empirical demand set. Their bytes remain `NULL` until each solved
value is assigned a type/layer home and coded.

The closure ordering in this finite testbed,
`Fisher cell > finer boundary > class birth`, supports refinement of existing
structure before conjuring absent interfaces. Its verdict scope is
`INSTANCE_EXTENDED_GRAMMAR_RG3`, not a family theorem.

## `ddm_is1_training_residual_exhaustion_gate_v2`

Status: `STRUCTURAL FAIL-CLOSED LAW`.

\[
\mathcal R_{\rm train}
=
\Pi_{\rm visible}\!\left(y_{\rm solve}-G(q^\star,c^\star)\right)
\]

may be training-eligible only if:

1. the description is of a selected solution or its exact oracle diff;
2. all 25 RG3 demand rows have solved-value type/layer assignments and real
   prices;
3. metric/receiver custody is complete;
4. #669c exhausts all five types across all five layers;
5. metric-active MS2 or a same-scope deterministic competitor converges with
   real-coder price;
6. exact bounded search covers every preregistered hard block;
7. E5 exact parse-back survives receiver → uint8 → \(R\) → frozen scorers;
8. the remaining visible component is nonzero; and
9. the exact byte deficit and exhausted deterministic families are recorded.

If any predicate is false, the result is
`TRAINING_NECESSITY_UNPROVEN`. That is the current result. The current
training-necessary residual set is empty as an evidence claim, not as a
universal mathematical claim.

Training used to fit the score-quotient functional family is a search
algorithm for \(\arg\min S\), not evidence of a training-necessary residual.
Its honest job is to fit entropy-penalized parameters/latents, including the
25 interface placements, against the exact score. Representational necessity
and optimization method must remain separate.

## `ddm_is1_residual_only_finisher_admission_v2`

Status: `STRUCTURAL`.

\[
\operatorname{admit}(a)
\iff
\Delta S_{\rm realized}(a)<0
\land
\Delta B_{\rm exact}(a)\ {\rm counted}
\land
a\ {\rm survives\ parseback/uint8}/R
\land
a\subseteq\mathcal R_{\rm train}.
\]

As a standalone finisher, this is #366’s authority boundary. Inside the
score-quotient functional family, #366/J5 may instead be the fitting engine
for \((f,z)\); that role still requires exact scorer/coder-in-loss validation
and does not make the confounded 2,709,004-error base gap its workload.

## Provenance

- MS1 receipt SHA-256:
  `1b7063a44574b0839ede08c807f348ad417be0492ac32d68634b124b9c2b1e97`.
- MS2 receipt SHA-256:
  `04060edf9834b661f12a9794e50ceadf7dd4ab114baf55a15555537abc71e419`.
- RG2 support summary SHA-256:
  `15b12224e3abb0d93f4fb9693402794d27969783b1d796114f0208277fe5a9ed`.
- RG3 source commit: `4a1728d9ae`; summary SHA-256:
  `3d4c4fb635ec37668cbf6037cefca63fe7c08a9ad950e6724ae023deb0473fd2`.
- Directive hashes are recorded in
  `ddm_is1_full_inverse_solve_to_the_end_directive_consumption_20260724.json`.
- The exact residual receipt hash is recorded in the findings after the
  measurement closes.

No registry row, contest score, promotion, or pointer mutation is claimed
until MAIN reviews these laws.
