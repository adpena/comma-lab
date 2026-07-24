# DDM DM3 CONNECTION-1 — canonical-equation note

`research_only=true` · `score_claim=false` ·
`evidence_axis=[macOS-CPU frozen-scorer advisory]` ·
`main_review_required=true`

## `ddm_dm3_heldout_connection_conditional_codelength_v1`

Status: `MEASURED INSTANCE LAW + STRUCTURAL FIREWALL`.

For eligible bucket family \(b\), let \(h_b=(p,p+1)\) be its deterministic
lower-median held-out transition and let
\(\mathcal T_b\setminus\{h_b\}\) be the fit population. The static price is:

\[
B^{\rm static}_b =
\min_{c\in\{\mathrm{zlib9},\mathrm{lzma9},\mathrm{ctx1}\}}
\left|C_c(r_{b,p+1})\right|.
\]

For generic history family
\(g\in\{\mathrm{identity},\xi\text{-advected},\mathrm{affine}\}\), fitted
without \(h_b\):

\[
B^{\rm hist}_{b,g}
= B^{\rm frame}+B^{\rm selector}+B^{\rm state}_{b,g}
 \min_c\left|C_c\!\left(r_{b,p+1}\oplus
\widehat r_{b,p+1}^{\,g}\right)\right|.
\]

The charged program costs are 16 B for identity, 20 B for ξ-advection, and
40 B for affine tracking. The held-out conditional saving is:

\[
\Delta B_b =
B^{\rm static}_b-\min_g B^{\rm hist}_{b,g},\qquad
\Delta B_{\rm CONNECTION}=\sum_b\Delta B_b.
\]

Admission requires `pair_gap=1`, held-out exclusion from fit, deterministic
state/coder tie breaks, and exact packet parseback to \(r_{b,p+1}\).

## SHA-bound measured anchor

For 36 eligible bucket families:

\[
\sum_b B^{\rm static}_b=7{,}049\ {\rm B},
\quad
\sum_b B^{\rm program}_b=624\ {\rm B},
\quad
\sum_b B^{\rm residual}_b=5{,}237\ {\rm B},
\]

\[
\boxed{\Delta B_{\rm CONNECTION}=+1{,}188\ {\rm B}}.
\]

Thirty-two rows are positive and four nonpositive. The selected families are
identity 34, ξ-advection 0, and affine tracking 2. Identity owns all 1,557
positive selected-family savings. The static-boundary grouped row is negative
(-242 B), while static-cell (+657 B), transient-boundary (+242 B),
transient-cell (+492 B), and the one static-ξ-proxy boundary row (+39 B) are
positive.

## Typology and NULL law

\[
\text{history program}\mapsto(\mathrm{CONNECTION},L4),\qquad
\text{exact correction}\mapsto(\mathrm{RESIDUAL},L4).
\]

No new stream type or layer was created. A bucket with no same-bucket
consecutive transition has:

\[
B_{\rm CONNECTION}(b)=\mathrm{NULL},
\]

not zero. The only such represented family is
`lane_mycar__cell__transient`.

## Receiver/score firewall

\[
\Delta B_{\rm archive}
=\mathrm{NULL}
\quad\text{until}\quad
D_{\rm RGB}
\xrightarrow{\mathrm{uint8},R,\mathrm{Seg},\mathrm{Pose}}
\text{the receiver-closed verdict}.
\]

The callable therefore returns
`score_slack_arithmetic_permitted=false`,
`score_claim=false`, and `pointer_moved=false`. The +1,188 B semantic delta
must not enter #613 slack, tangent, archive, score, promotion, or frontier
arithmetic.

## Callable and registry receipt

Module:
`tac.canonical_equations.ddm_dm3_connection_conditional_codelength_20260724`.

Callable:
`account_dm3_connection_rows(rows)`.

Registry helper:
`populate_ddm_dm3_heldout_connection_conditional_codelength_v1`.

The helper appended the equation through `register_canonical_equation` with
`subagent_id=codex_delegate:ddm_dm3_connection1_conditional_codelength:20260724T135912Z`
and source receipt SHA
`2c175366da196b8f79e2d3de1ad0a8c1844e78a1d2cab25921eaa15bec46346b`.
Direct evaluation over the receipt rederived 7,049/624/5,237/+1,188 B,
32 positive and four nonpositive rows, with family counts 34/0/2.

## Recalibration trigger

Rebuild after any solved-plane, PF2 event-index, frozen SegNet, DM1 semantic
grammar/coder, history packet, correspondence, fit, or holdout-policy SHA
changes. An all-fold or receiver-closed measurement adds a new anchor; it must
not silently promote this deterministic one-fold semantic result.
