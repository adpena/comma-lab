# DDM CC3 mixed-coder receiver equations

Date: 2026-07-25  
Evidence: `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`

## Exact recursive replacement

Let \(Z\) be the counted source composition and let \(L(Z)\) be its ordered
27-leaf recursive ZIP traversal. For the eight CC2-negative rows,
\(c_i\) is the selected G4 or Bellard-KT lossless frame; otherwise
\(c_i(x)=x\). The counted candidate is

\[
  Z_{\mathrm{CC3}} = \operatorname{ZIP}_{\mathrm{same\ metadata,\ order,\ suffix}}
  \{c_i(L_i(Z))\}_{i=1}^{27}.
\]

The receiver applies \(c_i^{-1}\), checks exact final-byte consumption,
raw length, raw SHA-256, and frame re-encoding, then re-emits every ZIP
layer. Admission requires

\[
  \operatorname{restore}(Z_{\mathrm{CC3}})=Z
\]

as byte strings, followed by exact PC1 parse/re-emit.

## Costate and LP1 coordination

The measured counted-byte change is

\[
  \Delta B = 136116-139538=-3422.
\]

Because the independently rendered raw-source control and CC3 receiver
output have identical 3,662,409,600-byte SHA-256 identities,
\(\Delta d_{\rm seg}=\Delta d_{\rm pose}=0\) for this integration.
Therefore

\[
  \Delta S
  =100\Delta d_{\rm seg}
   +\sqrt{10d_{\rm pose}^{\rm after}}-\sqrt{10d_{\rm pose}^{\rm before}}
   +\frac{25}{37545489}\Delta B
  =-0.0022785693375840703.
\]

The coordinated LP1 accounting update is derived, not a new measured
archive identity:

\[
  B_{\mathrm{LP1,post}}=134211-3422=130789.
\]

Equation ID: `ddm_cc3_lossless_recursive_rate_costate_v1`.

## Verdict separation

The lossless integration result is independent of CC2's scorer-reuse
premise. The fresh full receiver score is
\((d_{\rm seg},d_{\rm pose})=(0.024731920030381944,163.0492342914382)\),
which does not equal CC2's reused parent-only endpoint. That reuse is
falsified for this instance; the mixed-vs-raw zero-distortion rate gain
remains measured.
