# Canonical equations note — typed E4/WS1 admission

`research_only=true` · `score_claim=false` ·
`[macOS-CPU frozen-scorer advisory]`.

For archive bytes `A`, grammar streams `s_i = (o_i, n_i, h_i, c_i)`, and
receiver `R_g`, admission is:

`Admit_g(A) = [g = g_ws1] ∧ [o_0 = 0] ∧ [o_{i+1} = o_i+n_i] ∧
[Σ_i n_i = |A|] ∧ [SHA256(A[o_i:o_i+n_i]) = h_i] ∧ [c_i is a named
receiver consumer] ∧ [parse_g(A).reemit() = A]`.

The packet proof is:

`P = E4_Brotli_Q11(A)`, `E4_parse(P) = A`, and
`SHA256(R_g(E4_parse(P))) = SHA256(R_g(A))`.

The named rate law is:

`ΔS_rate = 25 * (|P| - |A|) / 37,545,489`.

Measured instances are `-0.004768215963307869` for `W_seg` and
`-0.00499860316108814` for `W_joint`.

The fallback feasibility predicate is stricter than the outer packet codec:

`FallbackClosed(A) = [ImportError(brotli)] ∧
[all dependencies of R_g(A) remain available]`.

Both sealed WS1 instances contain a Brotli-coded G1 worldsheet stream, so the
second conjunct is false. Therefore the raw-LZMA1 outer-packet fallback is
`BLOCKED_FAIL_CLOSED` for this formulation. This does not weaken E4's legacy
V15 fallback or make a family-level claim about LZMA1 or WS1.
