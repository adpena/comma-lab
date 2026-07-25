# DDM LA1 canonical equations

## Uniform real-coder price

For exact stream payload \(x_s\), all arms pay the same 46-byte envelope
containing codec identity, \(|x_s|\), and \(\operatorname{SHA256}(x_s)\).

\[
B_{\rm explicit}(s)
=46+\min\{
|x_s|,
|\operatorname{Brotli}_{11}(x_s)|,
|\operatorname{LZMA1}_{\rm raw}(x_s)|
\},
\]

\[
B_{\rm context}(s)
=\min\{
|\operatorname{G4Frame}(x_s)|,
|\operatorname{BellardKTFrame}(x_s)|
\}.
\]

The G4 and Bellard frames already include the same 46-byte envelope. Both
models derive all probability state causally from the already-decoded prefix,
so counted model-parameter bytes are zero.

\[
\operatorname{owner}(s)=
\begin{cases}
\texttt{CONTEXT}, & B_{\rm context}(s)<B_{\rm explicit}(s),\\
\texttt{RESIDUAL}, & \text{otherwise}.
\end{cases}
\]

LA1 reports
\(\Delta B_s=B_{\rm context}(s)-B_{\rm explicit}(s)\), so negative values
select CONTEXT.

## Context-mass falsifier

The falsifier is evaluated on source accounting mass, not compressed output
mass:

\[
f_{\rm context}
=\frac{\sum_s B_{\rm C1}(s)
  \mathbf{1}[\operatorname{owner}(s)=\texttt{CONTEXT}]}
 {\sum_s B_{\rm C1}(s)}
=\frac{355}{134211}
=0.002645088703608497.
\]

Since \(f_{\rm context}<0.01\), CONTEXT is priced out only for the current
seven-home instance. A new same-object scorer-recursive geometry reopens it.

## Deepest-home assignment

For a stream \(s\) and scorer-recursion layer \(\ell\), let
\(F_{\ell\rightarrow5}(s)\) denote the receiver/scorer path by which the
stream affects the L5 argmax/Pose output. The admissible home is

\[
h(s)=\max\{\ell: F_{\ell\rightarrow5}(s)\text{ is evidenced and the
same-object payload is materialized}\}.
\]

Re-tagging one identical payload across admissible layers does not create a
byte gain:

\[
B(s,\ell)=B_{\rm selected}(s)
\quad\text{for every measured same-payload candidate }\ell\le h(s).
\]

A deeper unmaterialized representation has price `NULL`, not zero. L5 payload
homes containing scorer weights or ground-truth tables are inadmissible.

## Seven-home conservation

\[
\sum_s B_{\rm C1}(s)
=3345+100099+29878+85+151+383+270
=134211.
\]

\[
\sum_s B_{\rm selected}(s)=128254,
\qquad
\Delta B_{\rm LA1}=128254-134211=-5957.
\]

## CC3 coordination

CC3 and LA1 change overlapping descriptions of the same accounting object.
Their deltas are alternatives, not orthogonal increments:

\[
B_{\rm coordinated}
=\min(B_{\rm CC3},B_{\rm LA1})
=\min(130789,128254)
=128254.
\]

\[
\Delta B_{\rm vs\,CC3}=128254-130789=-2535.
\]

This is prospective until E5 proves receiver-closed selected-frame
consumption and exact output identity. It is not a contest score.
