# SDWL1 canonical equations

date_utc: 2026-07-23
lane_id: lane_ddm_dv2_grammar_sentences_20260723
research_only: true
axis: `[macOS-CPU frozen-scorer advisory]`
score_claim: false

## Frozen inputs and typed fact map

For pair \(t\in\{0,\ldots,T-1\}\), let

\[
L_t\in\{0,\ldots,4\}^{H\times W},\qquad
M_t\in\mathbb R_{\ge 0}^{H\times W},\qquad
p_t\in\mathbb R^6
\]

be the frozen last-frame argmax labels, winner-rival margins, and official
pair-output scalars. SDWL1 does not assert that these are source pixels or a
receiver-closed witness.

For stratum \(k\), define the indicator
\(\chi_{t,k}(y,x)=\mathbf 1[L_t(y,x)=k]\). Its partition-cell record is

\[
C_{t,k}=\left(
\sum\chi,\;
\sum y\chi,\;
\sum x\chi,\;
\operatorname{bbox}_{1/2\text{-open}}(\chi),\;
\beta_0^{(4)}(\chi)
\right),
\]

where \(\beta_0^{(4)}\) is the exact four-connected component count. Empty
cells use the canonical bounding box \((-1,-1,-1,-1)\).

With margin bands
\(\mathcal B=\{[0,.1),[.1,.5),[.5,1),[1,\infty)\}\), the separatrix record is

\[
S_{t,k}=\left(
\sum_{y,x<W-1}\mathbf 1[\chi(y,x)\ne\chi(y,x+1)],\;
\sum_{y<H-1,x}\mathbf 1[\chi(y,x)\ne\chi(y+1,x)],\;
\left\{\sum\chi\,\mathbf 1[M_t\in b]\right\}_{b\in\mathcal B}
\right).
\]

Let \(u_{t,j}=\operatorname{bits}_{64}(p_{t,j})\in\mathbb Z/2^{64}\mathbb Z\).
The pair-screw record is

\[
Q_t=(u_{t,0},\ldots,u_{t,5}).
\]

The complete per-pair tensor is the typed concatenation

\[
F_t=\operatorname{pack}\left(
\{C_{t,k}\}_{k=0}^{4},
\{S_{t,k}\}_{k=0}^{4},
Q_t
\right)\in\mathbb Z^{11\times 8}.
\]

The five cell rows use all eight scalar slots; each of the five separatrix rows
and the screw row uses six. Therefore

\[
N_{\mathrm{record}}=11T,\qquad
N_{\mathrm{fact}}=(5\cdot8+5\cdot6+6)T=76T.
\]

The remaining 12 storage slots per pair are canonical zeros and are rejected if
nonzero. For \(T=600\), the measured inventory contains 6,600 records and
45,600 non-padding scalar facts.

## Temporal grammar

The absolute arm encodes \(E_t=F_t\). The causal arm is

\[
E_0=F_0,\qquad
E_{t,r,j}=F_{t,r,j}-F_{t-1,r,j}
\]

for the ten discrete rows, while the six screw coordinates use

\[
E^{Q}_{t,j}=u_{t,j}-u_{t-1,j}\pmod {2^{64}}.
\]

Decoding is the exact inverse cumulative sum, with screw accumulation also
performed modulo \(2^{64}\). This preserves every finite float64 bit pattern,
including signed zero.

Predicates are inferred deterministic productions, not a second semantic
payload. For \(t=0\), every record produces `declare`. For \(t>0\):

\[
\pi(F_{t,r},F_{t-1,r})=
\begin{cases}
\texttt{hold} & F_{t,r}=F_{t-1,r},\\
\texttt{topology\_delta} & r<5\land \Delta\beta_0^{(4)}\ne0,\\
\texttt{transport} & r=10,\\
\texttt{deform} & \text{otherwise}.
\end{cases}
\]

The measured n600 counts are 11 declarations, 4,928 deformations, zero holds,
1,062 topology deltas (530 births, 532 deaths), and 599 transports.

## Byte syntax and objective

For numeric section \(a\), let

\[
\mathcal C_{557}(a)
=\operatorname{LeftUpArithmeticEncode}(a)
\]

be the repository's real spatial-context arithmetic coder. Let
\(\operatorname{Frame}\) include packet/section magic and versions, canonical
JSON lexicon and schema, exact lengths, and SHA-256 digests. For layout
\(\ell\), temporal mode \(\tau\), and optional same-semantics syntax \(d\),

\[
W_{\ell,\tau,d}
=\operatorname{Frame}\left(
J_{\mathrm{lex}},J_{\mathrm{schema}},
\{\mathcal C_{557}(a_s)\}_{s\in\ell},
d
\right).
\]

The only byte verdict is the complete coupled outer object:

\[
B(\ell,\tau,d)
=\left|\operatorname{zlib}_{9}\left(W_{\ell,\tau,d}\right)\right|.
\]

For 600 independent descriptions, every pair has its own absolute packet and
arithmetic reset:

\[
W^{\mathrm{ind}}_\ell
=\operatorname{CollectionFrame}
\left(\operatorname{Frame}(F_0)\Vert\cdots\Vert
\operatorname{Frame}(F_{T-1})\right),
\quad
B^{\mathrm{ind}}_\ell
=|\operatorname{zlib}_9(W^{\mathrm{ind}}_\ell)|.
\]

Thus the same-layout temporal-sharing gain is

\[
G_{\ell,\tau}=B^{\mathrm{ind}}_\ell-B(\ell,\tau,\varnothing).
\]

For the admitted typed causal row,

\[
B^{\mathrm{ind}}_{\mathrm{typed}}=521{,}139,\quad
B_{\mathrm{typed,delta}}=68{,}464,\quad
G=452{,}675\;(86.8626\%).
\]

## Empirical dimension and MDL admission

Layout and temporal mode are selected only on complete full-coverage payloads:

\[
(\ell^\*,\tau^\*)=
\arg\min_{\ell\in
\{\mathrm{mono},\mathrm{typed},\mathrm{stratum}\},
\tau\in\{\mathrm{absolute},\mathrm{delta}\}}
B(\ell,\tau,\varnothing).
\]

Measured best-layout bytes were

\[
B_{\mathrm{mono}}=91{,}958,\qquad
B_{\mathrm{typed}}=68{,}464,\qquad
B_{\mathrm{stratum}}=75{,}197,
\]

so \((\ell^\*,\tau^\*)=(\mathrm{typed},\mathrm{delta})\).

For an optional same-semantics dimension \(d\),

\[
\Delta_{\mathrm{MDL}}(d)
=B(\ell^\*,\tau^\*,d)-B(\ell^\*,\tau^\*,\varnothing),
\]

and \(d\) is admitted iff \(\Delta_{\mathrm{MDL}}(d)<0\). Measured deltas were
\(+938\) frame-index bytes, \(+226\) repeated-provenance bytes, \(+157\)
event-mask bytes, and \(+11\) split-topology-vocabulary bytes. Therefore all
four are pruned. Any subject, predicate, or modifier with measured use count
zero is also omitted from the charged lexicon.

## Provenance boundary

The type registry is derived from frozen scorer geometry: last-frame SegNet
versus pairwise PoseNet roles, shared 512x384 bilinear map, 2x2 chroma box,
rank-four Laguerre head geometry, range/kernel resize split, exact resize
adjoint, measured ERF and Fisher/margin structure, Morse-Smale separatrices,
se(3)/Chasles pair motion, power diagrams, road-frame geometry, and the local
polytope/KKT/tropical/Whitney corpus. Every declared type has named repository
provenance; only productions present in \(F\) are charged.

These equations canonically describe the SDWL1 fact and byte surfaces only.
They do not define a pixel renderer, receiver proof, contest archive, scorer
value, or promotion rule.
