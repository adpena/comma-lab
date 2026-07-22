---
title: Year-blind source-coding reference mine for DDM v3
date_utc: 2026-07-22T04:47:13Z
task: 603
feeds_task: 613
research_only: true
no_recency_floor: true
main_landing_review_required: true
---

# Selection rule

Publication year is not a feature. A result enters the ranked stream table only if its source model
matches a real stream and either (a) an exact theorem gives a useful rate, or (b) its complete local
bitstream is measured. Representation-changing mathematics remains a build hypothesis until the new
semantic object is receiver-proven and distortion-constrained.

# Primary sources and decision value

| Work | Primary source | Exact result relevant here | Stream decision |
|---|---|---|---|
| Huffman, 1952, minimum-redundancy prefix codes | [IEEE DOI 10.1109/JRPROC.1952.273898](https://doi.org/10.1109/JRPROC.1952.273898) | Optimal symbol-by-symbol binary prefix code for a known discrete distribution | Baseline theorem, but integer/model headers and non-memoryless structure make whole-stream measurement decisive. |
| McMillan, 1956, unique decipherability inequalities | [IEEE DOI 10.1109/TIT.1956.1056818](https://doi.org/10.1109/TIT.1956.1056818) | Extends the Kraft length constraint to uniquely decipherable codes | Feasibility bound for every prefix framing; not itself a stream coder. |
| Golomb, 1966, run-length encoding | [IEEE DOI 10.1109/TIT.1966.1053907](https://doi.org/10.1109/TIT.1966.1053907) | Gives the explicit Huffman-optimal family for geometric run lengths | Measured on static-mask runs, xi values, Pose temporal deltas, events, and exceptions. It wins the Pose proxy at Rice `k=6`. |
| Rice and Plaunt, 1971, adaptive variable-length spacecraft coding | [NASA NTRS 19720033735](https://ntrs.nasa.gov/citations/19720033735) | Adaptive concatenated codes reported within 0.25 bit/pixel of one-dimensional difference entropy over a broad range | Supports cheap parameter/block selection rather than one globally fixed `k`; v3 should store the selected `k`. |
| Elias, 1975, universal integer codeword sets | [IEEE DOI 10.1109/TIT.1975.1055349](https://doi.org/10.1109/TIT.1975.1055349) | Universal prefix representations of positive integers, with asymptotically optimal families | Gamma/delta bit counts measured exactly. Delta nearly matches Brotli on the static run stream but loses elsewhere. |
| Tunstall, 1967, variable-to-fixed noiseless codes | [Georgia Tech dissertation record](https://repository.gatech.edu/entities/publication/52d5897e-54ca-4d22-bd29-b9a750fccd46) | Variable-length source phrases mapped to fixed-length codewords for a stationary memoryless source | Candidate for tiny event-type alphabets, but dictionary cost must be counted; not promoted without a measured real-stream win. |
| Cover, 1973, enumerative source encoding | [IEEE DOI 10.1109/TIT.1973.1054929](https://doi.org/10.1109/TIT.1973.1054929) | Indexing a member of a known set costs approximately `log2 |S|`; constant-weight binary support costs `log2 C(N,K)` | Colex/binomial counts measured on all applicable real supports. Dense/large supports make it lose here. |
| Rissanen and Langdon, 1979, arithmetic coding | [IBM DOI 10.1147/RD.232.0149](https://doi.org/10.1147/RD.232.0149) | Separates source model from finite-state arithmetic channel coding and approaches model log-loss | The strict local AQC1 bitstreams and #557 context models are valid tests of particular models, not a family-wide verdict. |
| Witten, Neal, and Cleary, 1987, practical arithmetic coding | [ACM DOI 10.1145/214762.214771](https://doi.org/10.1145/214762.214771) | Practical adaptive arithmetic coding with explicit model/coder separation | Reinforces the rule that the model must be ranked, not merely the arithmetic backend. |
| Krichevsky and Trofimov, 1981, universal encoding | [IEEE DOI 10.1109/TIT.1981.1056331](https://doi.org/10.1109/TIT.1981.1056331) | Universal sequential probability assignment with finite-sample redundancy control | KT0/KT1 negative-log-probability ceilings were computed. They do not beat the complete winners and are not mislabeled as byte streams. |
| Willems, Shtarkov, and Tjalkens, 1995, CTW | [IEEE DOI 10.1109/18.382012](https://doi.org/10.1109/18.382012) | Binary context-tree mixture achieves the relevant finite-tree redundancy bound | High-value only if v3 streams exhibit stable deeper context after framing. KT1's losses make a CTW build secondary to exception recode and event representation work. |
| Cleary and Witten, 1984, PPM | [IEEE DOI 10.1109/TCOM.1984.1096090](https://doi.org/10.1109/TCOM.1984.1096090) | Adaptive higher-order context modeling with escape handling | Plausible for global exception/event record concatenations; complete model/escape/termination bytes are owed before ranking. |
| Burrows and Wheeler, 1994, block sorting | [DEC SRC Research Report 124 catalog record](https://cir.nii.ac.jp/crid/1573668923835612928) | Reversible block sorting clusters context to expose runs before simple entropy coding | Relevant to long exception/event byte streams; not assumed superior to the measured Brotli/LZMA rows. |
| Duda, ANS | [2009 primary preprint](https://arxiv.org/abs/0902.0271), [2013 expanded preprint](https://arxiv.org/abs/1311.2540) | Entropy coding near the modeled entropy with table/state mechanics suitable for fast implementations | ANS can replace the channel coder, not the model. At 600 records, table/state/termination bytes must beat the measured whole stream. |
| Slepian and Wolf, 1973 | [IEEE DOI 10.1109/TIT.1973.1055037](https://doi.org/10.1109/TIT.1973.1055037) | Lossless correlated-source rate region with decoder side information | Only applicable if another receiver-owned stream is sealed as side information; no free side information is assumed. |
| Wyner and Ziv, 1976 | [IEEE DOI 10.1109/TIT.1976.1055568](https://doi.org/10.1109/TIT.1976.1055568) | Rate-distortion with decoder side information | Potential representation-level xi/Pose coupling, not a lossless recode of the frozen six streams. |
| Rissanen, 1978, shortest data description | [DOI 10.1016/0005-1098(78)90005-5](https://doi.org/10.1016/0005-1098(78)90005-5) | Select model plus parameters by total description length rather than payload entropy alone | Governs the v3 selector: count codec tag, model/table, headers, termination, decoder dependencies, and final container delta. |

# 1800s-1930s representation mine

These candidates were consulted year-blind through the project coder corpus and no-recency doctrine.
They act *before* entropy coding, so the survey does not invent exact-lossless byte claims for them.

| Lineage | Mathematical promise | Applicable object | Exact gate before adoption |
|---|---|---|---|
| continued fractions; Stern-Brocot/Farey | Short rational approximants and canonical rational paths | static coefficients and xi knots | Count numerator/denominator/path plus scale and receiver code; preserve Seg cells and Pose tube. |
| Chebyshev, Legendre, Gram bases | Near-minimax or orthogonal low-order descriptions of smooth curves | xi-curve controls and PPCS trajectory | Fit on real controls, count basis/degree/coefficient syntax, parse back, and pass receiver constraints. |
| Gauss/Lagrange lattice reduction | Short integer basis for correlated coefficient vectors | coefficient blocks and cross-coordinate dxi | The transform matrix and inverse must be counted or free by sealed grammar; exact integer parse-back required. |
| combinadic/binomial number system | Rank a constant-weight support in `ceil(log2 C(N,K))` bits | mask/event/exception positions | Already measured by colex counts; rejected on current dense/large supports, family remains open for genuinely sparse successor sets. |
| Pruefer, 1918 | Encode a labeled tree with `n-2` labels | future event or grammar tree | A receiver-owned tree must actually exist; current flat records do not justify a tree tax. |
| Whittaker, 1915 sampling | Reconstruct band-limited functions from samples | smooth temporal xi carrier | Requires a verified band-limit/approximation error and counted sample description under the same evaluator-cell constraints. |

# Other corpus lines screened but not promoted

| Line | Why it does not displace the measured assignments |
|---|---|
| Kraft's 1949 thesis and Shannon-Fano-Elias lengths | They give prefix feasibility/construction bounds, not a better finite stream model. Complete Huffman/arithmetic/dictionary bytes decide here. |
| canonical-Huffman length-vector rank, project L26 | It optimally enumerates a Kraft-valid code-length-vector family. The current six real objects are not code-length vectors; apply it only to a future codebook description. |
| Lynch-Davisson universal coding | Important universal-source lineage, but no complete local finite-stream implementation/model row was available. KT0/KT1 ceilings cover the immediate unknown-probability check without inventing bytes. |
| LBG vector quantization and TCQ | Representation/quantization candidates for coefficients and dxi, not lossless coders for frozen semantic bytes. They need a real distortion-constrained receiver experiment. |
| fractal/PIFS, EZW/SPIHT/EBCOT, MPEG-4 CAE | Potentially valuable changed representations for images, masks, or boundaries. They do not provide an exact recode of the present PPCS/PCE3/PCOMP3 objects and therefore remain upstream reformulations. |
| learned priors | Admissible only when model plus weights plus stream plus decoder beats the classical complete byte count at this 600-record scale. No such measured row exists here. |

# Explicit non-results

- No source publication proves that Brotli or LZMA is globally optimal. Their assignments are empirical
  winners on these exact bytes.
- No entropy theorem removes representation cost, model tables, termination, codec selection tags,
  final ZIP framing, or receiver code.
- No CTW/PPM/ANS/learned-prior byte row exists here; those families are open, but unmeasured estimates
  cannot displace exact local winners.
- Constriction is an implementation surface, not a separate information-theoretic optimum; its table
  and decoder dependency still count.

0.1910828242 [contest-CPU] — unchanged by construction.
