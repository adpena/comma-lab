# DDM TR2 Crosswalk - Tsallis Regularized Optimal Transport

Tags: [no-triality] [p0-ledger-ok]

## Verdict First

Pointer unmoved. No scorer, no archive, no evaluator, no Metal, and no paid or long job was run.

The useful adoption is narrow: TROT is an `ADOPT-CLASS` design for a scorer-free, byte-only
joint-from-marginals probe on the already measured CR1 edge-conditioned support payload. It is not a
drop-in sparse-coding win, and it does not replace the existing scorer-derived Fisher/margin metric.

The probe to queue is TR2-P1: encode the CR1 selected edge-labeled n600 support as cheap marginals plus
all counted side information and exact correction residuals, then race TROT-q reconstruction against the
incumbent CR1 edge-conditioned stream at `464,557 B` under the same payload and same-coder discipline.
It passes only if total counted bytes are below `464,557 B` and decode equality to the selected
edge-labeled support is exact. No scorer or RGB receiver survival is claimed.

## External Sources Read

- arXiv paper: https://arxiv.org/abs/1609.04495
- Full PDF: https://arxiv.org/pdf/1609.04495
- Author publication/code index: https://borismuzellec.github.io/publications/
- Author TROT implementation: https://github.com/BorisMuzellec/TROT
- POT documentation/index: https://pythonot.github.io/ and https://github.com/PythonOT/POT
- Later sparse/deformed-q caution: https://pmc.ncbi.nlm.nih.gov/articles/PMC7517100/

## Paper Facts

TROT solves a transport problem with Tsallis entropy regularization,
`min_P <P,M> - (1/lambda) H_q(P)` over the transport polytope. The paper explicitly connects the
limits to ordinary OT as `q -> 0` and Sinkhorn/Shannon entropy as `q -> 1`.

The divergence family is broad: the paper maps special q values to KL, Pearson chi-square, Neyman
chi-square, square Hellinger, and the Cressie-Read family. That gives a vocabulary for regularizer
families, not an automatic Pact metric replacement.

The ecological-inference section is the closest Pact match: recover a joint distribution from two
marginals, optionally using side information in the cost matrix. The paper's best source-derived
TROT row (`q=2.8`, `lambda=101`) beats the best Sinkhorn source-derived row in KL on that dataset, and
even the no-prior cost matrix has a useful TROT row. The lesson is not "use the paper's q"; it is
"when the unknown is a joint table and the marginals are cheap, a q-family reconstruction can be a
real comparator."

Sparse support needs stricter language than the charter's seed. The 2016 TROT paper does not prove
that Tsallis regularization by itself is a Pact coding win. Later deformed-q work explicitly separates
deformed-q sparsity from Tsallis entropy and reports that Tsallis did not empirically induce sparsity in
their comparison. Therefore sparsity is a probe hypothesis only.

## OSS Check

The author repo `BorisMuzellec/TROT` is a Python implementation with scripts and notebooks for TROT and
ecological inference. I would treat it as a reference implementation for the $0 probe, not as a vendored
runtime dependency.

I did not find a native Tsallis/TROT solver in the checked POT docs/search results. POT remains useful
for exact OT, Sinkhorn, unbalanced/partial OT, and generic baseline comparisons, but a Pact TROT probe
should either call a pinned local reference outside production or implement the minimal deterministic
solver needed for the byte-only race.

## Recall Evidence

| Query or source | Result | Plan impact |
|---|---|---|
| Charter recall set: #288, #616, #617, #539, #550, #504, m65, rd1, ms2r | Found prior OT/Bregman/Nielsen surfaces with exact head-offset, Brenier/Laguerre/power-diagram, categorical Fisher/Bregman, and typed waterfill null preservation. | Do not re-open metric selection or Laguerre mass-matching as if TROT discovered them. |
| `.omx/research/ddm_cr1_20260808/CR1_FINDINGS.md` and `CR1_ROWS.jsonl` | CR1 P2 measured a byte-only n600 edge-conditioned support stream: `575,095 B -> 464,557 B`, `-110,538 B`, `-19.221%`, exact decode equality, selected support `2,554,360 px`, selected direct flips `506,837`, `0.9165806758 B/flip`. | Sets the incumbent byte target and same-payload comparator for TR2-P1. |
| `.omx/research/ddm_gdl1_20260807/GDL1_CROSSWALK.md` | GDL1 already queued the per-edge conditional carrier/coder race for #984/ty1. | TR2-P1 must be a successor/refinement of the edge-graph row, not a duplicate graph-carrier row. |
| `.omx/research/ddm_rl1_roadlane_interface_price_20260803.md` | RL1 projected a Brotli q11 per-class Lane-mask crop at `1.1604 B/flip` versus `W = 1.273108 B/flip`, but only from n32 evenly-strided GT geometry. | Use RL1 as historical price context only; TR2 must use CR1's n600 support payload for a current byte test. |
| `.omx/research/ddm_g4_spatial_stationarity_603_canonical_equations_20260722T212138Z.md` | G4 measured all flip events `M=4,011,236`, strong pixel recurrence concentration, and a zero-payload context law that cuts ideal KT bits when pixel identity is free. | TROT side information can be free only if generic/decode-deterministic; video-derived marginals/residuals are counted. |
| `.omx/research/ddm_ms2r_r3_366box_typed_fisher_g4_waterfill_20260725T162107Z/DAG_FEED.md` and `EQUATIONS.md` | MS2R preserves 162/162 NULL RD1 cells and refuses typed waterfill until receiver/coder homes and measured adjacent rungs exist. | TROT q/lambda is not allowed to fill RD1 dual prices without same-object receiver-realized bytes. |
| `.omx/research/nielsen_infogeo_crosswalk_20260719_codex.md` | Nielsen crosswalk keeps Euclidean/Fisher/byte secants distinct and already treats Bregman/Fisher as diagnostic unless held-out receiver-closed secants validate allocation. | Divergence-family selection is lesson-only here. |
| `.omx/research/bregman_all_surfaces_504_derivation_20260715.md` | Categorical output Fisher is already the logsumexp Bregman Hessian after quotienting the additive-logit gauge. | TROT does not supersede the scorer-native Fisher metric. |
| `.omx/research/brenier_polar_factorization_crosswalk_20260722_codex.md` | Semi-discrete Brenier/Laguerre head-offset solver exists; n600 mass-matching objective worsened d_seg despite exact solver convergence. | Exact OT solver correctness is not enough; only byte/receiver/scorer outcomes matter. |
| `.omx/research/ddm_sv2_smevr_base_rule_race_20260803.md` | #940 same-coder/races-not-reputation discipline: token coder ideas must beat the live field under identical lossless payload/coder races. | TR2-P1 uses same-payload, counted side-info, exact decode equality, and same-coder controls before any queue promotion. |
| `.omx/state/main_hot_state.md` | Live #984 rate axis exists; own-vehicle pointer is `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`. | Report pointer honesty and route TR2 only as a #984 byte-side queue item. |

## Ranked Crosswalk

| Rank | TROT surface | Disposition | Honesty label | Named consumers | Reason | Follow-on disposition |
|---:|---|---|---|---|---|---|
| 1 | Ecological inference as joint-from-marginals reconstruction for edge-labeled support | `ADOPT-CLASS` | `INFERRED` from paper plus CR1 `MEASURED` bytes; `NOT-MEASURED` here | #984 rate axis, CR1 successor, ty1 edge carrier rows | Pact's closest object is not full RGB transport; it is the selected per-edge joint/support table. Marginals and a generic decode-time q-family solve might reduce the exact correction residual below CR1's incumbent stream. | `QUEUED-WITH-FIRE-ORDER`: run TR2-P1 below. |
| 2 | q/lambda ladder as a regularization/tolerance ladder | `ALREADY-EMBODIED` with lesson-only naming | `DERIVED` analogy; local ladders already governed | rd1, ms2r, tolerance-knee work | TROT connects q=0 OT and q=1 Sinkhorn, but Pact already requires measured adjacent rungs, finite duals, and receiver/coder homes. q cannot populate null RD1 cells. | `FOLDED`: cite as vocabulary only. |
| 3 | Sparse transport plan as a coding property | `LESSON-ONLY` | `CONJECTURE`; paper does not prove Pact coding sparsity | CR1 successor only through TR2-P1 | Later deformed-q literature warns that Tsallis entropy itself is not automatically a sparse-plan guarantee. Sparse support is eligible only if exact counted residual bytes fall. | `FOLDED` into TR2-P1; no independent sparse-plan row. |
| 4 | Divergence-family choice for quotient solve metric | `ALREADY-EMBODIED` / `LESSON-ONLY` | `DERIVED`; scorer metric already selected by local derivations | m65, #504, Nielsen crosswalk, metric diagnostics | The paper's Cressie-Read family is useful language, but Pact's metric is tied to frozen scorer Fisher/margin geometry and must be validated against receiver-closed byte/secant outcomes. | `FOLDED`: no metric change. |
| 5 | Sinkhorn/TROT solver reuse for assignment menus | `N-A` for current assignment surfaces | `SOURCE-CHECKED` local shape plus OSS check | ms5/ms6 menu candidates, small LAP/flow surfaces | Current live assignment surfaces are tiny exact LAP/DP or byte-coder races. TROT does not create a new graph, objective, or byte stream. POT can supply baselines if a future transport graph exists. | `FOLDED`: no standalone solver task. |
| 6 | Author TROT code and POT baselines | `LESSON-ONLY` | `SOURCE-CHECKED`; no production dependency | TR2-P1 implementer | Author code is useful for parity/reference; POT did not expose a native TROT solver in checked docs. A Pact probe must count payload bytes and keep generic algorithm bytes on the free side only when contest-compliant. | `QUEUED-WITH-FIRE-ORDER` only as TR2-P1 implementation reference. |

## TR2-P1 Fire Order

Status: `QUEUED-WITH-FIRE-ORDER`.

Consumer: #984 rate axis, CR1 edge-conditioned support successor.

Axis: `[byte-only scorer-free]`.

Inputs:

- CR1 n600 argmax caches:
  `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy`
  and `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy`.
- CR1 selected top edge-labeled support object and incumbent bytes from
  `.omx/research/ddm_cr1_20260808/CR1_ROWS.jsonl`.
- Selection mode must remain `n600_all_pairs_no_prefix`.

Required arms:

1. Incumbent CR1 edge-conditioned stream: `464,557 B` LZMA raw winner.
2. TROT residual stream: counted marginals, counted video-derived side information, q/lambda/config
   tags, and exact residual corrections needed to reconstruct the same selected edge-labeled support.
3. Sinkhorn q=1 residual stream through the same residual format.
4. Marginals-only and identity/container controls to expose framing overhead and false compression wins.

q grid:

`q in {0.5, 0.8, 1.0, 1.5, 2.8}` with a small deterministic lambda grid seeded by the ecological
paper's scale but selected only by same-payload validation bytes. Any adaptive refinement must be
logged after the base grid and cannot replace it.

Pass condition:

- Exact decode equality to the CR1 selected edge-labeled support arrays.
- Total counted bytes, including all marginals, side-cost payload, tags, residuals, and framing, are
  less than `464,557 B`.
- The Sinkhorn q=1 control and incumbent are reported in the same table.

Falsifier:

- No TROT-q arm beats `464,557 B` under exact decode equality, or a byte win appears only after
  omitting counted video-derived side information. That is a formulation negative for this payload,
  not a kill of all q-family transport.

Forbidden shortcuts:

- Do not call a reconstructed soft plan a lossless support unless the selected edge labels decode
  exactly.
- Do not hide per-frame or per-pixel video-derived tables in code.
- Do not use prefix subsets for a population claim.
- Do not spend the scorer slot; a byte-only lossless payload has no scorer question until a receiver
  consumes it.

## Boundaries

- No scorer/eval/archive work was run.
- No pointer moved.
- No TROT sparsity, receiver survival, RGB realization, or exact contest score is claimed.
- POT was checked as a baseline source; I did not find a native TROT solver in the checked docs.
- The common-contract stale frontier line was superseded by `.omx/state/main_hot_state.md`.

## Typed JSONL

Typed rows are in `.omx/research/ddm_tr2_20260808/TR2_ROWS.jsonl`.

Schema summary:

- `schema`: `ddm_tr2_crosswalk_row.v1`
- `rank`: integer rank in this memo
- `disposition`: one of `ADOPT`, `ADOPT-CLASS`, `LESSON-ONLY`, `ALREADY-EMBODIED`, `N-A`
- `honesty_label`: authority label for this turn
- `named_consumers`: queue or work surfaces consuming the row
- `smallest_zero_dollar_probe`: concrete falsification probe or `null`
- `follow_on_status`: `FIRED`, `FOLDED`, or `QUEUED-WITH-FIRE-ORDER`
- `score_claim`: always false here

## Frontier Honesty

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.

Contest pointer remains the borrowed contest-CPU row `0.1910828242`; it is not moved by this crosswalk.
