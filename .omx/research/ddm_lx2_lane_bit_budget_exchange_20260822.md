# DDM LX2 Lane bit-budget exchange — REFUSED before sweep: the named 19-member control is lossless and cannot instantiate the requested distortion exchange

**No joint-S optimum was measured.** The charter's specified controls—class-conditioned precision,
context depth, or mixing weights inside the shipped 19-member HPAC law—change only the RC64 coding
probabilities. The shipped native source states that the decoded token field is bit-identical and that
distortion is absent by construction. Changing those controls can produce a lossless rate curve, but
not the requested Lane flip, per-class collateral, or MS9-slice curve. Changing decoded symbols would
be a different, presently unspecified lossy token-selection mechanism. Under NO-FAKE, I refused to
invent five distortion-bearing rungs from a probability-only control.

`date_utc: 2026-08-22` · `arm: ddm_lx2_lane_bit_budget_exchange` ·
`disposition: REFUSED_BEFORE_SWEEP_NO_DISTORTION_BEARING_CONTROL` ·
`verdict_scope: FORMULATION — class-conditioned changes confined to the shipped 19-member HPAC
probability law on the fixed DX2 token field` ·
`axis: [macOS-CPU scorer-free retained-field and receiver-source audit, n600]` ·
`score_claim: false` · `promotion_eligible: false`

## Pin verification and BL1 reproduction

All charter pins matched. BL1's primary field was reused read-only; the decoder was not
re-instrumented.

| object | bytes | SHA-256 | status |
|---|---:|---|---|
| DX2 archive | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | MATCH |
| shipped RC64 token stream | 113,777 | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` | MATCH |
| TO2 decoded token field | 117,964,800 | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | MATCH |
| BL1 primary per-position cost field | 943,718,400 | `99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86` | MATCH / REUSED |
| BL1 result | 318,937 | `f8835acf27c3b46bf95f7cd1954e08d72d591854f8f78ac6c902889a064b6621` | MATCH |
| BL1 manifest | 56,421 | `0b2ca8ec51738b6e7ee5940d262be7226457fcd5a4f8e56f4bfb5b98184a59ac` | MATCH |

I independently streamed the retained BL1 f64 field against the retained QS3 DALI-GT argmax and
reconstructed the exact primary-cost multiplicities. Denominators are 117,964,800 positions and
910,209.2806090614 modeled bits. The physical stream is 910,216 bits; the 6.719391-bit residual
reproduces BL1's arithmetic-terminal accounting.

| quantity | independently reproduced value | charter value at displayed precision |
|---|---:|---:|
| top 0.1% positions' bit share | 52.9506876901% | 52.950688% |
| top 1% positions' bit share | 96.3238419085% | 96.323842% |
| weighted Gini | 0.995159378701 | 0.995159 |
| Lane positions / denominator | 690,754 / 117,964,800 | 0.5856% |
| Lane bits / denominator | 305,463.969473 / 910,209.280609 | 33.559751% |
| Lane bits per position | 0.442218169526 | 0.442218 |
| Lane enrichment over population mean | 57.31228964x | 57.31x |

There is no disagreement to adjudicate. Lane's modeled mass is 38,182.996 B-equivalent, but this is
an attribution inside one arithmetic stream, not a separately removable byte section.

## Why the specified sweep cannot move d_seg

The exact shipped native source is
`/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2/runtime/f26_corrector_native.c`, SHA-256
`4405d66c882d7051c7044f23a31623a554a0f845ce2cce31ac4b986410027955`. It fixes `N_FAMILIES=19`
and says, verbatim in the mechanism contract, that only the probability row handed to RC64 changes
and the decoded token field is bit-identical.

The matching Python sources confirm the same boundary:

- `fx2_model_axis_corrector.py`, SHA-256
  `77e81ac827d6d1f820229c7d21b1c749caf18acc23c6635fb327884a0da04be1`, freezes the 19 families,
  class-conditioned mixer context `cls_boundary_agree_homog_ubin8`, and online learning.
- `fx1_logistic_mixer_corrector.py`, SHA-256
  `8038119d065d578b6c163d2ee515e437cab273737ecf82f8c30619844c0f7452`, uses those weights only to
  transform the coding row. The actual symbols are supplied later to `observe`; the weights do not
  choose or coarsen them.
- `free_corrector.py`, SHA-256
  `dd337159bd84e96e767cbde9a6dffecc909e824c2f092399e09095bebaf094a5`, is the receiver wrapper over
  that frozen configuration.

For a synchronized arithmetic encoder/decoder, changing the probability row changes the number and
placement of coded bits but not the decoded symbol sequence. On the fixed `cc10...3eefb` token field,
all legal probability-only rungs therefore have identical renderer input. They move neither MS9's
2,264 surviving representation errors nor its 21,493 manufactured errors. A real render/scorer replay
would verify identity; it would not create a distortion exchange.

To make `d_seg` vary, an encoder must instead choose a different token field—for example, a governed
toward/away-from-argmax substitution. That requires a named selection rule, receiver interpretation,
real re-encode, retained changed field, and real-path scorer measurement. It is not a precision,
context-depth, or mixer-weight allocation inside the lossless 19-member law. Treating it as one would
conflate the probability model with the representation it losslessly transports.

## Requested curve status

| required object | status | boundary |
|---|---|---|
| at least five rungs below and above 33.559751% | **NOT BUILT** | no distortion-bearing control was named |
| real RC64 stream per rung | **NOT MATERIALIZED** | a rate-only sweep would not satisfy this charter's two-currency result gate |
| real-path `d_seg` per rung | **NOT RUN** | MST1 held the sole n600 scorer lane; LX2 had no grant |
| Lane and four-class collateral | **UNMEASURED** | no changed decoded field exists |
| joint `Delta S` and optimum | **UNMEASURED** | neither bytes-only nor flips-only is admitted as a result |
| MS9 slice moved | **NONE MEASURED** | probability-only controls move neither slice by construction |
| prior-law prediction | **UNTESTED** | no optimum-side conclusion is licensed |

This is not evidence that the shipped 33.559751% allocation is optimal. It is evidence that the
charter's proposed control surface cannot answer that question as a rate-distortion exchange. The
negative is `FORMULATION`-scoped, not a closure of class-aware lossy token selection or jointly trained
HPAC representations.

The common contract independently blocked a scorer launch. At preflight,
`ddm_mst1_manufactured_stage_split` held the sole n600 lane under instance
`ddm_mst1_stage_split_n600_20260822`, status `active_local_advisory_instrumentation`. LX2 launched no
scorer, Metal, MPS, Modal, training, archive mutation, or receiver mutation. `upstream/` and the sacred
JO r9 directory were untouched.

## RECALL EVIDENCE

The full-corpus recall searched `.omx/research/`, arm receipts, the canonical research indexes,
`sub015_DAG_*` FEED blocks, design/SPEC surfaces, and live/task ledgers by content for `Lane`,
`bit allocation`, `33.56`, `38,183`, `HPAC`, `19-member`, `mixer`, `class-conditioned`, `RC64`,
`semantic RD`, `coarsen`, `token substitution`, `real re-encode`, `joint exchange`, and `within miss`.
I also queried the canonical equation registry with
`.venv/bin/python tools/list_canonical_equations.py --json` and inspected the live receiver sources.
I did not find another active LX2 lane or a canonical law that permits a probability-only mixer change
to stand in for a changed semantic field in those scopes.

Beyond the charter's seeds, recall found three plan-changing facts:

- The shipped native and Python mechanism contracts explicitly close distortion under mixer changes;
  this converted the planned curve build into a no-fake refusal before any rung was typed or launched.
- `token_rate_model_direction_dependence_v1` and CF2 distinguish real symbol moves toward/away from the
  model argmax from probability-model prices, and require real re-encoding. That prevents borrowing
  those symbol-edit distortion effects for a mixer-weight rung.
- VF1's current-DX2 census requires a complete tuple of real re-encode, receiver-realized Seg/Pose, and
  repeat noise before evaluator-equivalence credit. Its measured denominator is zero, so it supplies no
  missing class-conditioned distortion row. JX1 likewise has no unused measured fifth move.

The existing Road-Lane interface-pricing evidence found during recall is n32 and representation-price
only; it is not n600 current-DX2 scorer evidence and was not promoted into this result.

## Retention and boundaries

The durable preflight receipt is on the charter-required Vertigo tier at
`/Volumes/VertigoDataTier/pact/ddm_lx2_lane_bit_budget_exchange/measurement_v1/MECHANISM_PREFLIGHT.json`.
No receipt was written to APDataStore. No new stream, mask, cost field, token field, render, or other
payload was materialized, so there was no payload to retain or discard. The 943,718,400-byte BL1 field
remains under BL1 ownership and was read in place.

The required serializer landing was attempted with the post-edit memo SHA and message tags
`[no-triality] [p0-ledger-ok]`, but the managed sandbox refused Git's object/index temporary-file write
with `Operation not permitted`. The staged index remained empty. This memo is therefore a verified
working-tree artifact, **not a commit**; no commit hash is claimed.

Measured here: exact pin hashes and sizes, an independent n600 BL1 class/concentration reproduction,
the frozen receiver's control surface, and live scorer-lane ownership. Not measured here: any rung,
changed archive bytes, changed decoded field, scorer result, noise floor, `Delta S`, optimum, candidate,
or new exact score.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — disposition: `BLOCKED_PENDING_CHARTER_MECHANISM_CORRECTION`; owner:
  `MAIN`; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_lx2_lane_bit_budget_exchange/measurement_v2/RESULT.json`; fire
  trigger: MAIN amends LX2 to name a receiver-valid distortion-bearing token-selection control within
  the incumbent lineage, or explicitly narrows LX2 to a lossless rate-only mixer curve; if real-path
  scoring remains required, MST1 is terminal and MAIN transfers the sole scorer slot; the reviewed
  runner is deterministic, stage-checkpointed, retains every rung's stream/masks/fields, and is launched
  only through `tools/fire_local_advisory.py`.

## LIVE-HYPOTHESES

- A receiver-valid, class-aware token selector could reveal an interior optimum below 33.559751%; this
  remains plausible because Lane carries 33.56% of modeled bits and 24.65% of final errors while only
  0.5856% of GT area, but the selector and its collateral are unmeasured.
- A lossless Lane-conditioned mixer retune could reduce total RC64 bytes with exactly zero distortion;
  this is plausible because class is already in the mixer context, but it is a rate-only question and
  does not test the charter's predicted flip budget.

## DEAD-ENDS

- Treating mixer precision, context depth, or weights as a fidelity knob is closed for this shipped
  implementation: those controls change coding probabilities while preserving the decoded field.
- Borrowing FS2/CF2 toward-argmax symbol-edit distortion for mixer rungs is closed: a symbol edit and a
  probability-model retune are different mechanisms and have different arithmetic-context prices.
- Declaring the shipped 33.559751% share optimal from the absence of rungs is closed: the relevant curve
  remains unmeasured.

Own-vehicle frontier remains **S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**,
DX2 archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; LX2 did not move it.
