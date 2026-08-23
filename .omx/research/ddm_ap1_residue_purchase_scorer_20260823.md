# DDM AP1 — DX2 residue purchase scorer

Date: 2026-08-23
Verdict scope: `INSTANCE:DX2_N600_FIXED-CODER_ONE-GROUP_NATURAL-QUANTIZATION_LADDERS`
Measurement axis: `[macOS-CPU advisory; DALI-GT pinned n600]`
Promotable score claim: **no**

## Verdict

All twelve registered coarsenings are **net-positive**. The net-negative set is empty, returns
**0 B**, and clears **0/42,382 B = 0%** of the rate demand. The prior-law prediction that at least
one group over 5,000 B would coarsen net-negatively is therefore **FALSIFIED on this instance and
these measured natural ladders**.

The cheapest measured damage is the carrier lattice-step-2 row: it returns 2,742 B but costs
`+0.306332390066 S` in distortion and lands at `+0.304506604816 S` net. The largest single byte
credit is semantic level 3 at 18,427 B, but it costs `+17.8549944410 S` net. Even the carrier,
whose SegNet argmax field is exactly unchanged at all three levels, is strongly load-bearing on
PoseNet. The 96 B residual table returns zero ZIP bytes at every tested level while causing
catastrophic distortion.

This closes the charter's fixed-coder, one-group allocation question for the current DX2 body:
there is no measured residue purchase to harvest. Composed with LD1/LX2's token-allocation closure
and the five-arm current-field coding closure, AP1 routes away from another current-body
single-group allocation pass and toward a representation born with different field/body
coupling. It does **not** settle the already-fired JF1 joint field-plus-model refit diagonal, which
changes two surfaces together and lies outside AP1's isolation rule.

No shipping candidate was built. The own-vehicle frontier is unmoved.

## Exact residue census

The pinned DX2 archive is 180,368 B, SHA-256
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`. Its pinned RC64 token
stream is 113,777 B, SHA-256
`e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5`. The AR1B census was
reproduced exactly before any perturbation:

| residue class | bytes | share of 66,591 B | share of 42,382 B demand |
|---|---:|---:|---:|
| semantic renderer | 30,856 | 46.3378% | 72.8045% |
| carrier | 22,010 | 33.0526% | 51.9324% |
| HPAC probability model | 13,515 | 20.2976% | 31.8885% |
| fixed residual table | 96 | 0.1442% | 0.2265% |
| ZIP + RX1 structural framing | 114 | 0.1712% | 0.2690% |
| **total** | **66,591** | **100.0000%** | **157.1209%** |
| unexplained remainder | **0** | **0%** | **0%** |

The 114 B framing row is receiver-required structure, not a parameter group with a natural
quantization axis. It was held exact and is not assigned a distortion-load-bearing label or a
fictional coarsening credit.

## Authority, denominators, and control

Every candidate was rendered through its copied public receiver, resize/R path, uint8 output, and
the frozen CPU-torch SegNet/PoseNet. Candidate outputs were reduced against the pinned
contest-CUDA DALI-GT tables:

- SegNet GT argmax: 600 x 384 x 512 = **117,964,800 pixels**, SHA-256
  `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248`.
- PoseNet GT first-six: 600 x 6 = **3,600 values**, SHA-256
  `8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff`.
- Per-class SegNet denominators: Road 27,407,372; Lane 690,754; Undrivable 58,413,067;
  Movable 1,460,386; MyCar 29,993,221.

The exact local control is `d_seg=0.00020134819878472223` (23,752/117,964,800),
`d_pose=0.000006365873831275037` (3,600-value mean), and recomputed
`S=0.14821311017119657` at 180,368 B. This is an advisory diagnostic, not a new frontier row:
the contest-CUDA authority control has five more segmentation flips and remains the pointer.

The exchange used below is the canonical TX1 value
`25/37,545,489 = 6.658589531221714e-7 S/B`. No score is copied from evaluate.py's rounded display;
all `S` terms are recomputed from components.

## Natural coarsening ladders

- Semantic: preserve the shipped keep pattern and coder; level 1 maps q4 tensors to q3 while
  keeping existing q3 tensors, level 2 maps q4 to q2 while keeping existing q3, and level 3 maps
  every quantized tensor to q2.
- Carrier: coarsen the signed-5 basis codes and signed-12 coefficient codes to nearest lattice
  steps 2, 4, and 8. The shipped 36-byte CAP1 predictor metadata, CAP1 coder, DX2/RR5 riders,
  Brotli q9/lgwin16, selector, and all other sections stay fixed.
- HPAC: decrement every IHS1 row's signed bit depth by 1, 2, and 3 respectively, clipping only to
  the new legal row domain; row order, model topology, tail, and coder stay fixed.
- Residual: coarsen the signed-6 fixed residual codes to nearest lattice steps 2, 4, and 8; the
  fp16 scale and RC64 token stream stay exact.

The level-zero identity controls reproduced the semantic stream, canonical CPR1, source CAP1,
physical carrier stream, HPAC stream, and residual-plus-token tail byte-for-byte.

## Purchase table

`Delta d_seg` is over 117,964,800 pixels; `Delta d_pose` is over 3,600 values. `Delta S_dist`
is `100*Delta d_seg + sqrt(10*d_pose_candidate) - sqrt(10*d_pose_control)`. Credit is measured
from the complete re-encoded ZIP. All rows are receiver-required (`RR=YES`): the shipped reader
parses and consumes each target field (with MZ2 independently establishing all 38 semantic
tensors). All are distortion-load-bearing (`DLB=YES`) because each measured coarsening increases
realized total distortion after receiver parse-back. Positive net is worse.

| candidate | held B | archive B | credit B | Delta d_seg | Delta d_pose | Delta S_dist | credit S | net Delta S | S damage/B credited | RR | DLB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| semantic_l1 | 24,970 | 174,482 | 5,886 | 0.00871830410428 | 5.90306338870 | 8.54699549584 | 0.00391924579808 | **8.54307625004** | 0.00145208893915 | YES | YES |
| semantic_l2 | 13,841 | 163,353 | 17,015 | 0.0571536509196 | 13.1994702791 | 17.1962839791 | 0.0113295900874 | **17.1849543890** | 0.00101065436257 | YES | YES |
| semantic_l3 | 12,429 | 161,941 | 18,427 | 0.0610589599609 | 13.8517462099 | 17.8672642239 | 0.0122697829292 | **17.8549944410** | 0.000969624150645 | YES | YES |
| carrier_l1_fixed_coder | 19,268 | 177,626 | 2,742 | 0 | 0.00987277665396 | 0.306332390066 | 0.00182578524946 | **0.304506604816** | 0.000111718595939 | YES | YES |
| carrier_l2_fixed_coder | 16,135 | 174,493 | 5,875 | 0 | 0.0546586253751 | 0.731379127655 | 0.00391192134959 | **0.727467206305** | 0.000124490064282 | YES | YES |
| carrier_l3_fixed_coder | 12,975 | 171,333 | 9,035 | 0 | 1.54927024096 | 3.92810647691 | 0.00601603564146 | **3.92209044127** | 0.000434765520411 | YES | YES |
| hpac_l1 | 11,603 | 178,456 | 1,912 | 0.670700480143 | 135.487521179 | 103.870699820 | 0.00127312231837 | **103.869426697** | 0.0543256798219 | YES | YES |
| hpac_l2 | 9,607 | 176,460 | 3,908 | 0.488460515340 | 136.528198192 | 85.7877961051 | 0.00260217678880 | **85.7851939284** | 0.0219518413780 | YES | YES |
| hpac_l3 | 7,489 | 174,342 | 6,026 | 0.503762986925 | 136.834989981 | 87.3595347294 | 0.00401246605151 | **87.3555222633** | 0.0144971016809 | YES | YES |
| residual_l1 | 96 | 180,368 | 0 | 0.406644278632 | 144.837491588 | 78.7139712043 | 0 | **78.7139712043** | n/a | YES | YES |
| residual_l2 | 96 | 180,368 | 0 | 0.421007554796 | 148.120722400 | 80.5792327359 | 0 | **80.5792327359** | n/a | YES | YES |
| residual_l3 | 96 | 180,368 | 0 | 0.409847827488 | 144.183163956 | 78.9482631671 | 0 | **78.9482631671** | n/a | YES | YES |

## Realized per-class SegNet response

Each cell is `Delta flips / Delta conditional d_seg` against the exact control. Lane is kept on its
own row. These are measured values, not interpolations.

### Semantic renderer

| GT class (denominator) | L1 | L2 | L3 |
|---|---:|---:|---:|
| Road (27,407,372) | +859,074 / +0.0313446323858 | +5,045,790 / +0.184103386490 | +5,295,946 / +0.193230711795 |
| Lane (690,754) | +73,685 / +0.106673287451 | +348,499 / +0.504519698764 | +352,451 / +0.510240983042 |
| Undrivable (58,413,067) | +59,786 / +0.00102350386772 | +317,288 / +0.00543179833375 | +313,144 / +0.00536085530315 |
| Movable (1,460,386) | +27,148 / +0.0185896057618 | +415,466 / +0.284490538803 | +444,215 / +0.304176430067 |
| MyCar (29,993,221) | +8,760 / +0.000292065997180 | +615,076 / +0.0205071672696 | +797,052 / +0.0265744049297 |

### Carrier

| GT class (denominator) | L1 | L2 | L3 |
|---|---:|---:|---:|
| Road (27,407,372) | 0 / 0 | 0 / 0 | 0 / 0 |
| Lane (690,754) | 0 / 0 | 0 / 0 | 0 / 0 |
| Undrivable (58,413,067) | 0 / 0 | 0 / 0 | 0 / 0 |
| Movable (1,460,386) | 0 / 0 | 0 / 0 | 0 / 0 |
| MyCar (29,993,221) | 0 / 0 | 0 / 0 | 0 / 0 |

The carrier is SegNet-inert at all three levels but not distortion-inert: PoseNet supplies the
entire positive distortion delta.

### HPAC probability model

| GT class (denominator) | L1 | L2 | L3 |
|---|---:|---:|---:|
| Road (27,407,372) | +18,477,299 / +0.674172591228 | +27,398,069 / +0.999660565778 | +27,398,069 / +0.999660565778 |
| Lane (690,754) | +684,894 / +0.991516516734 | +684,900 / +0.991525202894 | +684,900 / +0.991525202894 |
| Undrivable (58,413,067) | +57,957,352 / +0.992198406565 | +46,505 / +0.000796140356746 | -4,428 / -0.0000758049564492 |
| Movable (1,460,386) | +1,457,154 / +0.997786886481 | +1,457,154 / +0.997786886481 | +1,457,154 / +0.997786886481 |
| MyCar (29,993,221) | +542,349 / +0.0180823860165 | +28,034,519 / +0.934695176620 | +29,890,605 / +0.996578693565 |

### Fixed residual table

| GT class (denominator) | L1 | L2 | L3 |
|---|---:|---:|---:|
| Road (27,407,372) | +26,383,677 / +0.962648917963 | +26,333,997 / +0.960836266972 | +24,691,002 / +0.900889074662 |
| Lane (690,754) | +684,115 / +0.990388763583 | +683,225 / +0.989100316466 | +683,602 / +0.989646096874 |
| Undrivable (58,413,067) | +15,664,163 / +0.268161967938 | +12,334,624 / +0.211162067556 | +15,647,783 / +0.267881551229 |
| Movable (1,460,386) | +1,163,397 / +0.796636642641 | +1,185,315 / +0.811645003444 | +987,206 / +0.675989772567 |
| MyCar (29,993,221) | +4,074,359 / +0.135842662580 | +9,126,911 / +0.304299128126 | +6,338,024 / +0.211315216862 |

## Exchange ranking and waterfill

Ascending distortion damage per byte credited:

1. `carrier_l1_fixed_coder` — 0.000111718595939 S/B
2. `carrier_l2_fixed_coder` — 0.000124490064282 S/B
3. `carrier_l3_fixed_coder` — 0.000434765520411 S/B
4. `semantic_l3` — 0.000969624150645 S/B
5. `semantic_l2` — 0.00101065436257 S/B
6. `semantic_l1` — 0.00145208893915 S/B
7. `hpac_l3` — 0.0144971016809 S/B
8. `hpac_l2` — 0.0219518413780 S/B
9. `hpac_l1` — 0.0543256798219 S/B
10. `residual_l1`, `residual_l2`, `residual_l3` — no ZIP byte credit; not rankable per byte.

The independently best level per disjoint group is semantic L1, carrier L1, HPAC L2, and residual
L1. Every one is net-positive. Therefore:

- net-negative set: **empty**;
- independent byte credit admitted to waterfill: **0 B**;
- demand covered: **0/42,382 B = 0%**;
- unmeasured cross-group score composition: **none**.

## Payload custody and apparatus

The explicit-opt-in local tier was used:
`/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ap1_residue_purchase_scorer/`.
The launch preflight measured 445,197,041,664 B free locally, versus 8,986,886,144 B on Vertigo
and 11,870,142,464 B on APDataStore; both SSDs were below the 96 GiB fail-closed threshold. No
artifact was written to either SSD tier. The retained tree contains every control/candidate
archive and deterministic repeat, every physical section payload, every receiver tree, every
3,662,409,600-byte raw output, every 16-pair argmax/Pose6 chunk, every concatenated n600
argmax/Pose6 array, scorer receipts, and SHA-256/byte facts. The queue held one full-n600 candidate
at a time and checkpointed after every candidate. At finalization, `du -sh` reported the retained
tree as `61G` and `df -h` reported `403Gi` free locally; these are rounded disk-usage displays, not
an exact tree byte census. The shared decoded-token cache is bound by a
full 117,964,800-byte token payload SHA-256
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.

The source runtime and archive were never edited in place. Candidate receivers were copied to the
local receipt root and pinned to their own archive SHA/size. The canonical upstream snapshot was
copied to a clean retained local mirror because the source mirror contained generated AppleDouble
and Python-cache files; its compliance snapshot SHA-256 is
`fa7c4bf51d47a6140ec0f95275ebf86b0e6c3c1dc00caff03a417ee989645799`.

The first carrier materializer was rejected after its level-zero physical stream was 22,000 B
instead of the shipped 22,010 B. It refit CAP1 predictor metadata and applied DX2/RR5 in the wrong
inverse order. All its payloads and scorers remain retained; none of its byte credits enter this
table. The corrected rows freeze the shipped predictor and reproduce source CPR1, CAP1, and the
22,010 B carrier stream byte-for-byte. For all three levels, corrected and superseded rows have
identical raw, n600 argmax, and n600 Pose6 SHA-256 values; only the fixed-coder archive byte credit
changes. The supersession receipt is
`superseded_impure_carrier_v1/SUPERSESSION.json`.

One detached old queue monitor briefly duplicated the semantic-L1 post-score and then launched
semantic L2. It was interrupted by its exact session, all outputs were retained, and the surviving
results were byte-identical. The replacement queue restored full-n600 concurrency to one; see
`QUEUE_MONITOR_INCIDENT.json`. Two earlier advisory attempts also remain folded as terminal
evidence: the native-HPAC refusal and the dirty-upstream-snapshot refusal.

## RECALL EVIDENCE

I searched the full memo/receipt corpus, canonical-equations registry, canonical research index,
sub-0.15 DAG FEED blocks, design/SPEC surfaces, live hot state, and task-status ledger. Content
queries included `residue|semantic renderer|carrier|HPAC|fixed residual|purchase|load-bearing`,
`task-cell|quotient|#1187`, `coarsen|quantization|amplification`, and
`receiver-required|derive-at-decode`.

Beyond the charter seeds, the search found:

- no complete current-DX2 per-group purchase table in the bounded searched corpus;
- the live board had already fired JF1 to test the distinct field-plus-model diagonal, so AP1 must
  not call its fixed-model conclusion a result for joint refitting;
- the live NR1/#1187 consumer is a body-rebase route, but the existing NR1-K32 instance is
  distortion-dead and must not be retried as if it were the untested representation;
- MZ2's 38/38 semantic result is plumbing-only and cannot substitute for realized purchase;
- MST1's 78.71% native-render share is advisory evidence supporting real-path measurement, not a
  transferable distortion estimate;
- RI1/NI1 are measured whole-body ancestor negatives, not permission to interpolate any AP1 row;
- the canonical registry contributes the score/exchange arithmetic but no missing purchase row.

This changed the handoff in two ways: AP1's verdict is explicitly scoped to fixed-coder one-group
ladders, and the next body route excludes a retry of NR1-K32 while preserving JF1's already-owned
diagonal.

## What was and was not measured

Measured: exact DX2 census; level-zero coder identity; twelve complete n600 receiver-closed local
advisory rows; DALI-GT-pinned SegNet argmax and Pose6 outputs; per-class flips; complete archive
byte credits; recomputed distortion and net score deltas; and an empty measured waterfill.

Not measured: a contest-CPU or contest-CUDA candidate row; any shipping candidate; any
cross-group composition; any interpolated level; any alternate coder, keep-percent sweep, model
refit, sub-tensor allocation, structural-framing deletion, or new-body representation. AP1 does
not globally prove that no finer allocation exists; it closes the four registered whole-group
natural ladders on this exact body under fixed-coder isolation.

## Handoff disposition

- `FOLDED` — owner: AP1; consumer store: this memo plus the retained local receipt root; fire
  trigger: none. Do not rerun the current fixed-coder whole-group ladders.
- `FIRED` — owner: `ddm_jf1_joint_field_model_refit`; consumer store: JF1's chartered retained
  store; fire trigger: already satisfied by MAIN. Harvest its field-plus-model diagonal without
  treating it as an AP1 row.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN / `ddm_nr1_taskcell_body_rebase`; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_nr1_taskcell_body_rebase/retained/`; fire trigger: JF1 is
  terminal, the frozen JO endpoint has exact archive/render/component hashes and a fresh carrier
  fingerprint, a deterministic born-small one-member receiver and actual coder pass no-op,
  corruption, mutation, and parse-back tests, retention/resume/distinct stage checkpoints are
  wired, storage preflight admits a tier, and MAIN grants a unique scorer lane. This is a new
  jointly optimized body; it must not retry the distortion-dead NR1-K32 instance.

**Own-vehicle frontier: S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600], archive 976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674 — UNMOVED by AP1.**
