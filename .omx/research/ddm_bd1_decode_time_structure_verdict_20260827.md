# DDM BD1 decode-time structure first rung — verdict

**Disposition:** **B+D CLOSES at FORMULATION scope.** On a seeded-stratified random
`n=32` sample, an optimistic per-pair oracle over thirteen scorer-free
token-topology reaction--diffusion rungs reduced `d_seg` only from
`0.016702016194661457` to `0.014959017435709635`: **1.116518265x recovery**, not
the pre-registered `2x`. It failed both the matched gate
`d_seg <= 0.008351008097330729` and the charter-fixed gate
`d_seg <= 0.006158325212247259`.

Axis: `[macOS-CPU frozen-scorer advisory, seeded-stratified random n32]`.
`score_claim=false`; no full-n600 scorer, PoseNet pass, Modal job, GPU run, exact
contest evaluation, or pointer move occurred.

Within the fb2 route table, all three reference routes were already closed and
this was the only arithmetically open pre-derived B+D cell. This negative therefore
leaves that route table **EMPTY**. The honest successor is a fresh object derivation,
not a second rung on this body.

## The bar and the measured answer

At a 119--138 KB body, rate contributes about `0.079--0.092 S`, leaving only
`0.028--0.041 S` total distortion under sub-0.12. The cited born-small floor thus
needed roughly a 30x reduction. The charter deliberately set a much weaker first
rung: recover at least 2x before buying more decode-side work.

| quantity | value | status |
|---|---:|---|
| exact inherited body | 101,150 B, SHA-256 `5743f0ac7e8881e970ef8ba53c4bee3fd2a7a6157d2a50d381fd609ae624fea6` | byte-closed input |
| fresh BD1 baseline | `d_seg=0.016702016194661457`, Seg `S=1.6702016194661458` | measured n32 |
| per-pair oracle selection | `d_seg=0.014959017435709635`, Seg `S=1.4959017435709634` | measured n32 |
| recovery | `1.1165182650827719x` | below 2x |
| matched 2x gate | `d_seg <= 0.008351008097330729` | FAIL |
| fixed charter 2x gate | `d_seg <= 0.006158325212247259` | FAIL |
| exact body + constraint archive | 101,486 B, SHA-256 `f06dea2147104444dfab9f5927e700d8e59d037b84f597b35c7df3ed9ac435a8` | byte-closed probe packet |
| exact archive rate term | `0.06757536171655668 S` | derived from 101,486 B |
| perfect-pose subset expression | `1.56347710528752 S` | n32-derived, **not** an n600 score |

The earlier BS3 receipt needs a denominator correction. Its stage-40 file names
`sealed_sample_pairs=32`, but the aggregate explicitly says
`measured_pair_count=20` and contains 20 measured rows. Its
`d_seg=0.01188074757810682` and perfect-pose floor
`1.2316650424494517 S` are therefore n20 aggregates, despite the surrounding
memo/charter calling them n32. BD1 did not transfer that headline. It measured a
new stratified-random n32 identity control, and tested both that matched gate and
the fixed charter gate. Both fail.

## Legal mechanism and receiver boundary

The exact body is parsed through `experiments/ddm_rb1_born_small_receiver.py`.
The receiver reconstructs the 117,964,800-byte semantic token field, decodes the
30,856 counted semantic-renderer bytes inside the body, and renders the starting
frame-1 RGB. Every selected body-derived starting frame was byte-identical to the
pinned BO2 raw validation frame before a transform ran. The 3.66 GB BO2 raw was a
validation oracle only; it was not an input to the solve.

The added generic algorithm uses only:

- same-token four-neighbour diffusion or sharpening;
- connected-component mean pull within each token class;
- a different-token boundary reaction;
- one counted four-bit rung id per pair.

SegNet ran only after each candidate frame had been retained. It selected the rung
ids at measurement/encode time; it is not a decoder dependency. The candidate
archive contains the exact nested body plus a real-coded rung stream. It contains
no SegNet/PoseNet weights, GT mask, logits, or scorer cache. The outer operation is
therefore called **selection**, not a solver; the deterministic inner transform is
the only decode-side solve.

This is an optimistic ceiling: the encoder gets per-pair scorer selection over all
thirteen rungs. Even that stronger surface misses 2x.

## Mechanism table

All thirteen rungs share one n600 four-bit selection stream. Its 300-byte nibble
body plus 72-byte identity/hash header is 372 B raw; zlib-9 wins at 152 B. The
outer archive adds exactly 336 B to the inherited body: 152 B coded constraints,
84 B BD1 section header, and 100 B deterministic stored-ZIP framing. `ceiling_only`
is false for every row below. Times are the added structure transform only; full
integrated inflate wall clock was not measured.

| id | rung | mean d_seg | Seg S | recovery vs identity | mean sec/pair | projected n600 sec |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `identity` | 0.016702016 | 1.670202 | 1.0000x | 0.001880 | 1.13 |
| 1 | `diffuse_2` | 0.016523043 | 1.652304 | 1.0108x | 0.018636 | 11.18 |
| 2 | `diffuse_8` | 0.016217868 | 1.621787 | 1.0299x | 0.041882 | 25.13 |
| 3 | `diffuse_32` | 0.015876293 | 1.587629 | 1.0520x | 0.139217 | 83.53 |
| 4 | `sharpen_1` | 0.016796430 | 1.679643 | 0.9944x | 0.012396 | 7.44 |
| 5 | `sharpen_4` | 0.016966502 | 1.696650 | 0.9844x | 0.024603 | 14.76 |
| 6 | `component_pull_0125` | 0.016051292 | 1.605129 | 1.0405x | 0.028370 | 17.02 |
| 7 | `component_pull_0500` | 0.016093095 | 1.609310 | 1.0378x | 0.028358 | 17.01 |
| 8 | `boundary_push_0125` | 0.016690254 | 1.669025 | 1.0007x | 0.013203 | 7.92 |
| 9 | `boundary_push_0500` | 0.016704877 | 1.670488 | 0.9998x | 0.013184 | 7.91 |
| 10 | `smooth_joint` | 0.015676816 | 1.567682 | 1.0654x | 0.065940 | 39.56 |
| 11 | `sharp_joint` | 0.016166051 | 1.616605 | 1.0332x | 0.049704 | 29.82 |
| 12 | `strong_joint` | 0.023437182 | 2.343718 | 0.7126x | 0.164941 | 98.96 |
| oracle | best rung per sampled pair | **0.014959017** | **1.495902** | **1.1165x** | selected mean | **23.22** |

The best single fixed rung is `smooth_joint` at only 1.0654x recovery. The
per-pair oracle selects a non-identity rung for 28/32 sampled pairs and still
reaches only 1.1165x. This is not a hyperparameter-default miss.

## Per-pair rows

Selection: seed `20260827`, stratified random without replacement over ten
60-pair temporal blocks crossed with dominant token class. The observed dominant
class was class 2 in every temporal block; quotas were four samples in the first
two blocks and three in each remaining block. This is not a prefix.

| pair | baseline d_seg | selected d_seg | rung | pair recovery |
|---:|---:|---:|---|---:|
| 16 | 0.023503621 | 0.013417562 | `component_pull_0500` | 1.751706x |
| 21 | 0.010279338 | 0.010238647 | `sharp_joint` | 1.003974x |
| 32 | 0.019337972 | 0.014155070 | `component_pull_0500` | 1.366152x |
| 39 | 0.012135824 | 0.011479696 | `smooth_joint` | 1.057156x |
| 75 | 0.011337280 | 0.011240641 | `component_pull_0125` | 1.008597x |
| 78 | 0.012624105 | 0.012619019 | `boundary_push_0125` | 1.000403x |
| 98 | 0.010187785 | 0.010187785 | `identity` | 1.000000x |
| 102 | 0.013422648 | 0.013300578 | `smooth_joint` | 1.009178x |
| 126 | 0.007934570 | 0.007843018 | `diffuse_2` | 1.011673x |
| 155 | 0.007308960 | 0.007227580 | `component_pull_0125` | 1.011260x |
| 158 | 0.011306763 | 0.011250814 | `component_pull_0125` | 1.004973x |
| 216 | 0.005544027 | 0.005482992 | `component_pull_0125` | 1.011132x |
| 225 | 0.007273356 | 0.007253011 | `diffuse_2` | 1.002805x |
| 227 | 0.005559285 | 0.005518595 | `sharp_joint` | 1.007373x |
| 261 | 0.009246826 | 0.009246826 | `identity` | 1.000000x |
| 263 | 0.007446289 | 0.007324219 | `diffuse_8` | 1.016667x |
| 267 | 0.007090251 | 0.007080078 | `sharpen_1` | 1.001437x |
| 319 | 0.057805379 | 0.050776164 | `diffuse_32` | 1.138435x |
| 327 | 0.053314209 | 0.048675537 | `diffuse_32` | 1.095298x |
| 335 | 0.006469727 | 0.006444295 | `diffuse_2` | 1.003946x |
| 393 | 0.010025024 | 0.010019938 | `boundary_push_0125` | 1.000508x |
| 410 | 0.008850098 | 0.008834839 | `boundary_push_0500` | 1.001727x |
| 419 | 0.010304769 | 0.010304769 | `identity` | 1.000000x |
| 439 | 0.023437500 | 0.014953613 | `smooth_joint` | 1.567347x |
| 454 | 0.039820353 | 0.028610229 | `component_pull_0500` | 1.391822x |
| 470 | 0.028071086 | 0.028040568 | `component_pull_0125` | 1.001088x |
| 483 | 0.008214315 | 0.008199056 | `boundary_push_0125` | 1.001861x |
| 518 | 0.021581014 | 0.021581014 | `identity` | 1.000000x |
| 531 | 0.009679159 | 0.009480794 | `smooth_joint` | 1.020923x |
| 548 | 0.012125651 | 0.012069702 | `boundary_push_0500` | 1.004635x |
| 555 | 0.015116374 | 0.015106201 | `boundary_push_0500` | 1.000673x |
| 575 | 0.048110962 | 0.040725708 | `strong_joint` | 1.181341x |

## RECALL EVIDENCE

The recall searched the full `.omx/research/` corpus by content for
`decode-time`, `constraint shipping`, `bits-back`, `hash sieve`, `scorer-free`,
`task-cell`, `score quotient`, `receiver-close`, `worldsheet`, `topology`,
`temporal advection`, `xi`, `persistence`, `K32`, `NI1`, `NR1`, `DB1`, `QS5`,
and `born-small`. It also searched `CANONICAL_RESEARCH_INDEX*`, all
`sub015_DAG_*` FEED blocks, design/SPEC documents, the canonical task-status and
live hot-state surfaces, and the actual receiver/evaluator code. The canonical
equation registry was listed with
`.venv/bin/python tools/list_canonical_equations.py --json` and filtered for
`quotient`, `task-cell`, `receiver`, `rate`, `temporal`, `topology`,
`constraint`, and `bits-back`.

Beyond the charter's seeds, recall changed the plan in five ways:

1. The actual BS3 stage-40 receipt is n20, not n32. This forced a new n32
   identity control rather than transferring the 1.2317-S headline.
2. The append-only NI1 erratum/retraction records the authority K32 quotient
   failure: `[contest-CUDA T4 n600] d_seg=0.07583781`,
   `d_pose=40.53479004`, recomputed `S=27.7984`. That closed a tempting task-cell
   shortcut and kept this probe on the exact body rather than relabeling NI1.
3. DC1 found no compact scorer-free task-cell certificate; its explicit object
   is about 44.244 MB, its five-group hash sieve is aggregate-negative, and a
   deterministic delta posterior refunds zero bits. DB1 later showed that
   fixed-grid support relocation and group-uniform widths remain larger than the
   token member. This changed the probe from an asserted cell-membership solver
   into a tiny, explicit, real-coded transform-id packet whose effect can be
   parsed without a scorer.
4. Existing worldsheet/row-boundary work is already negative at its measured
   formulation: WS0's best receiver-readable lossless row is 269,921 B, while
   OR1's current-object row-boundary packet is 140,377 B and temporal-XOR is
   259,617 B. Those are representations, not free post-render structure, so the
   probe did not retry them.
5. `temporal_advection_stratified_20260715.md` found xi's marginal value over
   persistence approximately zero at the two-frame gap. The body also contains
   no newly derived compact pose-to-token certificate. This removed raster
   xi-advection from the first rung and left token-conditioned spatial topology
   as the nonduplicate legal mechanism.

The bounded search did not find, in these scopes, a complete receiver-runnable
compact task-cell certificate that had not already been measured or queued.
That is scoped absence, not a global nonexistence claim.

## Custody and independent verification

Durable root:
`/Volumes/APDataStore/pact/ddm_bd1_decode_time_structure/`.

- Preflight observed 85,886,500,864 B free; required free space was
  12,114,168,064 B including an 8 GiB reserve. At memo-write audit,
  `df -k` reported 80,083,072 KiB free.
- Store: 2,165 non-AppleDouble files, 3,789,312 KiB allocated at audit time.
- `RESULT.json`:  SHA-256
  `bf943e775080367b62d06196e676439d47d6f18691b327bc6a4f4f4dff3ed09f`.
- Solve receipt: 234,443 B, SHA-256
  `acf1a21fad757c4069a133ca4073f5e148496e12acb9dd3db8b09cf85714ede1`.
- Measurement receipt: 601,017 B, SHA-256
  `ea3a09d2e67d5d5c392b72b06c8132fc5df1c0df27ddf17f031b6514ae6cd45a`.
- Packet receipt: 3,933 B, SHA-256
  `951c8e5b44492a94d481ea3f7e44e9c4a9ebc4722cb87c6f9295006b99a9b704`.
- Runner at measurement: 42,455 B, SHA-256
  `ab14b0f65d022179b59a37253f6811172b0777fe31223848b7b9a0e859963483`.
- Git provenance before serializer:
  `8e78cbe2af467f2861fa5a8fa08fe3e1fb990156`; seed `20260827`.

Every one of the 416 candidate frames was retained before scoring, along with
every SegNet logit array, argmax, and target. All four constraint coder contenders
and deterministic repeats are retained. The exact archive and repeat are
byte-identical. Parse-back recovered the exact nested body and all 600 rung ids;
a one-bit constraint mutation was refused.

An independent post-run audit rehashed 1,291 unique retained payloads totaling
2,993,863,284 B, reconstructed all 416 `d_seg` values from retained argmax and
targets, recomputed every per-pair winner and both aggregates, rechecked all real
coder repeats, parsed the candidate archive, and recomputed the exact rate term.
It passed. Focused pytest (3 tests), Ruff, py_compile, and `git diff --check` also
passed.

## Boundaries and disposition

- **MEASURED:** one token-topology reaction--diffusion formulation on the exact
  body, n32 stratified random; 13 rungs, 416 retained scorer rows; real-coded
  constraint packet; structure-transform timing.
- **NOT MEASURED:** PoseNet, full-n600 distortion, complete integrated inflate
  wall clock, contest CPU/CUDA score, or any second representation/object.
- **CLOSED(FORMULATION):** exact BS3 body plus token-conditioned local
  reaction--diffusion/component/boundary transforms and a four-bit per-pair rung
  packet. The outer per-pair scorer oracle is part of the measured ceiling.
- **FOLDED:** every B+D second rung on this object. Consumer: this memo and the
  fb2 route table. Reason: 1.1165x is far below the 2x preregistration, before the
  30x end requirement.
- **QUEUED-WITH-A-FIRE-ORDER:** fresh object derivation from the measured laws.
  Owner: `MAIN frontier object-derivation router`. Consumer store:
  `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` plus a
  newly assigned retained SSD root. Fire trigger: MAIN consumes this B+D closure,
  confirms no duplicate active lane, and charters a representation that changes
  the 101,150-byte object itself with a pre-registered n32 serialized gate. No
  scorer/GPU job is implied by this queue row.

## LIVE-HYPOTHESES

- A **new body born in a task-cell/worldsheet quotient**, rather than a
  post-render transform of the exact token body, may expose structure the current
  renderer never learned. This remains plausible because the current packet
  spends counted bytes reproducing a fixed categorical field, while BD1 changed
  only RGB inside fixed decoded cells.
- A compact scorer-free certificate may become viable only when co-designed with
  a new grammar. It remains plausible because generic verification code is free,
  but DC1/DB1 show that attaching support/width questions to the current body is
  too expensive.
- The hard pairs 16, 32, 439, and 454 show 1.37--1.75x local recoveries. A new
  object that allocates representation capacity specifically to those structural
  regimes could outperform uniform local smoothing; this is plausible as a
  training/allocation clue, not transferable score credit.

## DEAD-ENDS

- A second local reaction--diffusion rung on the same exact body: closed at
  formulation scope because an optimistic thirteen-rung per-pair scorer oracle
  reaches only 1.1165x, below the 2x first gate.
- Stronger smoothing/sharpening as the missing optimum: closed within the tested
  formulation; the strongest joint row worsens aggregate `d_seg` to 0.023437182,
  and the best fixed row reaches only 1.0654x.
- Treating BO2 raw or scorer outputs as decoder state: refused; the legal receiver
  reconstruction was proved from counted body bytes and the raw was validation
  only.
- Retrying NI1/K32 as a free task-cell shortcut: closed by its measured
  `[contest-CUDA T4 n600]` distortion failure.
- Current-body hash-sieve, fixed-grid metadata relocation, or delta-posterior
  bits-back as the compact certificate: closed at their recorded formulation
  scopes by DC1/DB1 arithmetic and receiver evidence.
- Raster xi-advection at the two-frame gap as the missing generic prior: closed
  for this first rung by prior measured near-neutrality versus persistence and no
  new compact receiver certificate.

gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]
