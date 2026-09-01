---
schema: ddm_qx4_decodable_conditioning_reprice_memo.v1
date: 2026-09-01
arm: ddm_qx4_decodable_conditioning_reprice
status: FORMULATION-CLOSED
axis: "[scorer-free exact rate and receiver-conditioned parse-back measurement]"
score_claim: false
pointer_moved: false
selection_mode: full_n600
custody: /Volumes/APDataStore/pact/ddm_qx4
---

# QX4 decodable-conditioning reprice — all six forms are over; distance rank is cheapest at 33,435 B versus the 24,093 B cap

## Conclusion

**Verdict: `FORMULATION-CLOSED`.** Reconditioning QX2 on the partition field
that the QX1/QBT receiver actually produces does not clear the event-section
envelope. The primary boundary-transition enumerative subset rises from QX2's
22,661 B to **34,083 B**, **9,990 B over** the 24,093 B section cap. Per the
preregistered falsifier, I then reconditioned and real-coded all five QX2
siblings. They reorder: decoded-state distance rank is cheapest at **33,435 B
Brotli q11**, but remains **9,342 B over** the cap. Its exact complete archive
is **147,327 B**, 9,342 B above the largest legal 137,985-byte archive.

All measurements are `[scorer-free exact rate and receiver-conditioned
parse-back measurement]`, full n600: 600 pairs, 117,964,800 conditioning sites,
and all 17,926 original `PartitionEvent` tuples. No scorer, contest evaluator,
Metal, MPS, Modal, or remote process ran.

## Stage 0 — the only conditioning field

The only candidate/rank conditioning source was QX3's proven QX1/QBT receiver
path. QX4 freshly decoded the seven-section QX1 core and reproduced the retained
QBT native field exactly:

- field: 117,964,800 B;
- SHA-256:
  `afeb8c94d5181b03992aefad1daef49ee7aaf1f768d11aa5964dacbfa1e22dbd`;
- fresh QX1 decode versus QBZ1's retained quantized-packet native field:
  **0 / 117,964,800 mismatches**;
- encoder-only QX2 C1 baseline consulted by QX4's coder or receiver: **no**.

The event object cannot be silently relabeled to the QBT field. At the 17,926
event sites, QBT equals the original event baseline at only **8,307** sites,
already equals the event target at **9,177** sites, and is a third class at
**442** sites. QX4 therefore uses QBT only for candidate identity/order and
retains each event's original baseline class as counted content. This preserves
the exact source tuple `(pair,row,col,target_class,baseline_class)` rather than
claiming a cheaper cross-predictor event object.

The geometry explains the price change. QX2's encoder-only C1 field put
17,691 events on its radius-0 boundary and left 235 residuals. The decodable QBT
field puts only **6,669** events on its radius-0 boundary and leaves **11,257**
distance-rank residuals. The approximate field's 98.58% whole-raster agreement
with C1 therefore did not transfer to the event-conditioned alphabet.

## Stage 1 — all decodable-conditioned forms

Every row uses the same bit-pinned QBT field, reconstructs all 17,926 original
event tuples exactly, races Brotli q11, LZMA-9-extreme, and zlib-9, and retains
all three payloads plus deterministic repeats.

| form | QX2 payload on exact C1 | QX4 payload on decoded QBT | change | exact archive | delta vs cap |
|---|---:|---:|---:|---:|---:|
| boundary-transition enumerative subset | 22,661 B | **34,083 B** Brotli q11 | +11,422 B | 147,975 B | +9,990 B |
| boundary bitmap, radius 0 | 27,848 B | **35,275 B** Brotli q11 | +7,427 B | 149,167 B | +11,182 B |
| boundary bitmap, radius 1 | 30,192 B | **36,513 B** Brotli q11 | +6,321 B | 150,405 B | +12,420 B |
| boundary bitmap, radius 2 | 31,804 B | **37,439 B** Brotli q11 | +5,635 B | 151,331 B | +13,346 B |
| boundary bitmap, radius 4 | 34,028 B | **38,778 B** Brotli q11 | +4,750 B | 152,670 B | +14,685 B |
| decoded-state distance ranks | 30,216 B | **33,435 B** Brotli q11 | +3,219 B | **147,327 B** | **+9,342 B** |

The primary's 47,275-byte raw representation contains 58,243 enumerative bits
(7,281 B), a 5,420-byte transition-count stream, a 2,501-byte original-baseline
stream for the 6,669 boundary events, and a 31,965-byte residual stream for the
11,257 off-boundary events. The sibling order changes because the QBT field's
boundary is a poor event-site alphabet; a global distance order pays less than
splitting the population into a small boundary subset and a large residual.

No scope or mechanism reduction was used. In particular, the 9,177 QBT-relative
no-ops were not deleted: they remain part of the exact 17,926-event source
object. Omitting them or replacing original baseline classes with QBT classes
would price a different object and violate the charter's exact-reconstruction
gate.

## Stage 2 — complete receiver proof and terminal verdict

The selected packet keeps QX1 core sections 1-7 byte-for-byte and adds the
33,435-byte distance-rank stream under one 48-byte QXE section header. Two
byte-identical archives independently ran the receiver path:

1. parse and integrity-check all eight QXE sections;
2. reconstruct the counted QBT model and fresh n600 conditioning field;
3. derive every rank order from that field;
4. decode all **17,926 / 17,926** original event tuples exactly; and
5. overwrite each decoded event site with its retained target class.

Primary and repeat receiver outputs are bit-identical: 117,964,800 B, SHA-256
`9079929d004cc9638a80159d61371c2982c198f0eb2b19eac4084da981ababc7`.
Application changes 8,749 sites and leaves 9,177 QBT-relative target no-ops.
This output is a receiver artifact, **not** a SegNet/PoseNet or score result.

`verdict_scope=FORMULATION: QX2's primary boundary-transition enumerative form
plus all five named sibling forms, each reconditioned on the bit-pinned QX3
receiver field and raced under the three real coders. This is not a family
theorem against a structurally new decoder-native site grammar or a changed
core representation.`

The conditional scorer-realization step is **FOLDED**, not queued: no form
clears the byte prerequisite, so QX4 does not consume the sole n600 scorer slot.

## Prior negatives — named or folded

- **QX3 exact-baseline bridge, 510,404 B / +486,311 B:** remains dead. QX4
  transmits no full-field bridge and directly conditions on the decoded field.
- **QX2's five over-cap siblings:** all were reopened only for the mandatory
  same-field reprice; every one remains over, now by 11,182-14,685 B.
- **Pincer, 352,525 B exact-address side information:** remains folded. QX4
  sends no literal `(pair,row,col)` tuple stream; ranks are derived against the
  receiver-produced field.
- **m143 cross-regime transfer:** honored. No QX2 price or ordering was carried
  over; the full six-form race shows the ordering actually changes.
- **Known-site G3 ideal class-identity floor, 2,724.8733 B:** not used as a
  closing price because it excludes site grammar, headers, finite coder, and
  receiver costs.

## RECALL EVIDENCE

Recall covered the full `.omx/research/` memo/receipt surface, canonical
equations, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design/SPEC
documents, task/queue/final-message ledgers, source, and both SSD roots. Queries
included `decoded QBT field|QBT native field|approximate core field`,
`boundary-transition enumerative|enumerative subset`, `receiver-produced
field|receiver-derived partition`, `17,926|17926`, `m143|cross-regime
transfer`, `predictor-bound residual`, and `argmax cell identity`. The equation
command was `.venv/bin/python tools/list_canonical_equations.py --json`, filtered
for `argmax|cell identity|procedural|residual|conditional|receiver|archive|rate`.

Findings beyond the charter seeds changed the implementation:

1. `codex_findings_c0b_exact_residual_composition_20260726_codex.md` and its
   premise-falsification receipt identify the source baseline as part of the
   residual ABI and forbid silent cross-predictor transplantation. QX4 therefore
   retains original baseline classes while using QBT only as context; this is
   why the 9,177 QBT-relative no-ops remain counted.
2. `g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.md` prices only
   known-site class identity and explicitly excludes site locations, candidate
   transport, headers, receiver, and realization. QX4 therefore built and
   retained finite site grammars instead of citing the 2.7 KB ideal figure.
3. `ddm_qx1_section_census_20260831.md` re-derived the exact 24,093-byte section
   cap and establishes the explicit 39,836-byte S2 packet as a negative control,
   not a valid QX1 section. QX4 kept the event packet encoder-side only and put
   only the newly coded section in the archive.
4. The m143 examples, including
   `ddm_bz2d_distortion_verdict_20260830.md`, require constants and rankings to
   be re-derived on the consuming object. That changed the plan from a primary
   price plus inherited sibling ordering into the mandatory full six-form race.
5. Queue/final-message custody showed QX2 and QX3's full source landings are
   verified fallback bundles after managed Git object-write denial, while only
   their final-message custody records are currently at shared HEAD. QX4's
   intended landing bundle therefore includes those exact, SHA-pinned source
   dependencies rather than leaving an untracked runtime dependency.

No QX4-specific alternative or cheaper decodable-conditioned representation
was found beyond the same-day QX2/QX3 seeds in the searched index, DAG, design,
task, and custody scopes.

## Custody and reproducibility

- Result: `/Volumes/APDataStore/pact/ddm_qx4/RESULT.json`, 66,841 B, SHA-256
  `a147b3d08c7f485a323be3d41388f72e095ef8d3989e8a813993f0f36679d8bf`.
- Run manifest: `/Volumes/APDataStore/pact/ddm_qx4/RUN_MANIFEST.json`, SHA-256
  `63b36bfa55b624ca2c6c163725d6d15598c9c380df57cc84ede60e37c9b356df`.
- Runner: `experiments/ddm_qx4_decodable_conditioning_reprice.py`, SHA-256
  `0defd490a952de0e0d7b3bc0740b0586f040589bcca93faa22930dfc443c626d`.
- Selected payload: 33,435 B, SHA-256
  `6637733eafee7d57510a3d3738d0222cddcc30fb4e034608f8058521ae9767b3`.
- Selected archive: 147,327 B, SHA-256
  `19809991d47be7856e2aed5570bbcbecaa43e4a2252b7bad526786e27c55cf19`.
- Command: `.venv/bin/python
  experiments/ddm_qx4_decodable_conditioning_reprice.py --resume-from
  /Volumes/APDataStore/pact/ddm_qx4`.
- Independent verification rehashed 85 QX4-owned files totaling 361,270,449
  logical bytes, re-decoded all six retained raw forms, and matched all 17,926
  source tuples. All raw/coded/repeat/packet/archive/conditioning/receiver bytes
  remain retained; no cleanup fired.

## Authority boundaries

- **Measured:** fresh decoded QBT field identity; all event/context relations;
  six complete raw representations; eighteen real-coder outcomes plus repeats;
  six exact archives; full receiver parse-back; all 17,926 original event
  tuples; event-application output; deterministic archive/output repeats.
- **Not measured:** SegNet or PoseNet distortion, score components, contest
  score, CPU/CUDA parity, contest runtime, or QX1's pose corridor. No scorer or
  evaluator ran.
- The selected 147,327-byte archive is a scorer-free representation receipt,
  not a promotable candidate. It exceeds the required archive envelope and
  does not move the pointer.

**Own-vehicle/effective frontier remains:** afr1 — S =
**0.14797617125559104 @ 180,002 B `[contest-CUDA T4, n600]`**, archive SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`;
QX4 made no score measurement and did not move the pointer.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN-assigned new-representation owner;
  consumer store: `/Volumes/APDataStore/pact/ddm_qx4/RESULT.json`; fire trigger:
  a structurally new decoder-native site grammar, distinct from all six closed
  QX2 forms, is specified with a concrete mechanism capable of removing at
  least 9,343 payload bytes while preserving all 17,926 original event tuples;
  then run one full-n600 payload-retaining real-coder race on the same pinned
  QBT field. Do not rerun any QX4 form unchanged.

## LIVE-HYPOTHESES

- A decoder-native event grammar may code target overwrites rather than the
  historical C1 syndrome ABI. This is plausible because 9,177 retained events
  are already target-equal on QBT and only 8,749 sites change the actual QBT
  output, but it is a **new object**: it must first prove that dropping original
  C1 baseline identity is semantically admissible to the QX1 consumer, then
  reprice and score from scratch.
- A changed QBT core trained jointly with event-site candidate entropy may move
  more of the 17,926 sites onto a decoder-native boundary. This is plausible
  because the present price explosion is localized to 11,257 off-boundary
  residuals, but it is a new core representation whose model/latent bytes and
  distortion cannot inherit QX4 numbers.

## DEAD-ENDS

- QX2's 22,661-byte price does not transfer to the decoded QBT field: the exact
  same enumerative mechanism costs 34,083 B there.
- All six QX2 forms are closed at formulation scope on this pinned QBT field;
  the cheapest is 33,435 B, still 9,342 B over cap. Do not rerun them unchanged.
- Boundary dilation is monotone-worse here: radius 0/1/2/4 payloads are
  35,275/36,513/37,439/38,778 B.
- Dropping QBT-relative no-ops or substituting QBT classes for original event
  baseline classes is closed for QX4: either change would price a different
  event object and fail exact reconstruction.
- QX3's 510,404-byte exact-baseline bridge remains closed, as do explicit
  raster-address carriage and the 352,525-byte pincer.
- Scorer realization is folded for this formulation because the byte gate
  fails before scorer admission; the 147,327-byte archive is not promotable and
  afr1 remains unchanged.
