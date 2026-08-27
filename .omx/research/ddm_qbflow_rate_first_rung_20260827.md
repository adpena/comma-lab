# ddm_qbflow rate-first rung — initialized implicit-boundary-flow packet verdict — 2026-08-27

## Verdict first

| quantity | result | authority |
|---|---:|---|
| Preregistered complete `B_hat` | **122,797 B** | `[macOS-CPU scorer-free advisory, initialized-untrained]`; real reset-record coders plus complete shared ZIP framing |
| Exact initialized full-n600 `archive.zip` | **107,582 B** | `[macOS-CPU scorer-free advisory, initialized-untrained]`; exact retained bytes |
| Complete cap | 137,986 B | charter/no2 gate |
| Projected headroom | **15,189 B** | cap minus preregistered `B_hat` |
| Exact initialized headroom | **30,404 B** | cap minus exact full archive |
| Gate | **CLEAR — `RATE_SHAPE_EXISTS_INITIALIZED_UNTRAINED`** | both byte denominators pass |

The charter's falsifiable prediction was confirmed: this particular counted QBFLOW shape exists
below the initialized rate gate. This is not a trained candidate and supplies no evidence that it
can reach the distortion wall. No training, scorer, Metal, Modal, full-n600 scorer job, contest
evaluation, `d_seg`, `d_pose`, or `S` was run or claimed. The canonical pointer did not move.

Primary evidence:

- result: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/RESULT.json`, 44,739 B,
  SHA-256 `e52c16825632952b9f8fceb2946c452078b776a4f5d4c370f5d727a1a0c7a06b`;
- exact initialized archive:
  `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/stage_02_gate/retained/archive.zip`,
  107,582 B, SHA-256
  `0c833881a47c92d608fe54d7f14fbc481e580a5318b38f16efbfd4ef904da1c0`;
- committed schema/receiver/builder implementation: `aa58109f86`, with schema SHA-256
  `5405ccd499d14d28230874059e47d47f1f2818038519f1b27c97ed9377f132aa` and packet-module
  SHA-256 `cdf90d1a4d7d13001118f50a76692c04605f8e5ae9a7816c80f6e346160c7b9c`.

## Capacity derived backward from the distortion arithmetic

The frozen qbw2 field contains 1,625,624 four-neighbour interfaces across 600 pairs. At the
101,150-B reference rate and current pose envelope, no2's cap arithmetic permits
`d_seg <= 0.00044667138915998396`. Across `600 * 384 * 512 = 117,964,800` scorer cells, that is
52,691.501 allowed cell errors, 87.819 per pair, or only 3.241309% of the interface count. This is a
representation burden, not a QBFLOW distortion prediction.

The 84,910-B quotient allowance is 0.0522323 B per observed interface, so an explicit
interface-address object is not the selected form. The counted object instead uses a decoder-
generated continuous field with:

- ten signed outputs, one for every unordered pair among five classes;
- four dedicated along-tangent frequencies, 8/16/24/32 cycles, each with sine/cosine phase, giving
  `10 * 4 * 2 = 80` interface-frequency-phase degrees;
- 16 boundary-conditioning degrees, giving the derived flow width 96;
- four trainable step stages, including the input map;
- a 16-dimensional per-pair boundary latent and a separate 12-dimensional per-pair interior
  latent/pose head.

The resulting receiver has 79,513 learned scalars. Quantization is role-specific rather than
fp32-by-default: 62,272 scalars at 8 bits, 11,072 at 10 bits, 3,792 at 12 bits, and 2,377 at
16 bits. Boundary latents use 10 bits; interior latents use 12 bits. This is an initialized capacity
portrait. The canonical precision-waterfill law requires measured per-layer option costs and sound
error contributions; those do not exist before training, so no optimal-precision claim is made.

## Both required mechanism changes are real

This is not more width on the old flat paint template.

1. **Basis changed.** A nonlinear, step-reachable coordinate field emits signed inter-class
   interfaces. Its second flow pass conditions on coarse Road probability and tangents computed from
   the receiver's own decoded output. The dedicated 8/16/24/32 along-tangent comb reaches the
   measured lane-dash need near 25 cycles and therefore does not inherit the old basis's ceiling at
   8 cycles.
2. **Objective surface changed.** Boundary features condition both RGB frames directly, while a
   separate interior branch owns RGB interiors and pose12. The queued trained rung is joint
   realized-through-`R` Seg/Pose descent from birth. There is no fixed class paint and no post-hoc
   pose correction mechanism.

The NumPy receiver executes these branches on actual coordinate grids. Tests proved that changing
the interface, renderer, or pose tensors changes signed interfaces/class logits, RGB, or pose12,
respectively. The selected 32 receiver outputs were retained at 16x16 as real NPZ payloads; this
small grid is a functional receiver check only and carries no distortion authority.

## Rule-118 counted/free split

Counted inside the packet/archive are the architecture/config section, every quantized learned
tensor, quantizer metadata, all 600 boundary/interior latent records, integrity fields, packet
headers, and single-member ZIP framing. The retained initialized random values were counted even
though generic code could recreate a seed.

Free receiver computation is limited to generic normalized coordinates and perspective features,
the deterministic comb, the fixed ten-interface incidence matrix, Road probability/tangents derived
from decoded QBFLOW output, parsing, dequantization, and the NumPy forward. No GT table, source mask,
scorer weight, pair fact, learned basis table, or video-derived constant is embedded in free code.

## Real serialization receipts

Every section was fully encoded through Brotli quality 11, LZMA preset 9 extreme/XZ, and zlib level
9. Primary and repeat payloads for every candidate were retained; selection was smallest coded
payload with codec-ID tie-break. Brotli q11 won all sections:

| section | raw B | selected coded B | selected payload SHA-256 |
|---|---:|---:|---|
| config | 1,054 | 488 | `c9eea36beee58c63941afba078effc5972c06316c756b9b60470fcf9789fb8ff` |
| model | 87,854 | 80,629 | `c78069d07b1339ee60c30f97add5bfcfb1f83df650c20d2f0c91ca91efa09343` |
| latent metadata | 16 | 20 | `b4619e06c0dae760062a22b9cd92fc7abe05bde52a8dbb46aa272c0e8f286429` |
| full-n600 latents | 31,206 | 26,137 | `41884766bdd12a1f1bf898bcd95552c466bda9cf80c14c1c972d27a56cb94413` |

The full QBF1 packet is 107,474 B, SHA-256
`1375b87306c655dfa8c84c1238e3c62f5b392195b71787807288c5891a012ab0`.
The deterministic stored ZIP adds 108 B, yielding the 107,582-B exact archive above. The complete
primary/repeat packets and archives were byte-identical.

The parser reproduced the exact config, model tensor set/shapes/codes, latent metadata, and every
pair latent. One retained one-bit mutation in each of the four counted sections was independently
refused with a coded-CRC mismatch. The postrun audit re-parsed all sections, replayed all four
refusals, and recomputed both byte decisions.

## Preregistered n32 projection

The source is the 117,964,800-byte qbw2 decoded field, SHA-256
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`. NumPy PCG64 seed
20260827 selected the frozen, non-prefix stratified lineage:

`4, 31, 49, 52, 62, 90, 100, 113, 128, 148, 173, 179, 186, 187, 214, 236, 256, 260, 268, 278,
326, 328, 341, 352, 368, 382, 444, 456, 483, 508, 563, 573`.

Each independently reset selected latent record measured 69 B after its real coder race. The
Horvitz-Thompson variable projection is 41,400 B. The complete shared archive is 81,397 B, SHA-256
`fe77fb060343b2198336568e3f06b1eaf4870f0564178f524c6be210f6b58191`, so:

```text
B_hat = 81,397 + 41,400 = 122,797 B <= 137,986 B.
```

The materialized n32 archive itself is 82,899 B, SHA-256
`b52f3400646ea91be068e7c78c0e9a936cd229bb58db64d0a56554324dd00a29`; that sample artifact is
not substituted for the complete projection. The independently materialized full-n600 archive is
the stronger initialized-rate observation and also clears.

## Payload custody, recovery, and reproducibility

APDataStore passed the write-time storage preflight with 81,526,652,928 B free against
9,663,676,416 B required reserve-plus-work. All materialized model values, latent values, raw
sections, every three-coder candidate and repeat, every reset record candidate and repeat, packets,
archives, mutations, and receiver outputs remain under
`/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/`.

- custody manifest: 764 files / 5,413,165 logical B, SHA-256
  `e93cc52c865413bc91abea4aa4c9a2c4e85e8021bb04cd5287bcfe289e046abd`;
- postrun audit: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/POSTRUN_AUDIT.json`,
  SHA-256 `be23b4a709f7ee67d0255fa3a6374e29e3aac4e1a0d6b33d296e9059c987ebde`, `pass=true`;
- sealed fire order:
  `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/SEALED_TRAINING_FIRE_ORDER.json`,
  SHA-256 `7fa18f51d741b9f079b14c67d1c6560edd534bcaab6e2b8984f5bd0bd4b1ba8a`;
- a checkpoint resume returned the immutable result; the audit again recomputed 122,797 B and
  107,582 B.

Before the schema freeze, a developer-only parameter-count diagnostic instantiated the seeded
generic initialization without persisting it. No rate or scientific result used those bytes. The
exact deterministic initialization was recreated after the freeze and retained in full as
`initialized_float_params.npz` and `initialized_float_latents.npz`. The incident and recovery are
recorded at
`stage_01_initialize_quantize/PRE_RUN_DEVELOPER_DIAGNOSTIC_RECOVERY.json`, SHA-256
`6194acff7d47a26df8ca6ddab71be00548fe9963df2b87058632ea0625935ec2`. The schema was frozen before
the first experimental packet or byte measurement.

## RECALL EVIDENCE

The recall covered `.omx/research/` memos and receipts, `CANONICAL_RESEARCH_INDEX*`, the complete
`sub015_DAG_*` FEED corpus, design/SPEC files, task/live-state stores, source/receiver code, and
`.venv/bin/python tools/list_canonical_equations.py --json`. Content queries included:

- `QBFLOW|implicit boundary flow|quotient-born|signed interfaces|pose-separated interiors`;
- `along-tangent|3.2|fixed beta|hosc|FINER++|bias init`;
- `Road Lane|interface determinism|per-class crop|1.1604128`;
- `precision waterfill|per-layer sensitivity|mixed precision`;
- `QBFLOW|qbw2|no2` across the task ledger and `main_hot_state.md`.

The no2 and qbw2 seeds remained the only direct prior QBFLOW specifications/receipts found in those
scopes; no earlier receiver-closed QBFLOW packet was found. Beyond those seeds, recall changed the
build in four ways:

1. DAG #497 measured the old directional bank's along-tangent ceiling at 8 cycles against a lane-
   dash need near 25, a 3.2x deficit. This changed the basis from a generic directional field to a
   dedicated 8/16/24/32 comb.
2. The hosc receipts showed fixed-high-beta saturation/divergence while annealed/step-reachable
   forms remained trainable; FINER++ bias initialization is a live future lesson. This excluded a
   fixed-beta periodic output and kept a trainable finite-slope step map. No FINER++ benefit is
   claimed in this initialized rate rung.
3. The canonical precision-waterfill equation says its bit formula is only an initializer until
   measured layer costs and sound error contributions exist. This changed the plan from a false
   “optimal mixed precision” claim to named role precisions plus mandatory measured retriage and
   full re-encoding after training.
4. `ddm_rl1_roadlane_interface_price_20260803.md` measured an n32 per-class Lane-mask crop at
   1.1604 B/Road-Lane flip under Brotli q11, but explicitly did not establish an n600 or temporal
   verdict. Together with qbw2's 93.2331% Road-touch observation, this supported using Road as free
   receiver-derived conditioning while forbidding transfer of any explicit crop price or source
   mask into the packet.

The live-state/ledger search confirmed that this arm owned only the scorer-free rate gate and that
the qbw2 explicit family was already closed at its measured scope. It did not authorize training or
an n600 scorer dispatch.

## Boundaries

- **MEASURED:** initialized role-quantized weights/latents; every retained real coder output; n32
  reset-record HT projection; exact initialized full-n600 packet/archive bytes; deterministic
  receiver execution; parse-back; primary/repeat identity; four counted-section mutation refusals;
  storage/custody/resume audit. Axis:
  `[macOS-CPU scorer-free advisory, initialized-untrained]`.
- **DERIVED, not measured:** the 52,691.501-cell error ceiling, 3.241309% interface burden,
  84,910-B quotient allowance per interface, width-96 capacity portrait, and role precision choice.
- **NOT MEASURED:** training stability; trained packet bytes; full-resolution RGB; round trip through
  `R`; SegNet/PoseNet; `d_seg`; `d_pose`; `S`; decoder wall time; contest CPU/CUDA behavior; whether
  QBFLOW beats a same-budget QBW1 control.
- **NOT CLOSED:** QBFLOW distortion or the continuous implicit-flow family. The clear rate gate only
  buys the right for this formulation to confront the wall where born-small, NR1, W72/W96, and v15
  failed.
- **POINTER:** unmoved. A 107,582-B initialized random archive is not a contest candidate.

## Sealed training fire order

Disposition is `QUEUED-WITH-A-FIRE-ORDER`, not fired. Owner is `MAIN QBFLOW joint-training owner`;
consumer store is `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/`. The trigger is:
MAIN consumes this committed rate verdict, confirms no duplicate active lane and no full-n600
scorer job, lands and reviews a real scorer-in-loop QBFLOW trainer consuming this exact packet ABI,
and passes a live at-most-116-GiB memory/storage preflight.

The sealed stages are:

1. stage 03: joint realized-through-`R` Seg interface/RGB plus Pose6 descent from birth, chunk at
   most 30 pairs, with a distinct atomic EMA checkpoint at stage end plus periodic saves;
2. stage 04: measure receiver/scorer sensitivity by role, choose real precision options, retain a
   distinct EMA/optimizer/resume checkpoint, and re-encode every checkpoint/archive;
3. stage 05: compare the same seeded n32 QBFLOW and discrete QBW1 control at the identical complete
   serialized byte budget, retaining frames before scoring plus logits/argmax/Pose6 and archives.

Admission remains complete archive at most 137,986 B, `d_pose_hat <= 0.000125`, and `S_hat < 0.12`.
Initialized rate does not transfer to trained rate; no n600 or contest dispatch is authorized now.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — scorer-in-loop QBFLOW stages 03–05.** Owner: `MAIN QBFLOW joint-training owner`. Consumer store: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/`. Fire trigger: MAIN consumes committed verdict `aa58109f86` plus this memo, confirms no duplicate active lane and no full-n600 scorer job, lands/reviews a real trainer for the frozen QBF1 ABI, and passes the live at-most-116-GiB preflight; then run chunks of at most 30 with distinct periodic/stage EMA checkpoints, trained re-encoding, and the same-budget QBW1 control.

## LIVE-HYPOTHESES

- A decoder-generated continuous field can preserve the rate clearance after training because its
  81,397-B shared initialized object and 26,137-B full latent section leave 30,404 B of exact
  initialized headroom. Plausibility only: learned entropy can rise and every trained checkpoint
  must be re-encoded.
- The dedicated along-tangent comb can express lane-dash births that the old ceiling-at-8 basis
  missed, because the bank directly includes 24 and 32 cycles around the measured need near 25.
- Decoder-self-conditioned Road features can reduce interface-search burden without shipping a
  mask, because qbw2 measured 93.2331% Road-touch while rule 118 permits generic computation from
  decoded state.
- Separating boundary flow from interior/pose can escape the fixed-paint and post-hoc-pose walls,
  because future training can shape RGB boundary evidence and pose-bearing interiors jointly from
  birth; this remains the decisive untested distortion hypothesis.
- Measured post-training precision waterfill may recover trained-rate growth, because terminal
  interface/pose maps and threshold parameters need not share the bulk hidden precision; there is
  no sensitivity evidence yet.

## DEAD-ENDS

- Another coder/header/dictionary pass over explicit qbw2 cracks/masks: closed at that measured
  formulation; its best full-n600 floor was 188,860 B, above this arm's complete cap.
- Shipping an explicit Road/Lane mask or source-derived tangent table as free conditioning:
  forbidden by rule 118; Road/tangent conditioning must be recomputed from receiver output.
- Reusing the old flat fixed-paint objective with more width: excluded because v14/v15 and the
  predecessor carriers failed on realization/distortion; QBFLOW changes both basis and objective.
- A fixed-high-beta hosc output: closed as a basis choice for this formulation by measured
  saturation/divergence; the receiver uses finite-slope step stages and a dedicated comb instead.
- Calling the initialized role-bit assignment an optimal precision waterfill: closed by missing
  measured per-layer sensitivities/cost bounds; that claim cannot exist before the trained rung.
- Treating 107,582 B as a scored archive or transferring initialized bytes to trained bytes:
  forbidden; no video witness, scorer measurement, or score exists at this rung.

## Own-vehicle frontier

GB1 remains **S 0.14811799921260607 @ 180,215 B** (`archive.zip` SHA-256
`ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`)
**[contest-CUDA T4 n600]**; this QBFLOW rate-only arm did not move it.
