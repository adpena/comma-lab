# ddm_mc35 MICRO35 union build — terminal gate failure, no fire order (2026-08-14)

## Verdict

`RFO1-MICRO35` was built as one real receiver object and retained. It reaches the
minimum Seg gain exactly, but it exceeds both the archive-byte cap and the
projected pose-debt cap. The instance disposition is
`TERMINAL_GATE_FAILURE_NO_FIRE_ORDER`; no Modal/T4 request was emitted, no lane
was claimed, and the canonical frontier pointer did not move.

| local admission surface | required | measured | result |
|---|---:|---:|---|
| Seg gain, base flips minus candidate flips | >=35 | 35 (34,970 -> 34,935) | PASS |
| archive delta vs cp135 | <=+29 B | +44 B (186,252 -> 186,296 B) | **FAIL by 15 B** |
| projected delta d_pose | <=5.9739759814e-10 | +5.263319929768801e-9 | **FAIL by 4.666e-9 (8.81x cap)** |
| receiver parse-back | exact | exact | PASS |

The score-form projection on this local advisory surface is
`delta S = +3.132265911888886e-7`, so even apart from the hard caps this object
does not justify an exact-evaluation fire order.

Axis and denominators: `[macOS-CPU advisory frozen CPU-torch SegNet/PoseNet;
eight changed pairs over n600] NON-PROMOTABLE`; 600 pairs, 117,964,800 Seg
pixels, and 3,600 Pose scalars. Selection is all eight changed pairs after one
composed receiver closure, not a prefix. This is a local admission recount, not
an exact contest score.

## Built object

The exact union contains the six QS2 token objects, RE1's admitted singleton at
pair 96, and the smallest distinct-pair sign-verified neighbor at pair 7:

| bank | proposal | pair | measured Seg gain |
|---|---|---:|---:|
| QS2 | `js6_0000_9fbf75d81c43` | 105 | +4 |
| QS2 | `js6_0072_f790b6493122` | 176 | +1 |
| QS2 | `js6_0006_92685b3e3e44` | 178 | +3 |
| QS2 | `js6_0004_06fc74e20d9e` | 517 | +12 |
| QS2 | `js6_0001_da319a6b65d0` | 523 | +14 |
| QS2 | `js6_0118_83f376603d6e` | 532 | -2 |
| RE1 | `ec1_0164_3a4e239de5b9` | 96 | +2 |
| RE1 | `ec1_0004_3bc2b69c706c` | 7 | +1 |

The built support census found zero overlapping pair indices and zero
overlapping token pixels between QS2 and RE1. QS2 contributes +32 flips and RE1
contributes +3 flips on the actual union. The CPU reconstruction of the cp135
base field disagreed with the retained contest-T4 base field at 0 pixels across
the eight affected pairs.

Every pair received a fresh Schur solve against its final rendered token frame.
All eight compensation-object fingerprints matched the exact compile objects;
no QS2 compensation was transferred and no RE1 object was represented as a
fabricated JS6-bank entry. The per-pair projected pose deltas were:

| pair | projected delta d_pose |
|---:|---:|
| 7 | -1.5183318287833222e-10 |
| 96 | -5.724021201932346e-10 |
| 105 | +5.682158697389244e-9 |
| 176 | -1.4297259913784474e-9 |
| 178 | +1.251073426183296e-10 |
| 517 | +4.2466751897414964e-10 |
| 523 | +1.3921942344014621e-9 |
| 532 | -1.1226599409367803e-10 |

Pair 105 alone contributes more pose debt than the final total; the other seven
pairs sum to about `-4.1884e-10`. That localizes the pose failure without
claiming that a different compensation exists.

## Byte closure and custody

The fresh HP3/RC64 object was 186,339 B. Q2C1 split closure produced a 40 B
compensation overlay and a 186,301 B archive. Applying the exact HP4
order-0/Brotli-q11 receiver-identical repack reduced that archive by 5 B to the
terminal 186,296 B object. The terminal archive is still 15 B over MICRO35's
186,281 B ceiling.

| retained object | bytes | SHA-256 |
|---|---:|---|
| `micro35_candidate/archive.zip` | 186,296 | `ca0e2e785ff65260d63673bb8a734cfbe835b345395ad2bea19523f4c94ec4f1` |
| `micro35_candidate/archive.repeat.zip` | 186,296 | `ca0e2e785ff65260d63673bb8a734cfbe835b345395ad2bea19523f4c94ec4f1` |
| final archive member `p` | 186,196 | `6dde9e84199136981700a7f06cede95808a2f076e30bf206a6dc52c4931ab4d6` |
| HP4 model | 70,860 | `ca8a2ec7288d32212d8d6254d840e32a1978c6e61cfb0e27761f6df11575b155` |
| RC64 token stream | 115,240 | `5f3ac2299aa2ef99f16b2a6b0699e495e7354295d79a21f8c56f1512284ffdbf` |
| Q2C1 compensation overlay | 40 | `74b0cefa54a80cf9a191f163a22a185e6c16314a62a21faaf26299e388ed2163` |

Primary and repeat archives are byte-identical. Runtime parse-back recovered the
token stream and fresh compensation overlay exactly and decoded compensation
pairs `[7, 96, 105, 176, 178, 517, 523, 532]`. `unzip -tqq` passed. All
materialized token/event planes, exact rendered masters, solve products, coder
intermediates, archive/repeat, and local scorer inputs/logits/argmax/pose fields
are retained with byte counts and SHA-256 under
`/Volumes/VertigoDataTier/pact/ddm_mc35_20260814/`. Stage checkpoints are
distinct and the runner resumes only when `--resume-from` equals its output
root.

## Measurement boundaries

- The cp135 pin is 186,252 B, SHA-256
  `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`,
  with retained 34,970 flips and contest instrument d_pose
  `6.885642960696714e-6`.
- The local frozen CPU PoseNet baseline recomputed as
  `0.0001474653494795297`; this is an axis-specific advisory instrument and is
  not substituted for the pinned contest value. Only the composed delta is used
  for this charter's local admission gate.
- Receiver closure covers the exact HP4 container, token stream, carrier, and
  compensation overlay. A full public `inflate.sh` video materialization and
  `upstream/evaluate.py` contest-CPU/CUDA run were deliberately not launched
  after the local gates failed.
- No Metal, MPS, Modal, or remote evaluator was used. There is no score claim,
  promotion claim, or public-row claim.

## Implementation and verification

The resumable builder is `experiments/ddm_mc35_micro35_union_build.py`; its
post-edit SHA-256 is
`c88ac26b24f8685d40a96325dac26d8df875e4023ce244d844f98829dbd56b50`.
The retained source-revision receipt records the final source hash and the
reuse of complete retained stages. The implementation closes the previously
non-reentrant split stage from a sealed checkpoint, prevents primary/repeat
module-state collision, makes runtime archive-pin patching idempotent, and
asserts exact external-or-JS6 object bindings for the fresh solve.

Verification passed: Python bytecode compilation of the builder and generated
runtime parser; `ruff check`; `git diff --check`; archive integrity; primary vs
repeat byte identity; token/overlay/HP4 receiver parse-back; gate invariants;
and `tac.preflight.check_no_measure_and_discard_payload` with zero findings.
Two independent `tools/review_tracker.py` scan/mark/policy passes after the last
Python edit reported 20 compliant entities and zero violations.

## RECALL EVIDENCE

Seed sources read in full or at their load-bearing sections:

- `ddm_rfo1_fresh_hybrid_compose_20260814.md` at commit `6fab4cd3fc`, including
  the `RFO1-MICRO35` spec.
- QS2 verdict/memo and retained compile receipt, QS5 exact-object Schur verdict,
  RE1 verdict/memo and shipped decoded object, and HP4 order-0/Brotli-q11 final
  receipt.
- `GT_ATTRIBUTED_DECOMPOSITION.json`, QS3's retained GT identity proof, the
  current frontier pointer/hot state, and the actual receiver/compile code.

Corpus searches included `RFO1-MICRO35|MICRO35`,
`fresh.*compensation|stale.*compensation`,
`GT_ATTRIBUTED_DECOMPOSITION|QS2|QS5|RE1|HP4`, and
`micro35|Schur|compensation|composition|overlap` across `.omx/research`, the
canonical-equations JSON registry, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`,
and task/hot-state stores. Beyond the charter seeds, the QS3 identity evidence
confirmed that the retained GT field was the correct recount denominator, and
the receiver sources exposed the exact external-object binding needed for RE1.
No additional MICRO35-specific settled equation or DAG row was found in those
bounded registry/index scopes. This tightened the build to exact-object
fingerprints and an actual union recount; it did not change the chartered bank
or admission thresholds.

## Disposition

`RFO1-MICRO35` as built is **FOLDED / REFUSED (INSTANCE)**. Its archive and all
evidence remain in custody, but the failed rate and pose gates prohibit a T4
fire order. The effective frontier remains
`S=0.16195513827824176`, 186,252 B, cp135 SHA
`6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`
on `[contest-CUDA]`; this arm moved nothing.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER:** owner `MAIN` under a fresh successor charter;
  consumer store `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_pair105/`;
  fire trigger: first prove a byte-closed object at or below 186,281 B, then
  fresh-solve pair 105 on that exact object and run the same all-eight-pair
  advisory gate. Do not dispatch exact T4 unless all original MICRO35 gates
  pass simultaneously.
- **QUEUED-WITH-A-FIRE-ORDER:** owner `MAIN` under that same successor charter;
  consumer store `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532/`;
  fire trigger: build, rather than sum, the seven-object union without pair 532
  and retain its full coder closure; proceed to scoring only if it preserves at
  least 35 flips and removes the 15-byte rate deficit.

## LIVE-HYPOTHESES

- Pair 105 is the specific pose bottleneck: its `+5.6822e-9` contribution masks
  a slightly pose-improving sum from the other seven pairs, so a different
  exact-object compensation at pair 105 could plausibly close the pose gate.
- Pair 532 costs two Seg flips. Removing it would nominally raise the observed
  union gain from 35 to 37 and may also reduce token/overlay rate, but the
  receiver-closed seven-object result is untested and must not be inferred by
  subtraction.
- The 15-byte rate deficit is localized after an exact HP4 five-byte win. A
  joint overlay/container representation may recover it because the union's
  Q2C1 overlay is 40 B versus the smaller prior object, but no such coder has
  been built on this exact union.

## DEAD-ENDS

- Exact evaluation of this archive is closed: two mandatory local gates failed,
  so a T4 fire would violate the charter.
- Additive QS2 + RE1 + HP4 pricing is closed as evidence: the actual composed
  object is +44 B and has non-additive pose behavior despite zero support
  overlap.
- Reusing QS2's stale compensation is closed: compensation depends on the final
  receiver object, and all eight pairs required fresh fingerprint-bound solves.
- Calling the 35-flip result progress is closed: the effective pointer is
  unchanged and the archive is not promotion-eligible.
