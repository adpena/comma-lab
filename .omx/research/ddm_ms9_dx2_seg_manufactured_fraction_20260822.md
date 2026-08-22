# ddm_ms9 — DX2 Seg error is 90.47% manufactured after the transmitted labels

**Disposition:** `MEASURED-EXACT-FIELD-REPLAY / PARTIAL-STAGE-SPLIT`,
`verdict_scope=INSTANCE:DX2_T4_n600`. The answer is decision-grade for the
representation-versus-downstream split, but not for the finer renderer / resize / uint8 / SegNet
sub-stages. No scorer was launched because MAIN owns the sole n600 scorer lane and this charter did
not grant ms9 that slot.

The exact additive split is:

| charged object | errors / 117,964,800 px | fraction of final error | Seg S | byte-equivalent ceiling |
|---|---:|---:|---:|---:|
| Transmitted representation error | 9,182 | — | — | — |
| Representation error still wrong at final argmax | **2,264** | **9.5298%** | 0.00191922 | **2,882.3 B** |
| Manufactured after a correct transmitted label | **21,493** | **90.4702%** | 0.01821984 | **27,362.9 B** |
| Final DX2 Seg error | **23,757** | 100% | **0.02013906** | **30,245.2 B** |

The byte figures are oracle-equivalent ceilings, not mechanisms and not predicted archive savings.
They use the exact contest exchange rate, 1.2731082153 B per eliminated flip. Removing every
manufactured flip while preserving pose, rate, and all beneficial downstream corrections could be
worth at most 27.36 KB. No measured mechanism currently achieves that.

The retained numerator independently gives `23,757 / 117,964,800 = 0.00020139058430989585`, which
rounds to the official eight-decimal `d_seg=0.00020139`. Its exact 30,245.2 B exchange ceiling is
2.8 B below the charter's 30,248 B shorthand; this memo uses the exact integer numerator throughout.

## Exact definition and why RT1's old ratio cannot be reused

Let `G` be the contest-CUDA DALI GT argmax, `L` the transmitted DX2 semantic-label field after exact
decode, and `A` the terminal contest-CUDA argmax field. Define:

```text
final_error                         = A != G
manufactured_final_error            = (A != G) and (L == G)
representation_survived_final_error = (A != G) and (L != G)
```

These last two masks partition the final-error support exactly: `21,493 + 2,264 = 23,757`.

The RT1 diagnostic `count(A != L) / count(A != G)` is not an additive manufactured fraction on
DX2. Here it is `28,553 / 23,757 = 120.19%`. The downstream path corrects **6,918** transmitted-label
errors and makes **142** wrong-label-to-different-wrong-label changes. Treating every `A != L` as a
manufactured final error double counts the beneficial corrections. This is a definition correction,
not a changed scorer result.

## Authority and custody join

DX2 is pinned to archive SHA-256
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`, 180,368 B, with
`d_seg=0.00020139`, `d_pose=0.00000637`, and
`S=0.14821987563243377` on `[contest-CUDA T4 n600]`.

This arm did not claim a new score. It replayed exact retained component fields:

| field | SHA-256 | bytes | custody / meaning |
|---|---|---:|---|
| DALI GT argmax | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` | 117,964,928 | `/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy` |
| decoded DX2 labels | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | 117,964,800 | retained F26 n600 token field |
| retained FX5 argmax | `e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34` | 117,964,928 | component-only T4 argmax payload |

The FX5 field transfers exactly to DX2 because the independent FX5 and DX2 T4 receipts both bind
their complete inflated `0.raw` to SHA-256
`6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883`,
3,662,409,600 B. DX2 changes only the entropy representation; its shipped T4 render bytes are
byte-identical to FX5. The evidence grade is therefore
`[contest-CUDA T4 component-only exact field replay]`, `score_claim=false`.

The T4 raw payload itself is not retained locally by this arm; its bytes are bound only through the
two remote manifests. This arm did not rematerialize or discard that payload. The missing local T4
raw custody is recorded, not papered over by the different local decode.

All 45 measured global, per-class-support, per-class-error, and RT1-control masks were retained as little-endian packed-bit n600 payloads
under `/Volumes/VertigoDataTier/pact/ddm_ms9_dx2_seg_manufactured_fraction/retained/masks/`.
The retained store is 663,632,396 B across 49 files. `MS9_FIELD_REPLAY.json` has SHA-256
`872e0b19032d3357619913beea32956c49c2716f7e894610a963feb63d3918cc`; `MASK_MANIFEST.json`
has SHA-256 `2df0abbae76a1234f8af0e5a08bd857254cf9a6299f63a61b9ae06021b329cdd`.
`MS9_FIRE_ORDER.json` has SHA-256
`dfce97357c9e3b61eabedb38bb55f3f738596dda7b159cf1a8898a93d4e1b0cb`. A no-randomness resume
replay reproduced the receipt SHA byte-for-byte.

## Per-class result

Classes are charged by the GT class of each pixel. IoU is field-versus-DALI-GT on the same n600
population.

| GT class | GT px / area | final errors / share | transmitted-label IoU | final-argmax IoU | manufactured / class errors | manufactured ceiling |
|---|---:|---:|---:|---:|---:|---:|
| Road | 27,407,372 / 23.2335% | 9,305 / 39.1674% | 0.999725 | 0.999303 | 8,385 / 90.1128% | 10,675.0 B |
| **Lane** | **690,754 / 0.5856%** | **5,856 / 24.6496%** | **0.994004** | **0.984779** | **5,285 / 90.2493%** | **6,728.4 B** |
| Undrivable | 58,413,067 / 49.5174% | 4,432 / 18.6556% | 0.999942 | 0.999848 | 4,117 / 92.8926% | 5,241.4 B |
| Movable | 1,460,386 / 1.2380% | 3,232 / 13.6044% | 0.998359 | 0.995092 | 2,952 / 91.3366% | 3,758.2 B |
| MyCar | 29,993,221 / 25.4256% | 932 / 3.9231% | 0.999970 | 0.999941 | 754 / 80.9013% | 959.9 B |

Lane is still the densest target: 0.5856% of GT area carries 24.6496% of the final errors, a 42.10×
error-share enrichment. But its manufactured fraction, 90.2493%, is slightly **below** the body-wide
90.4702%. The registered claim that Lane would be more manufactured than average is falsified on this
object. The charter's seeded “GT IoU 0.263” is not the same exact DX2 field metric and does not
transfer: the current transmitted Lane IoU is 0.994004 and terminal Lane IoU is 0.984779.

## Stage attribution and cheapest honest mechanisms

The actual shipped ordering is more precise than the charter shorthand:

```text
decoded semantic labels
  -> native RGB render
  -> bilinear camera upsample + clamp/round to uint8
  -> evaluator float conversion + bilinear downsample
  -> SegNet logits
  -> argmax
```

The exact fields resolve only `decoded semantic labels` versus the **combined** downstream map. They
do not contain the pre-uint8 render, post-uint8/pre-evaluator field, post-resize RGB, or SegNet logits.
Therefore no percentage is assigned to render, resize, uint8, or the terminal argmax operation.

| stage | measured charge | cheapest plausible mechanism | collateral and admission rule |
|---|---:|---|---|
| Representation | 2,264 surviving errors; 9.5298%; 2,882.3 B ceiling | A scorer-aware quotient/token representation is the live cheap family. RI1 K=2,048 is instance-dead; NI1 K32 is byte-closed but still unscored. | Must price the real recode and preserve the 6,918 downstream corrections. A correction requiring more than 1.273 B per flip loses before pose collateral. |
| Native RGB render | not separately measured | Train or solve against the realized Seg objective at the renderer boundary; target the retained manufactured support, especially Lane. | Must not erase transmitted errors that the renderer currently corrects; must remeasure pose through the shipped receiver. |
| Camera lift + uint8 | not separately measured | A pre-quantization/preimage correction is the smallest plausible actuator. | RT2 found the corresponding uint8 effect small on HV1; that ancestor result does not transfer to DX2 and is hypothesis-only here. |
| Evaluator resize tail of R | not separately measured | Train-through-R or a bounded pre-resize inverse conditioned on exact receiver geometry. | RT2's generic deblur and blur-axis ladders were dead on HV1. Repeating those same unconditioned ladders on DX2 is not justified. |
| SegNet logits / argmax | argmax field retained, logits absent | Margin-directed realized acceptance, if exact logits are newly retained. | No current margin field means this stage is unreachable from the retained payloads. Any claimed split would be invented. |

The large manufactured ceiling does not license a generic decode-side perturbation. The current map
simultaneously manufactures 21,493 final errors and corrects 6,918 representation errors. A viable
mechanism has to discriminate those supports and then survive the pose axis; “make the render look
more like the labels” is not a sufficient objective.

## RT1 positive control

The retained scorer-free RT1 replay reproduced all registered counts exactly:

- final HV1 argmax versus DALI GT: 34,938 errors;
- retained HV1 labels versus DALI GT: 1,717 errors;
- final HV1 argmax versus retained HV1 labels: 33,743 changes.

This is a replay-only positive control, not a fresh scorer control. It verifies that the field-join
instrument preserves RT1's historical inputs and denominators. It does not transfer RT1's 96.6%
non-additive ratio into the corrected DX2 definition.

## Prior-law verdict

The primary qualitative prediction survives: DX2 is overwhelmingly downstream-manufactured, not
representation-limited. The registered 60–90% numeric band is narrowly falsified at **90.4702%**, 0.47
percentage points above its upper edge. The hard falsifier `<25%` did not fire. The Lane-above-average
sub-prediction did fire its falsifier, as described above.

The practical conclusion is asymmetric. Representation-only work can directly address at most 9.53%
of the current final Seg numerator, but may still win through large rate savings. Seg-quality work has
a 90.47% downstream support to target, yet its exact internal mechanism is still unmeasured.

## RECALL EVIDENCE

The pre-measurement recall searched the full `.omx/research` corpus by content, then the canonical
research index, the sub-0.15 DAG, the canonical task ledger, and the equation catalog. Queries included:

- `ddm_rt1`, `ddm_rt2`, `manufactured Seg`, `manufactured fraction`, `round-trip Seg`, `R_surv`,
  `uint8`, `resize`, and `realized` across Markdown, JSON, and JSONL;
- the canonical equation catalog via `tools/list_canonical_equations.py --json` with the same
  mechanism terms;
- bounded searches of `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, and
  `.omx/state/canonical_task_status.jsonl`.

Beyond the charter seeds, the search recovered
`.omx/research/ddm_rt2_manufactured_seg_mechanism_20260817.md`. RT2 fixes the actual operator order,
shows that generic deblur and blur-axis cures were dead on HV1, and warns that its n8 uint8 fraction is
ancestor-only. That changed this run in three ways: it prevented a false `R then uint8` ordering,
prevented reuse of HV1 mechanism percentages on DX2, and forced the finer stage split to remain
unmeasured. The DAG also preserved the train-through-R law and the exact 384->camera->uint8->384
geometry, but contains no retained current-DX2 progressive field set. The task ledger and indexes did
not contain an existing ms9 result in the bounded scopes searched.

The current RI1/NI1 handoffs also changed the inherited representation summary: RI1's K=2,048 RC1
instance completed and is dead at `d_seg=0.01605413`, while NI1's distinct K32 archive is byte-closed
but explicitly **not scored** because it does not own the lane. This memo therefore does not repeat
the charter's stronger claim that both current objects have measured distortion verdicts.

The own insight from the exact DX2 join is the additive-definition correction. It both lowers the
honest manufactured numerator from all `A != L` changes and reveals the 6,918 beneficial corrections
that any cure must preserve. The second new result is the exact class split: Lane is massively
error-enriched but not more manufactured than the average pixel.

## Reproducibility and verification

- Instrument: `experiments/ddm_ms9_dx2_seg_manufactured_fraction.py`.
- Durable store: `/Volumes/VertigoDataTier/pact/ddm_ms9_dx2_seg_manufactured_fraction/`.
- Determinism: no RNG; source SHA/byte pins fail closed; masks are atomically written; checkpoint
  advances after every retained payload; a second resume pass reproduced the receipt SHA exactly.
- Storage/retention: the runner requires the Vertigo tier, fails closed below 2 GiB free, and the
  targeted `check_no_measure_and_discard_payload` audit returned no findings. A deliberate APDataStore
  output attempt refused before creating a directory.
- Static and independent verification: `ruff check`, `python -m py_compile`, all 45 packed-mask
  count replays, class-partition identities, the DALI lineage binding, and decoded-token binding passed.
- Scope protection: no upstream file, RT1/TD1/rate tree, jo1 r9 payload, staged index, scorer lane, or
  Modal resource was modified or consumed.

## Follow-on disposition

`QUEUED-WITH-A-FIRE-ORDER`: MAIN owns the only remaining action. When MAIN grants ms9 the unique n600
scorer lane after all currently admitted jobs are terminal, build a fresh DX2 progressive instrument
that retains and scores the native render, camera-uint8 field, evaluator-resized field, logits, and
argmax in the actual operator order. It must first reproduce the retained RT1 control and this DX2
23,757-error terminal field, write every new field and mask under
`/Volumes/VertigoDataTier/pact/ddm_ms9_dx2_seg_manufactured_fraction/fresh_stage_split_r1/`, and use the
canonical local advisory firer for every full-n600 advisory. The consumer is the next DX2 Seg-cure
selector; the fire trigger is an explicit MAIN lane grant plus a fresh, empty attempt store and a
passing storage/provenance preflight. Until then, render/R/uint8/argmax sub-fractions stay unmeasured.

OWN-VEHICLE FRONTIER: UNMOVED — DX2 remains S=0.14821987563243377 @ 180,368 B [contest-CUDA T4 n600], archive SHA-256 976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674; ms9 made no score claim.
