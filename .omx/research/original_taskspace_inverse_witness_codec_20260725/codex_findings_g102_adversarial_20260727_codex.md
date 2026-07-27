# G103 adversarial review — G102 final-Y1 semantic-base escape

Date: 2026-07-27  
Lane: `lane_g103_g102_adversarial_review_20260727`  
Authority: bounded read-only adversarial review; no heavy launch, evaluator,
candidate, score, pointer mutation, implementation edit, or commit  
Verdict: `DIRECTION_ADMISSIBLE_S01_PHYSICALLY_BLOCKED_CORRECTIONS_OWED`

## Outcome first

G102's central substitution is admissible: the next representation must be a
fresh, P-free, whole-population selected-solution program whose counted state
generates the scorer-native RGB solution through exact `R`, followed by exactly
one final-Y1-conditioned Y0 owner. It must not be another additive prefix around
P/G85 and must not rediscover the settled direct-mask/palette plus independent
pose split.

No G102 row is physically executable from the two reviewed artifacts. That is
not a hidden failure in G102's prose: both artifacts say the prerequisite is
blocked. The exact current blocker remains:

`P_FREE_ROOT_SEMANTIC_PROGRAM_COMPILER_AND_PUBLIC_CODEC_SECTION_OWED`

The review found four corrections that must be carried before downstream
execution:

1. restore immutable artifact custody after both "frozen" files changed after
   G102's completion checkpoint;
2. quarantine the historical C1 low-distortion receipt to planning/existence
   evidence, not an encoder teacher;
3. distinguish the 133,941-byte semantic-P source archive from G85's actual
   129,392-byte charged outer archive when reporting additive headroom; and
4. narrow G57 reuse from "public receiver" to its staged raw-assembly pattern,
   because the existing receiver is hard-wired to PyAV raster layers.

The strict `187,563 B` and `154,523 B` whole-archive envelopes themselves are
correct.

## Reviewed artifact custody and mid-review drift

G102 completed at `2026-07-27T09:09:49.876766Z` and its canonical checkpoint
froze:

| artifact | checkpoint bytes | checkpoint SHA-256 |
|---|---:|---|
| findings memo | 16,837 | `43d84d60d4c88f2fb6565084d2d6c8913f5d00afcab08c2bd405cd70a10d1d93` |
| activation JSON | 12,915 | `eb43aa55e68234665fc1f890b7022c1fec97bcc59a25ce22300a9742e054d2fd` |

During this adversarial review, both files were modified at
`2026-07-27T09:16:22Z`, after the G102 completion checkpoint:

| artifact | reviewed current bytes | reviewed current SHA-256 |
|---|---:|---|
| `codex_findings_g102_final_y1_semantic_base_escape_20260727_codex.md` | 18,520 | `1748df4ebcf157136560106dea12a3132d6654a4035fdc41f07546520a2f3a36` |
| `g102_final_y1_semantic_base_escape_activation_20260727.json` | 14,380 | `f4902dfd414c0d69d4815e97856c1180fcf97144d2eda8b647e7aa5d423db059` |

The new bytes improve the formulation by adding the FEED-af/ah guard, but they
invalidate the old completion checkpoint as custody for the current files.
This memo records both identities; no G102 file was edited by this review.
Smallest correction: append a superseding G102 custody receipt/checkpoint that
names the current two hashes, or restore the completed immutable bytes and put
the FEED amendment in a new append-only artifact.

The activation JSON's `created_at_utc=2026-07-27T09:18:00Z` is also later than
its filesystem modification time `2026-07-27T09:16:22Z`. This is a provenance
timestamp defect, not score evidence.

## Score-envelope recomputation

For the cited batch-16 planning coordinate

```text
d_seg = 0.00015196058485243054
d_pose = 0.00010184347386600314
D      = 100*d_seg + sqrt(10*d_pose)
       = 0.047108982805336805
```

the strict integer envelope

```text
Bmax(T) = ceil((T - D)*37,545,489/25) - 1
```

recomputes exactly:

| target | real-valued byte crossing | strict `Bmax` | score at `Bmax` | score at next byte |
|---:|---:|---:|---:|---:|
| `0.172` | `187563.7724912415` | `187,563` | `0.17199948562979062` | `0.17200015148874376` |
| `0.15` | `154523.74217124152` | `154,523` | `0.14999950581863408` | `0.15000017167758722` |

The activation's byte-conditioned `d_seg` examples also recompute exactly. The
G57 score recomputes to `39.30593503092899`, and the MS2R score recomputes to
`194.42556029038283`, including its zero-byte distortion
`0.5238375028596528`.

## G85 byte identities: no hash collision, one mislabeled comparison

The G85 closure receipt and archive reopen as:

```text
G85 receipt SHA-256       2b36c510d6502bcff26deba486680980601a80f9726af40658a301a2c59f415e
charged outer archive     129,392 B
outer SHA-256             b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd
PVSA archive member       133,363 B
member SHA-256            d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31
source semantic P         133,941 B
source P SHA-256          759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df
```

Thus 129,392 and 133,941 are different physical objects, not contradictory
measurements of one object. G102 correctly names the former in its G85 row and
the latter as the historical-shape semantic base. The ambiguity is in calling
only the latter "additive headroom":

| base identity subtracted | headroom below `0.172` | headroom below `0.15` |
|---|---:|---:|
| 133,941-byte source semantic P | 53,622 B | 20,582 B |
| 129,392-byte current G85 outer archive | 58,171 B | 25,131 B |

The 4,549-byte difference does not change the substitutive-rate conclusion:
the new root can use the whole archive envelope only when the inherited base is
physically absent. Smallest correction: report both identities explicitly and
never use the 133,941-byte source-P subtraction as the charged-current-G85
headroom.

G102's other G85 claims are supported: its private full-n600 advisory row has
`d_seg=0.0274712`, `d_pose=163.06130981`, and its public default is not closed
because `pydantic` is missing. Private double decode is not public-entrypoint
closure.

## G57 public-mechanics scope

The G57 receipt supports the factual claims that its exact 182,220-byte archive
decoded twice from clean roots to identical 3,662,409,600-byte raw output,
finished in about 372 seconds, and then completed full-n600
`upstream/evaluate.py` on macOS CPU advisory. Its cited archive, stream, raw,
runtime, report, score, and source hashes match.

What is reusable is narrower than
`"public_receiver_reusable": true`. The current
`submissions/robust_current/taskspace_layered_public/inflate.py`:

- requires `y1-base-plus-centered-signed-diff-q2-y0-given-y1.v1`;
- requires `typed-pyav-packed-or-separate-rgb.v1` and a pinned PyAV contract;
- decodes the two raster layers inside `_build_stage`; and
- only then performs factor-2 realization and atomic stage/raw assembly.

Therefore the existing G57 receiver cannot consume a semantic-root packet.
The reusable object is its strict member census, checkpoint verification,
atomic stage/final assembly, raw hash validation, and extracted-directory
closure pattern. A new typed semantic-root dispatcher/adapter must call that
machinery after semantic decode. G102's prose says this correctly
(`G57-shaped public stage/raw assembly`); the activation boolean should be
scoped as:

```text
g57_staged_raw_assembly_pattern_reusable = true
existing_g57_public_receiver_accepts_semantic_root = false
```

## Fresh-lineage boundary

All 13 current activation JSON path/SHA pairs reopen and match, including the
pointer, fresh batch-16 target-label bank, C1 planning receipt, MS1/MS2R
receipts, FEED-af/ah evidence, G57 receipt, and five cited receiver sources.
The bare G89, G90, and G92 receipt hashes also match their files.

The G46 batch-16 target-label bank is a valid fresh encoder-only boundary
condition: 600 ordered planes, 117,964,800 bytes, SHA-256
`6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85`.
It must never be candidate payload.

The historical C1 receipt is different. Its own config points to
`c1_two_plane_receiver_20260719/out_serial/0.raw` and an M2 live-target raw.
G102's prose correctly limits historical MS1/C1 to low-distortion existence
and factor-family nomination. The activation JSON instead calls
`c1_live_target_debt_n600_batch16.json` a
`planning coordinate and encoder teacher`. That role is too broad and conflicts
with the fresh-lineage contract. Smallest correction:

```text
G46 batch-16 target bank = encoder boundary condition
C1/MS1/MS2R = metrics, existence, and factor-family evidence only
C1/MS1/MS2R values, planes, events, residuals, and derivatives = forbidden
candidate lineage
```

The activation's `historical_payload_reused=false` is true only for G102
itself, because G102 created no payload or candidate. It is not evidence that a
future S01 compiler is fresh. S01 must emit a source manifest and prove that
every counted video-derived value descends from the fresh G46/source/scorer
encode path, with no C1/MS1/MS2R/G85/G57/PR86/PR130 data dependency.

## FEED-af/ah settled guard and exact scope

The current reviewed G102 artifacts now carry the required anti-rediscovery
guard, and both evidence hashes match:

| settled evidence | exact fact | scope |
|---|---|---|
| `symbolic_topological_partition_mdl.json`, SHA `53659d3ca68cc1a054fa57151ad063991e70762be8471f28bfb839027528595a` | real full-n600 bit-exact temporal context coder, `255,288 B`, direct label `d_seg=0`, implied advisory `S=0.19285285137750108` | encoder-side label-rate evidence; not realized candidate evidence |
| `feedy_byteclosed_exact_row_20260625/exact_row.json`, SHA `12e455c5c05b4e00d3226f76545e257f96aee1e509b1d73c6e554aaf68b7af05` | 24-pair advisory scorer mirror: exact-label palette gives realized `d_seg=0.00640763176812066`, `d_pose=188.44457475619993`; its 600-pair archive number is extrapolated | kills only direct mask/label grammar + palette + independent pose split |

FEED-ah's negative is formulation-scoped. It does not kill topology, labels,
level sets, or worldsheets as factors inside a joint scorer-native generator.
It requires the opposite of the split:

`ONE_SCORER_NATIVE_RGB_SELECTED_SOLUTION_PROGRAM_WITH_COUNTED_TEXTURE_CHROMA_BOUNDARY_SURVIVAL_AND_PARALLAX_GAUGE_THROUGH_EXACT_R`

The current activation states this exact requirement and forbids a palette-only
root. The whole selected solution may remain causally factorized as final Y1
plus exactly one Y0 owner, but those are sections of one jointly scored RGB
program, not an independent label store plus pose carrier.

## S01 physical executability

`S01_ROOT_PROGRAM` is a treatment contract, not a runnable experiment. At the
review snapshot:

- no `taskspace_pfree_semantic_root_v1.py` implementation existed;
- no source-backed compiler emitted a counted full-n600 `SemanticRootY1V1`;
- no public dispatcher consumed such a packet;
- the activation named no runner path, command, typed config hash, checkpoint
  root, emitted archive identity, or executable resume command; and
- no S01 archive, receiver output, or full-n600 row existed.

The stage list's resumability assertions are requirements, not proof. S01
becomes physically executable only when one source-backed compiler and
scorer-free receiver exist, the public dispatcher consumes their exact packet,
and a governed runner can start/resume from a declared on-disk stage root.

The smallest honest execution order is:

1. land and source-seal the P-free counted wire/parser/renderer;
2. compile one full-n600 packet from the fresh G46 encoder boundary, with no
   historical data dependency;
3. integrate a semantic-root public dispatcher with the extracted G57
   stage/raw-assembly pattern;
4. emit a complete S01 archive and immutable five-stage resume receipts;
5. run full-n600 Seg/Pose/rate on that exact archive; and
6. only after final Y1 is selected, fit exactly one conditional Y0 owner and
   repeat public closure.

No n2/n96 substitute, direct-label score, estimated outer bytes, private-only
decode, or stage declaration satisfies S01.

## Final classification

| item | classification | smallest correction |
|---|---|---|
| substitutive P-free direction | `ADMISSIBLE` | preserve as one scorer-native RGB program through `R` |
| 187,563 / 154,523 envelopes | `VERIFIED_EXACT` | none |
| FEED-af/ah guard | `VERIFIED_FORMULATION_SCOPED` | keep topology as factor; forbid label+palette+pose split |
| S01 launch | `BLOCKED_NOT_PHYSICALLY_EXECUTABLE` | real compiler, receiver, dispatcher, governed resume command |
| frozen G102 custody | `BLOCKING_PROVENANCE_DRIFT` | append superseding hashes/checkpoint or restore immutable files |
| C1 role | `BLOCKING_LINEAGE_SCOPE_ERROR` | planning/existence only; G46 is the encoder boundary |
| G85 headroom | `NONFATAL_IDENTITY_AMBIGUITY` | publish both 133,941 and 129,392 subtractions |
| G57 reuse | `NONFATAL_MECHANISM_SCOPE_ERROR` | reuse assembly pattern, not the raster receiver |
| activation timestamp | `PROVENANCE_METADATA_ERROR` | supersede with observed write time |

Pointer delta from this review is exactly zero. No candidate or score was
created, and the exact mission remains unachieved.
