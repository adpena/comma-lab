# ddm_ar1b — the 66,591-byte DX2 residue is exactly mapped; its distortion purchase remains QUEUED

`date_utc: 2026-08-23` · `axis: [macOS-CPU scorer-free exact archive-byte parse]` ·
`score_claim: false` · `frontier_moved: false` · `verdict_scope: INSTANCE:DX2_PHYSICAL_CENSUS_ONLY`

## Answer first

The exact non-token residue is **66,591 B**, and all of it is reconciled to physical archive spans
with SHA-256 custody. There is **no unexplained remainder**:

| residue class | bytes | share of residue | share of 42,382 B demand |
|---|---:|---:|---:|
| semantic renderer | 30,856 | 46.3378% | 72.8045% |
| carrier | 22,010 | 33.0526% | 51.9324% |
| HPAC probability model | 13,515 | 20.2976% | 31.8885% |
| fixed residual table | 96 | 0.1442% | 0.2265% |
| ZIP + RX1 structural framing | 114 | 0.1712% | 0.2690% |
| **total** | **66,591** | **100.0000%** | **157.1209%** |

This is an anatomy result, not a purchase result. The exclusive n600 scorer lane is owned by MAIN
and was unavailable to this arm. Therefore every realized `delta d_seg`, per-class consequence,
`delta d_pose`, distortion price, candidate byte credit, net `delta S`, S-per-byte rank, and
waterfill remains **QUEUED**. A section's current byte mass is not called headroom, and the byte-only
priority below is not called purchase attribution.

No scorer, render, candidate build, Modal job, or Metal job ran. No payload was materialized or
discarded. The shipped archive, receiver, and GT tables were read only. No shipping candidate was
built.

## Pins, arithmetic, and custody

| object | bytes | SHA-256 | custody/use |
|---|---:|---|---|
| DX2 `archive.zip` | 180,368 | `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` | exact body, read only |
| ZIP member `p` | 180,268 | `365f1b8d70463b250a2fe95e3599318ac90b31875cce5d66a767819404431c7a` | one `ZIP_STORED` member |
| excluded RC64 token stream | 113,777 | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` | BL1-owned field, read only |
| shipped receiver tree | 685,975 | `7799b291a99027c705b42f094cf0533459399f3ea711ec34d754f81c1fde5f1d` | 39 files, read only |
| DALI Seg GT NPY | 117,964,928 | `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248` | `600x384x512`, `uint8` |
| DALI Pose6 GT NPY | 14,528 | `8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff` | `600x6`, `float32` |

The source archive is
`/Volumes/APDataStore/pact/ddm_dx2/r7/retained/candidate_dx2_cabac.zip`; the receiver is
`/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2`. The exact archive and token pins match
the charter.

The score exchange rate is cited from
`.omx/research/ddm_tx1_toolbox_crosswalk_20260819.md` section 0, not re-derived:
`25/37,545,489 = 6.658590e-07 S/B`. DX2 remains
`S = 0.14821987563243377`, `d_seg = 0.00020139`, `d_pose = 6.37e-6`, archive 180,368 B
`[contest-CUDA T4, n600]`. At that distortion the strict sub-0.12 ceiling is 137,986 B, so the
complete archive must lose **42,382 B**.

Receipts are on the charter's explicit local-disk tier:
`.omx/tmp/arm_receipts_local/ddm_ar1b_archive_residue_purchase/`. The observed free-space waterfall
was local 477 GiB, Vertigo 8.4 GiB, and APDataStore 11 GiB. **No write to either `/Volumes/*` tier
occurred.** `upstream/` was not modified, and the sacred JO1 r9 tree was not read or touched.

## Exact physical census: 66,591 B, zero remainder

Offsets are zero-based half-open archive offsets. Every row is a contiguous span of the exact
180,368-byte archive.

| physical residue region | archive span | bytes | SHA-256 | role |
|---|---:|---:|---|---|
| ZIP local header | `[0,31)` | 31 | `8da112adac53d7f8e1362d0490ab39faddc1a2b4bd9898d4201497e3e5ae2761` | local framing for member `p` |
| RX1 header | `[31,45)` | 14 | `1f46427524c9e640ae79a4607947e0909cdea3a2656cca250bede5d8aa9ef61c` | `RX1M` v1 codec/flag/length grammar |
| HPAC stream | `[45,13560)` | 13,515 | `602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98` | restores canonical IHS1 probability model |
| semantic stream | `[13560,44416)` | 30,856 | `39d1be52ba62933498395c48ce4d9482f37db097d504da76c2a321efe3e4a76f` | restores current SM3R renderer packet |
| carrier stream | `[44416,66426)` | 22,010 | `932b979f5181b331a9099162c6f392f558860b7998c62a36f38c2c99629c9b12` | restores CAP1 carrier + frame-0 selector |
| compact residual | `[66426,66522)` | 96 | `8ab2fe748ab7d69d2102ba2292289e22bd7ea503f8ae29938e0854ec46ca3da1` | fp16 scale + 125 signed six-bit codes |
| **excluded token stream** | **`[66522,180299)`** | **113,777** | `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` | outside this arm's residue |
| ZIP central directory | `[180299,180346)` | 47 | `0a43beeb892dfda1061dc3bc5f1bb01d4d1cf828a01afb8e2a9735df59e25112` | central-directory framing |
| ZIP EOCD | `[180346,180368)` | 22 | `57fb4da8116971bc2b4772936c9414db1ae63af6e023a23e8b267928bb78bdd1` | end-of-central-directory record |

The exact identities are:

```text
31 + 14 + 13,515 + 30,856 + 22,010 + 96 + 47 + 22 = 66,591 B
66,591 + 113,777 = 180,368 B
unexplained remainder = 0 B
```

This independently reproduces RB1/TO2's section lengths while adding per-span hashes and offsets.
It does not convert their fixed-representation coder closure into a distortion-purchase verdict.

## Current logical parameter map

The physical rows above are the only additive byte accounting. HPAC, semantic, and carrier are each
outer-compressed as a whole, so decoded subgroups do **not** own additive physical bytes. Assigning
each raw tensor a prorated share would fake the very S/B field the charter asks to measure. A future
purchase row must perturb one subgroup, run the real packer, and use the resulting complete archive
delta.

### Semantic renderer: already SM3R mode 6, keep 1%

The 30,856-byte physical stream restores a 36,130-byte packet, SHA-256
`17e0fd0b197ac147afe98397ef38f02f7915b69372d03c042e6be6fa0f992e50`. Its header is
`SM3R v1 mode=6 keep_percent=1`, selection mask `0x4900`. The three selected tensors are
`blocks.1.film.weight`, `blocks.2.film.weight`, and `blocks.3.film.weight`; each retains 2 of 192
rows. The additive decoded packet map is:

| decoded semantic subgroup | decoded bytes | current state / natural next quantum |
|---|---:|---|
| SM3R header, mask, and depth table | 18 | structural, not a parameter purchase |
| currently-q4 tensor values | 28,530 | one predeclared tensor group q4 to q3 |
| currently-q3 values: `frame_embed.weight`, `blocks.0.film.weight` | 2,776 | one group q3 to q2 |
| fp16 biases and norms | 4,806 | value coarsening only; real outer repack decides any credit |
| **total decoded SM3R** | **36,130** | exact |

MZ2's 38/38 receiver-required result still applies to plumbing. It does not supply current-DX2
distortion sensitivities. The current object has already consumed keep01 and compensated it before
promotion (`ddm_sa3_compensated_edit_rebased_verdict_20260818.md` and
`ddm_keep01_ninth_pointer_move_verdict_20260818.md`). Consequently, “try keep01,” another global
keep-percent rung, or an older uncompensated FiLM-prune replay is not a live AR1B row. A legitimate
current row is narrower: one bit-depth quantum for one extant tensor group, or one of the two
surviving rows in one already-keep01 tensor, with every other decoded object byte-identical.

### Carrier: CAP1 basis, coefficients, and selector

The 22,010-byte physical stream restores a 22,316-byte carrier blob, SHA-256
`89aab077cead53136f8f5f556bb3f9d1977fb8796d53b53f639f478fab0d0d82`. Its additive decoded map is:

| decoded carrier subgroup | decoded bytes | role |
|---|---:|---|
| F0C1 wrapper | 6 | structural carrier/selector split |
| CAP1 header | 14 | version and bit counts |
| predictor metadata | 36 | 12 AR factors + biases |
| basis and coefficient scales | 96 | 24 fp32 scales |
| basis lengths | 32 | packed basis geometry |
| Rice parameters | 12 | one `k` per dimension |
| basis payload | 12,277 | 98,213 meaningful bits |
| coefficient payload | 9,829 | 78,628 meaningful bits |
| frame-0 selector | 14 | sparse selector operations |
| **total decoded carrier** | **22,316** | exact |

Post-hoc carrier rank/refit and another carrier-coder race are already closed in their measured
scopes; they are not queued again. The remaining pure-purchase probes are direct one-quantum current
DX2 value perturbations to a predeclared basis, coefficient, or selector group, followed by the real
outer repack and full scorer path.

### HPAC and fixed residual

The 13,515-byte HPAC physical stream restores a 17,952-byte canonical IHS1 object, SHA-256
`e8c0cfd73d3275adeff2897ea83efa9d045855c43fb3bb66ac037e5c84f2e6dd`. Its natural purchase axis
is one signed-depth quantum on one predeclared IHS1 row/module group, not a coder swap and not an
unchanged-value recode.

The residual section is physically additive: 2 bytes of fp16 scale plus 94 bytes carrying 125
signed six-bit codes. The receiver restores the four-byte `RCF1` magic, yielding the 100-byte
canonical payload SHA-256 `74775aab04c7615cacabfddfa185efa05429cc0d8e21fd1a1d84f37bbc79d750`.
The table is `boundary_predicted`, scale `0.049591064453125`, and code-array SHA-256
`76afdc3ceda1212a530ade05bc2f23c8eafcf8a18895de2200d64cab69dc1060`. Coarsening its values can
measure distortion load, but the fixed current representation remains 96 bytes; it returns **0 B**
without the storage-layout change the charter forbids.

## Ranked byte table: measurement priority only

This is sorted by **current physical bytes held**, not S/B. “Byte credit” means the complete archive
delta after a real repack; except for the fixed residual's known zero, it is not estimated from raw
decoded sizes.

| byte priority | physical parameter group | bytes held | share of demand | byte credit | receiver-required | distortion-load-bearing | realized `delta d_seg` | realized `delta d_pose` | `delta S_distortion` | net `delta S` |
|---:|---|---:|---:|---|---|---|---|---|---|---|
| 1 | semantic SM3R mode-6/keep01 | 30,856 | 72.8045% | **QUEUED real repack** | **YES**: 38/38 tensors consumed | **UNKNOWN — QUEUED** | **QUEUED n600 real path** | **QUEUED n600 DALI Pose6** | **QUEUED** | **QUEUED** |
| 2 | CAP1 carrier + selector | 22,010 | 51.9324% | **QUEUED real repack** | **YES**: current parser and renderer consume it | **UNKNOWN — QUEUED** | **QUEUED n600 real path** | **QUEUED n600 DALI Pose6** | **QUEUED** | **QUEUED** |
| 3 | IHS1 HPAC probability model | 13,515 | 31.8885% | **QUEUED real repack** | **YES**: current parser and token decoder consume it | **UNKNOWN — QUEUED** | **QUEUED n600 real path** | **QUEUED n600 DALI Pose6** | **QUEUED** | **QUEUED** |
| 4 | RCF1 residual scale + codes | 96 | 0.2265% | **0 B in fixed layout** | **YES**: strict table parser consumes it | **UNKNOWN — QUEUED** | **QUEUED n600 real path** | **QUEUED n600 DALI Pose6** | **QUEUED** | **QUEUED** |

ZIP/RX1 framing holds 114 B but is structural rather than a parameter group. It is receiver-required
grammar, has no natural quantization axis, and is excluded from the purchase table.

## Required per-class rows and GT lineage

Every queued candidate must report all five rows, not only aggregate Hamming distance:

| DALI GT class | denominator | semantic | carrier | HPAC | residual |
|---|---:|---|---|---|---|
| Road (0) | 27,407,372 px | QUEUED | QUEUED | QUEUED | QUEUED |
| **Lane (1)** | **690,754 px** | **QUEUED** | **QUEUED** | **QUEUED** | **QUEUED** |
| Undrivable (2) | 58,413,067 px | QUEUED | QUEUED | QUEUED | QUEUED |
| Movable (3) | 1,460,386 px | QUEUED | QUEUED | QUEUED | QUEUED |
| MyCar (4) | 29,993,221 px | QUEUED | QUEUED | QUEUED | QUEUED |
| **all classes** | **117,964,800 px** | **QUEUED** | **QUEUED** | **QUEUED** | **QUEUED** |

Seg lineage is the pinned contest-CUDA DALI argmax field at
`/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy`; the counts above
were read directly from that exact NPY this turn. Pose lineage is the official DALI `600x6` table at
`/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_dali_n600.npy`, denominator 3,600 scalar targets. No PyAV-lineage
GT may fill either queued column.

The measurement must hold the 113,777-byte token stream and every non-target group byte-identical,
decode through the unchanged shipped receiver, then observe the actual render -> camera lift/R ->
uint8 -> frozen scorers. Weight MSE, raw tensor error, and ancestor-body score deltas are not
substitutes.

## Waterfill and prior-law status

There are **0 measured purchase rows**, so the net-negative set is **UNDETERMINED**, not an empty
measured set. The accounting book contains **0 B of admissible credit**, but that means “nothing may
be booked yet,” not “the residue supplies zero.” The material fraction of the 42,382 B demand is
therefore **UNDETERMINED**, not 0%.

The charter's prior-law prediction — at least one group over 5,000 B coarsens net-negative — is
**INCONCLUSIVE**. Neither its positive condition nor its falsifier fired. RB1's current tested-coder
headroom vector of zero is compatible with this boundary because coder headroom and lossy parameter
purchase are different questions.

## Scorer fire order

**Disposition: `QUEUED-WITH-A-FIRE-ORDER`.** Owner: `MAIN scorer-lane owner`. Consumer store:
`/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_ar1b_archive_residue_purchase/measurement_v2/`.
Fire trigger: MAIN owns the sole n600 scorer lane; exact DX2/archive/receiver/token/GT/upstream hashes
are pinned; a deterministic candidate builder has passed review and local storage preflight; and the
run is launched only through `tools/fire_local_advisory.py`, sequentially in chunks no larger than
120. Every perturbed section, complete archive, repeat, per-class argmax field, Pose6 field,
checkpoint, stdout/stderr, and result JSON must be retained with bytes and SHA-256.

Fire in byte-priority order, but do not interpolate between rungs:

1. Current SM3R extant-tensor one-bit rungs, then one-surviving-row ablations. Do not rerun global
   keep01 or ancestor uncompensated ladders.
2. Direct current CAP1 basis/coefficient/selector one-quantum rungs. Do not rerun rank/refit.
3. Direct current IHS1 row/module one-depth rungs. Do not change the token stream or coder.
4. Residual code/scale perturbations only if distortion attribution remains useful; fixed-layout
   byte credit is already 0 B.

No authority or shipping fire follows from a local advisory row automatically. A candidate can be
promoted only after its own complete archive and exact contest-axis result satisfy the governing
gate.

## RECALL EVIDENCE

The recall searched the full `.omx/research/` memo and receipt corpus by content; the canonical
research indexes; `sub015_DAG_*` FEED blocks; current task, bridge, ledger, hot-state, and durable-run
surfaces; design/SPEC files; the shipping DX2 receiver; and the JSON from
`.venv/bin/python tools/list_canonical_equations.py --json`. Queries included `DX2`, the exact archive
and token hashes, `113777`, `66591`, `42382`, `RX1M`, `IHS1`, `SM3R`, `CAP1`, `RCF1`, `semantic
carrier residual`, `section purchase`, `distortion load bearing`, `mixed q3 q4`, `keep01`, `FiLM
prune`, `carrier rank`, `receiver required`, `DALI GT`, and `per class`.

Findings beyond the charter seeds changed the plan:

1. `.omx/research/ddm_rb1_rate_bound_decomposition_20260822.md` independently recovered the seven
   physical DX2 regions and closed the tested fixed-representation coder supply at 0 B. That changed
   the report from “66,591 B of plausible headroom” to “66,591 B of parameter mass whose purchase is
   unknown.” AR1B adds exact span hashes and does not reopen RB1's coder races.
2. Direct receiver parsing showed the current semantic object is already SM3R mode 6, keep 1%, with
   mixed q3/q4 depths and only two surviving rows in each selected FiLM tensor. The SA3/keep01
   promotion receipts show that state was compensated and admitted. This consumed the obvious
   keep-percent and older mixed-depth suggestions; the queue now names only extant current-body
   one-quantum rungs.
3. MZ2/MP2/SA1/SF1 showed that older semantic gross credits can be pose-toxic and that raw
   tensor/weight effects do not transfer to realized score. Those are routing priors only; they do
   not label current DX2 groups load-bearing without this arm's missing real-path measurement.
4. RI1's current full receiver measured the RC1 whole-body representation catastrophically bad.
   NI1, however, has since built a real 122,250-byte K32 archive and correctly left its distortion
   unmeasured because the same scorer lane is unavailable. The charter's statement that NI1 is
   already dead is stale. AR1B neither duplicates NI1 nor uses it as a negative.
5. MST1's current-instance stage split places 78.7093% of final manufactured Seg errors at the
   native-render observation while R and uint8 are net repairers. That reinforced the requirement to
   score each parameter perturbation through the full real path; the exact share retains MST1's
   `[macOS-CPU advisory intermediate / contest-CUDA terminal]` scope.
6. The canonical equation `score_marginal_lagrange_multipliers_v1` preserves the standard rate law
   and warns that entropy-coded placement is not a uniform raw-byte surface. It supplied no current
   per-parameter price. Within the bounded index/DAG/ledger/corpus scopes, no prior current-DX2
   section purchase table with complete `delta bytes`, `delta d_seg`, and `delta d_pose` rows was
   found.

The charter-owned sibling trees BL1, AE1, LX2, TO2, EF1, LD1, MS9, MST1, and OE1 were read only or
left untouched. Their token/cost/stage fields were not copied, modified, or remeasured.

## Landing status

The required serializer was invoked with the memo's post-edit SHA-256, the message tags
`[no-triality] [p0-ledger-ok]`, and no co-author trailer. Its internal `git add` failed before staging:

```text
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_ar1b_archive_residue_purchase_20260822.md: failed to insert into database
fatal: adding files failed
```

The managed sandbox exposes the Git object/index surface read-only. The staged index remained empty.
This memo is therefore a verified working-tree artifact, **not a commit**; no commit hash is claimed.
The local receipts are intentionally under the gitignored charter receipt root.

## Boundaries

- **Measured here:** exact source identities; eight residue spans with hashes and roles; zero
  remainder; physical section sizes; receiver parse-back sizes; current SM3R/CAP1 logical anatomy;
  DALI GT identities and per-class denominators; scorer-lane ownership; storage routing.
- **Recalled at original scope:** prior coder/recoding closures, older semantic/carrier purchase
  negatives, MST1 stage ordering, RI1 score, NI1 byte closure, and the current keep01 lineage.
- **Not measured:** any perturbed parameter payload, changed archive, candidate byte credit, render,
  per-class scorer output, `d_seg`, `d_pose`, repeat noise, S/B exchange rate, net-negative set,
  waterfill, optimum, new score, or shipping candidate.
- **Negative scope:** no new family negative is claimed. The only new verdict is the exact
  `INSTANCE:DX2_PHYSICAL_CENSUS` and the scoped absence of a prior complete current-DX2 purchase table
  in the searched corpus/index/DAG/ledger surfaces.
- **Payload rule:** this arm materialized no candidate payload. Existing payloads stayed under their
  owners. Future fire requires retaining every candidate and loser; scalar-only output is forbidden.

## LIVE-HYPOTHESES

- One of the still-q4 semantic tensor groups may have a net-negative one-bit rung. It is plausible
  because the current packet still spends 28,530 decoded bytes on q4 tensor values, while keep01 only
  exhausted a different global row-mass axis; current-body realized distortion is unmeasured.
- The two surviving rows in each already-keep01 FiLM tensor may be highly unequal purchases. It is
  plausible because the global keep01 packet was selected by an ancestor ranking and admitted only
  after same-object compensation; neither surviving row has a current isolated real-path price.
- Some IHS1 row/module groups may be distortion-light even though HPAC is decoder-required. It is
  plausible because parser necessity does not prove sensitivity, but the token stream must remain
  byte-identical for the isolation to be valid.
- A narrow current CAP1 value group may trade bytes for less distortion than old rank/refit. It is
  plausible because direct quantization sensitivity is not the post-hoc rank/refit mechanism, though
  the tested family negatives make this lower priority than semantic.

## DEAD-ENDS

- Re-encoding, coder swaps, memoryless bounds, and storage-layout changes for the four sections are
  closed in their tested scopes by #996, #1124, MZ2, BP1, RB1, and the shipped FX5/DX2 wins.
- Treating the 66,591 B residue or any section's gross size as available headroom is closed: size is
  not a measured byte credit and no distortion purchase row exists.
- Global SM3R keep01, older uncompensated FiLM pruning, and dense/sparse/row-dictionary/hybrid
  semantic recodes are consumed or closed; repeating them would not measure a new current-DX2 rung.
- Post-hoc carrier rank/refit is closed at its measured family scope, and another carrier-coder race
  is outside this charter.
- Weight MSE, tensor error, transmitted-label agreement, or interpolation between perturbation levels
  cannot fill the queued scorer columns; RI1/NI1's amplification evidence makes that substitution
  specifically unsafe.
- Calling 0 booked bytes a measured 0% waterfill is closed: the honest status is UNDETERMINED until
  MAIN supplies the scorer lane.

**Own-vehicle frontier: UNMOVED — DX2 remains S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`, archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; AR1B measured no new score.**
