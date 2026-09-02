# QXR1/QXO1 born-realization binding — the exact under-gate field is retained, but the current RGB renderer does not consume either new semantic section

**Verdict: `BOUND-SCORER-FREE / QUEUED-WITH-A-FIRE-ORDER / NO FRONTIER MOVE`.**

QXR1 reproduced QXO1's exact 129,309-byte archive receiver output over all 600
pairs and retained the scorer inputs under the BR2 payload contract. The fresh
decode produced the pinned 117,964,800-byte overwrite field exactly. Its diff
from the QBT baseline is exactly 8,749 sites: 9,177 of the original 17,926
events are genuine target no-ops and are absent from the grammar.

The decisive binding result is causal rather than a distortion estimate. The
model, latent metadata, and all 600 latent records that determine the inherited
QBT RGB renderer are byte-identical to the BR2 packet. QXO1 section 8 changes a
semantic class field and section 7 carries a pose stream, but neither section is
read by that RGB renderer. QXR1 therefore retained the exact current camera-grid
scorer inputs without inventing a class palette or claiming that semantic bytes
changed RGB. No BR2 distortion number is attached to QXO1 bytes in this memo.

## FULL SHA RESOLUTION

| Object | Bytes | Full SHA-256 | Status |
|---|---:|---|---|
| QXO1 `RESULT.json` | 53,513 | `b7b9dd4fb1dbb70aa6dd41a32a6b998c30588103c0d2a8184d71c6ff9147a80a` | source authority matched |
| exact `archive.zip` | 129,309 | `2487f5150fd3c38087fb5ada48d00e953c7d88a8a7219e29fbf53420657bb07f` | matched and retained |
| complete eight-section QXE | 129,185 | `2308820b56b29abef69556bbd98e12758cdf7e3adc6f214fe38b83ab0066a6d6` | receiver-extracted identity |
| seven-section core QXE | 113,720 | `4e6a2f6669c590258fc6c5d194ae6cb30951f5881e2055761de0bff753bdfb95` | matched and retained; ZIP envelope price is 113,844 B |
| raw QXO1 grammar | 18,559 | `bb6c1b8626f06632ee1b3f2d6088a25d85e6d7db3c4d00b258686418b67c85ea` | matched and retained |
| Brotli-q11 grammar payload | 15,417 | `b0c68d2226febf336521d454fa13a9c0fa324a14d2b1cb14ab54038b89de34f2` | matched source payload |
| decoded overwrite records | 43,745 | `13e5b7419a1873c6543075d1fde4347644247fae25fc46d281450ad244cd2ee9` | fresh receiver output retained |
| QBT baseline field | 117,964,800 | `afeb8c94d5181b03992aefad1daef49ee7aaf1f768d11aa5964dacbfa1e22dbd` | matched and retained |
| QXO1 overwrite field | 117,964,800 | `9079929d004cc9638a80159d61371c2982c198f0eb2b19eac4084da981ababc7` | fresh decode matched and retained |
| reassembled QBT model | 87,854 | `2280c2d3c54d1781559ec130123a05ec664dbdf347b04f379805bfbe67f59085` | byte-identical renderer input |
| QBT latent metadata | 16 | `79128a18dec7177dcd9b6922f261f1f6dc3b637b27d04c39baae8c4fed0af2b2` | byte-identical renderer input |
| QBT n600 latent state | 31,206 | `ff7db019f3d774da8abf20a79c9bba4df7b2b73d277ac09ba4feedd0505df9d2` | byte-identical renderer input |
| separate pose stream | 7,208 | `9142ab46a65d7ef9b62bcf98d789ea9741212f163d16940fe284a3786e16bf4b` | retained; not consumed by current RGB renderer |
| BR2 QBT packet | 106,724 | `8c26684d33313ca44f3d4f02cf3c369f0f33d6de37eeba42ae4220faed3e6d38` | renderer-state comparator matched |

The byte arithmetic is exact: 15,417 B grammar section, 113,844 B core ZIP
price, and 129,309 B complete archive. The largest legal sub-0.12 archive is
137,985 B, so QXO1 is 8,676 B below that rate gate. This remains a rate fact,
not a distortion or score fact.

## BINDING RUNNER AND RETAINED RECEIPTS

The new runner is
`experiments/ddm_qxr1_qxo1_realization_binding.py`. Its scorer-free
`prepare` path:

1. pins the QXO1 result, archive, baseline, reference field, grammar payload,
   receiver sources, BR2 result, and BR2 packet by SHA-256;
2. parses every QXE envelope and integrity field, reassembles the QBT tensor
   groups, and proves the model/latent identities;
3. decodes the QXO1 grammar through the QXO1/QX2 receiver functions and writes
   a fresh output field;
4. retains the archive, core, raw grammar, model, latent metadata, latent state,
   pose stream, baseline field, decoded overwrite records, and decoded field;
5. writes 20 resumable 30-pair NPZ chunks containing exact camera-grid inputs,
   baseline classes, overwrite classes, and mutation masks; and
6. writes a typed binding receipt and a sealed MAIN-owned scorer fire order.

No scorer module is imported by module import, `prepare`, or `fire-order`.
The scorer-bearing imports sit inside the separately gated `score` action after
`--launch-authorized`, exact resume-root, fresh active-lane-claim, no-conflict,
and AP free-space checks. A negative smoke invocation without
`--launch-authorized` failed closed before any scorer import.

| Receipt | SHA-256 | Key fact |
|---|---|---|
| `/Volumes/APDataStore/pact/ddm_qxr1_qxo1_born_realization_binding/BINDING_RESULT.json` | `035300d177406a778e3bf8581d6102731a528a977f26aeecc9329f2ceba2ea93` | full n600, 20 render chunks, `scorers_loaded=0` |
| `/Volumes/APDataStore/pact/ddm_qxr1_qxo1_born_realization_binding/FIRE_ORDER.json` | `d1c177a0ee159bb59798ce1b65d717de37e2b1a0750ffce56f5345a5d2d68acc` | sealed queue command and trigger |
| runner source at receipt time | `9227f4c39d600e68672f740c63dc21ba650d80fa92bc483054479f553d06f084` | final reviewed source |

The AP custody tree contains 55 non-AppleDouble files and 565,871,398 logical
bytes. Every materialized payload is retained. Storage preflight observed
42,231,922,688 free bytes against the 2,500,000,000-byte prepare minimum. No
cleanup or deletion was performed.

Validation passed:

- `python -m py_compile`;
- Ruff with zero findings;
- `git diff --check`;
- scorer-free import assertion (`tac.scorer` absent from `sys.modules`);
- complete prepare/resume validation over all 20 payload chunks;
- fresh QXO1 decode equality against the pinned field SHA;
- launch-authorization negative smoke; and
- two genuine post-final-edit `review_tracker.py mark-file` passes,
  `qxr1-final-pass1` and `qxr1-final-pass2`, over all 23 tracked Python
  entities. Earlier review identified source-pin, resume-semantic, exact-camera,
  and pose-stream-consumption gaps; the final passes rechecked the corrected
  source, including byte comparison of resumed camera arrays against BR2.

## FIELD-DIFF PRE-READ

Axis: `[scorer-free exact receiver/render-input binding]`. Selection mode:
full n600. Denominator: 117,964,800 class sites. Mutations: 8,749
(`0.00007416619194878473` of sites), with 2–42 mutations per pair, mean
14.5816666667, and zero mutation-free pairs.

### By source and target class

| Class | Source sites overwritten | Sites written to target |
|---|---:|---:|
| Road | 4,850 | 2,862 |
| Lane | 21 | 2,536 |
| Undrivable | 1,669 | 1,600 |
| Movable | 932 | 776 |
| MyCar | 1,277 | 975 |
| **Total** | **8,749** | **8,749** |

The largest directed transitions are Road→Lane 2,467, MyCar→Road 1,262,
Undrivable→Road 1,138, and Road→Undrivable 1,136. The full transition table is
in `BINDING_RESULT.json`.

### By image region

Regions are equal normalized 3×3 image bins (`v0` top, `v2` bottom; `h0`
left, `h2` right).

| Region | Mutations |
|---|---:|
| middle-left `v1_h0` | 1,934 |
| middle-center `v1_h1` | 1,855 |
| middle-right `v1_h2` | 2,437 |
| bottom-left `v2_h0` | 743 |
| bottom-center `v2_h1` | 960 |
| bottom-right `v2_h2` | 820 |
| all top-row bins | 0 |
| **Total** | **8,749** |

This localization makes any future field-consuming realization attributable.
It does not imply a current RGB effect. In the present receiver, both changed
sections are causally downstream-dead with respect to RGB: section 8 is not a
renderer input, and section 7 is likewise not used to produce PoseNet's input
frames.

## CLOSED-FORM-FIRST BINDING

No fitted model, proxy, sampled estimate, or surrogate was used. QXE parsing,
hash equality, deterministic overwrite application, and the exact source-level
renderer input set decide the binding. The scoring chain remains the frozen
piecewise-analytic chain in CFA1, but a semantic partition has no unique RGB/YUV
preimage. Closed form therefore proves non-consumption; it cannot manufacture a
missing learned/joint realizer.

The preregistered adverse prediction remains recorded exactly as requested:
realized QXO1 `d_seg` remains order 0.17. Its falsifier is `d_seg <= 0.01` with
`d_pose <= 1.25e-4` on this exact binding. QXR1 did not run either scorer, so
the prediction is still formally untested by this arm. The byte-identity result
does, however, make a materially better result implausible on the present
renderer: section 7 and section 8 cannot affect a function that does not read
them.

## SEALED MAIN FIRE ORDER

- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER`.
- **Owner:** MAIN n600 local scorer-realization scheduler.
- **Consumer store:** `/Volumes/APDataStore/pact/ddm_qxr1_qxo1_born_realization_binding/SCORER_RESULT.json` plus retained scorer chunks under `retained/scorer_outputs/`.
- **Expected wall:** approximately 485 seconds, using BR2's measured 479.663-second realization and the charter's 484.769-second comparator.
- **Chunking:** 30 pairs per chunk, below the contract maximum of 120.
- **Components retained:** camera input references, SegNet logits/argmax/targets, PoseNet first-six/targets, per-pair rows, `d_seg`, `d_pose`, both nonlinear score terms, exact rate, and recomputed `S`.
- **Fire trigger:** MAIN verifies the newest relevant scorer row is terminal; appends a fresh unique active `local_macos_cpu` claim for `ddm_qxr1_qxo1_scorer_20260902`; confirms no other active scorer claim newer than 24 hours; confirms AP free bytes are at least 1,500,000,000; and rematches archive `2487f5150fd3c38087fb5ada48d00e953c7d88a8a7219e29fbf53420657bb07f`, field `9079929d004cc9638a80159d61371c2982c198f0eb2b19eac4084da981ababc7`, core `4e6a2f6669c590258fc6c5d194ae6cb30951f5881e2055761de0bff753bdfb95`, grammar `bb6c1b8626f06632ee1b3f2d6088a25d85e6d7db3c4d00b258686418b67c85ea`, and every prepared render-input fact.
- **Exact command:** `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python experiments/ddm_qxr1_qxo1_realization_binding.py score --output /Volumes/APDataStore/pact/ddm_qxr1_qxo1_born_realization_binding --resume-from /Volumes/APDataStore/pact/ddm_qxr1_qxo1_born_realization_binding --scorer-claim-id ddm_qxr1_qxo1_scorer_20260902 --launch-authorized`

This order does not grant this arm the scorer lane and was not fired. It measures
the current QXO1 archive's exact inherited render path on its own retained
inputs. It must not be described as evidence that the overwrite or pose
sections influence RGB; a new realizer is required for that claim.

## RECALL EVIDENCE

The full-corpus pass searched research memos/receipts by content with
`QXO1|QX1|QBT|born object|realization|preimage|field.*consum|fixed.*paint|palette`,
the canonical equation registry with
`realization|preimage|decoder|causal|palette|quotient|predict.*project|score`,
the canonical research indexes and `sub015_DAG_*` surfaces, design/SPEC files,
`main_hot_state.md`, and task-status/ledger surfaces for
`1372|1374|1376|qxo1|qxr1|born realization|target overwrite`.

Beyond the charter seeds, the pass found:

- `ddm_nx1_next_object_route_20260831.md` says the admissible QX1 object must
  generate its partition implicitly **and** choose the RGB/YUV preimage jointly;
  it explicitly distinguishes that object from QBT's inherited palette
  realization. This forbade treating the decoded field as already realized.
- `ddm_no2_quotient_born_object_20260827.md` records the n600 fixed-palette
  failure: small movable-mask error became about 1,700× worse through
  paint/R/SegNet. This forbade inventing a flat class paint to satisfy the
  charter mechanically.
- Canonical equations
  `predict_project_realization_admissibility_v1`,
  `palette_realization_ceiling_context_dominated_v1`, and
  `textured_power_diagram_sufficient_statistic_v1` agree that cells plus
  pose tubes and scorer-legible texture must survive the receiver; a partition
  alone is not the scored sufficient statistic.
- The current hot-state task #1374 is the separate SCMDL build chain, not a
  pre-existing QXO field-to-RGB consumer. It changed no implementation choice
  here.
- In the searched indexes/DAG/design/SPEC/task-ledger scope, no existing
  receiver that consumes QXO1 sections 7–8 into RGB was found. The source audit
  then positively established the current QBT renderer's input set.

These findings changed the plan: QXR1 bound and retained the inherited camera
inputs, added an explicit causal/non-consumption receipt, and refused both a
palette substitute and a distortion transfer. `upstream/` remained read-only.

## MEASURED AND NOT MEASURED

Measured `[scorer-free exact receiver/render-input binding]`: all full hashes
above; complete n600 receiver equality; exact 8,749-site class/region diff;
model/latent byte identity; current renderer non-consumption; 20 retained
camera-grid chunks; storage and resumability gates.

Not measured: SegNet, PoseNet, `d_seg`, `d_pose`, distortion, advisory score,
contest score, CPU/CUDA parity, or promotion eligibility. No scorer, Modal,
Metal, training, remote dispatch, exact contest evaluation, or pointer update
occurred.

## NEXT_IF_RESUMED

- **Disposition: `QUEUED-WITH-A-FIRE-ORDER`; owner: MAIN n600 local scorer-realization scheduler; consumer store: `/Volumes/APDataStore/pact/ddm_qxr1_qxo1_born_realization_binding/SCORER_RESULT.json` plus `retained/scorer_outputs/`; fire trigger:** satisfy the fresh-lane, no-conflict, free-space, full-hash, and prepared-chunk conditions in `FIRE_ORDER.json`, then run its exact command once and harvest both Seg and Pose components.
- **Disposition: `HOLD-CONDITIONAL`; owner: future QX joint-preimage builder assigned by MAIN; consumer store: a new separately chartered counted archive/receiver store, not the QXR1 result; fire trigger:** only after the current scorer order is terminal and a section-complete receiver design proves that sections 7–8 causally generate RGB/YUV through R without fixed-palette painting, hidden video-derived code, or uncounted learned state.

## LIVE-HYPOTHESES

- The queued same-object scorer run will reproduce BR2's component behavior to deterministic precision. This is plausible because every model/latent renderer input and every retained camera-grid scorer input is byte-identical, while QXO1's changed sections are not renderer inputs. The run is still owed because this arm did not invoke the frozen scorers and distortion transfer is forbidden.
- A genuinely joint field/pose-consuming preimage may make the 129,309-byte rate opening useful. This is plausible because QXO1 has 8,676 B of rate margin and its mutations concentrate in the middle/bottom scorer-relevant scene region, but the realizer must be jointly trained/solved and counted; the present receiver provides no evidence of such transfer.
- Co-designing the core with the overwrite grammar may reduce the 15,417-byte section. This is plausible because rank gaps dominate the raw grammar body, but it is a new core and must remeasure model bytes, realization, and exact rate together.

## DEAD-ENDS

- Current-QBT-renderer consumption of QXO1 semantic writes is closed at **INSTANCE** scope: section 8 is not in the renderer input set, so all 8,749 field mutations have zero causal path to the retained camera frames.
- Current-QBT-renderer consumption of the counted pose stream is closed at **INSTANCE** scope: section 7 is retained but does not generate PoseNet input pixels.
- Flat class-palette realization is closed at **FORMULATION** scope by the recalled n600 context/texture ceiling; it cannot be used to manufacture a QXO1 score.
- Cross-object arithmetic remains closed: BR2 distortion cannot be combined with QXO1 bytes. QXR1 records byte-identical renderer inputs but reports no distortion or score until the queued scorer consumer runs.
- Repricing or reserializing the same grammar cannot repair realization. The live defect is causal consumption of sections 7–8, not the already-under-gate section price.

Own-vehicle frontier: **AFR1 S `0.14797617125559104` @ `180,002 B` `[contest-CUDA T4 n600]`, archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` — UNMOVED.**
