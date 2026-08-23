# DDM HR3 — residual implicit carrier, storage-bounded receiver-closed price

- Date: 2026-08-23
- Arm: `ddm_hr3_residual_implicit_carrier`
- Axis: `[macOS-CPU advisory / scorer-free exact byte measurement]`
- Score claim: `false`
- Pointer moved: `false`
- Verdict scope: `FORMULATION / BOUNDING-NOT-SOLVING-CARRIER-FIT`
- Result: `/Volumes/APDataStore/pact/ddm_hr3_residual_implicit_carrier/RESULT.json`

## Conclusion first

The built width-8/16 Fourier-coordinate residual-action INR **misses**. Its best retained,
receiver-closed row is `w008_m04_b065`: **362,473 residual-equivalent bytes** versus the
**36,858 B** target, and **463,601 total container bytes** versus the **137,986 B** cap. This is
**9.8343x the allowed residual** and is 3,193 B worse than HG1's 359,280 B residual / 460,408 B
container. The learned model changes 1,569 cells, nets only 180 fewer mismatches, and costs 3,223 B
after its winning real coder.

The prior-law prediction is **UNADJUDICATED**, not confirmed: APDataStore fell from 8.9 GiB free to
241 MiB while another writer was active. The charter permits a declared bounding scope reduction,
so full fields stop at width 16; widths 32, 64, 96, and 128 retain counted model-only bounds. It
would be a NO-FAKE error to turn this bounded miss into the charter's requested complete-container
family closure.

No scorer, Metal, Modal, MPS, CUDA, or `upstream/` write occurred. Exact categorical-field identity
and inherited-section identity make the selected row's distortion unchanged from its source object;
this is not a new score measurement.

## Provenance and custody

| object | verified identity |
|---|---|
| V8 reference spec | working-tree SHA-256 `74417253b351f25185106d150fa67dae2b3357aeb33faed982e9b4756e2c4e72` |
| HG1 owning memo | SHA-256 `1ea85c9d0f0be6cb91c7bf300121bca676b941e450e510abd58f3144057c575d`; owning commit `1eb31298ec` |
| ET1 exact-horn memo | SHA-256 `2fba8f45c3bacd0187a791b782f8b67a4d55a533ea5d74512096a300eac2c0ec`; owning commit `3d82e291c3` |
| SY2 pinned composition memo | blob at commit `fe2ba12dc2` SHA-256 `32fc8fcc206bf76cc2631938955f4c64a4bce61325ddbe1601200eb389c98277` |
| SY2 current working copy | SHA-256 `0d909876cb5c987fdf3c7d4eccb2af26050d3ee4bf9795e6291b5d308411646d`; not substituted for the charter pin |
| current DX2 archive | 180,368 B; SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674` |
| source categorical field | 117,964,800 B; SHA-256 `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| HG1 generated field | 117,964,800 B; SHA-256 `2884c5701dc2b2059df0e9f8e4ee3ed81809457b127a48ad3fd3fb6f7a17152b` |
| BL1 cost field | 943,718,400 B; SHA-256 `99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86` |
| HR3 result | 362,945 B; SHA-256 `6ada86ad26b167dc434c3dc0680384e2d7d1ef7a5f1f8d349df4053294bebaeb` |
| HR3 manifest | 877,206 B; SHA-256 `7055e8b1a5ebf152486ed2543042e14e8cd745dfa2617f4353d037e799afc98b` |

The manifest inventories 1,877 retained files totaling 1,837,036,060 B. A current-fact audit
rehashed 739 nested live file facts and found all 739 matching. Three older runner facts deliberately
name historical script bytes at the same mutable path; they remain in `runner_migrations` as
historical provenance and were not misreported as current file facts. The aborted first render's
partial payload also remains retained; nothing was deleted.

## Residual characterization

HG1 differs from the exact categorical field at **1,334,939 / 117,964,800 positions =
1.1316418%**.

| facet | measured result `[macOS-CPU advisory / scorer-free]` | meaning |
|---|---:|---|
| target classes 0 / 1 / 2 / 3 / 4 | 639,336 / 319,147 / 262,741 / 1,331 / 112,384 | 47.89% / 23.91% / 19.68% / 0.10% / 8.42%; not a Lane-only residual |
| Bernoulli event entropy | 0.089399 bits/position | iid occupancy is a poor code by itself |
| target-label entropy | 1.774301 bits/event | class labels remain mixed |
| iid event+label floor | 1,614,315 B | far above HG1's 359,280 B real code, proving strong structure/context already harvested |
| connected components | 131,336 | 52.31% of components are single-pixel, but 89.48% of flips belong to contiguous components |
| distance 0 / below 4 / below 8 from generated boundary | 45.67% / 73.63% / 86.09% | residual mass is boundary-heavy, not flat |
| per-frame mismatch Gini | 0.211095 | much less concentrated across frames than BL1's per-position cost field |
| lag-1 persistence | 47.10% | substantial temporal address reuse |
| same target given lag-1 overlap | 94.36% | persistent residuals usually preserve class identity |
| residual positions in BL1 top 1% | 420,967 = 31.53% of residual | much higher than the 1% area baseline |
| residual incumbent-model cost in BL1 top 1% | 98.806% | residual strongly intersects BL1's expensive object |
| share of all BL1 model bits on HG1 residual | 46.318% | the join is material but not the whole BL1 concentration |

The result is neither flat nor an easy low-dimensional trajectory. It is spatially and temporally
structured, boundary-heavy, and extremely cost-concentrated on BL1's top positions, while still
containing 131,336 components and all five target classes. That supports trying structured/hybrid
carriage, but it does not make an unpriced implicit win.

Retained characterization payloads:

- residual mask: 14,745,600 B, SHA-256 `81132b007ad974e3075964b5938c765daa8b9223a0e57e9bdd8e49b1387fd76c`;
- BL1-overlap mask: 14,745,600 B, SHA-256 `f67d81e1462824be694ab1e742d0e814c7d7a316ae3fcb1615b39daaa855d23e`;
- component rows: 223,873 B, SHA-256 `8340cdf568020220289c285e6d21b972e045be45fb4f6c7cfdf4fbdca6bff387`;
- frame rows: 78,999 B, SHA-256 `fbd6b7854820e4f26c5e5b3beca2653c41f26c87cb0e439dd392d4ebd4e30dfd`.

## Carrier-form accounting

| status | form | what is and is not established |
|---|---|---|
| `DERIVED` | generators-not-boundaries; base-class-conditioned coordinate carrier; deterministic Fourier coordinates; counted per-pair modulation; six per-class actions whose argmax implies cells; exact unique-home residual | Derived from the V8 implicit-cell/per-class constraints plus the complete predictor+residual accounting law. The particular residual-action INR is HR3's derivation, not a measured V8 transfer. |
| `SPEC_ONLY` | V8's edge-centric Road/Undrivable bulk carrier, Lane carrier, Movable silhouette carrier, and class-specific merge/diff/correct reconciliation | Preserved as the optimal reference. HR3 did not fabricate these residual-specific carriers or claim they were built. |
| `BUILT` | deterministic CPU INR widths 8/16 full-field; widths 32/64/96/128 trained, checkpointed, quantized, serialized, and real-coder-priced as model-only bounds | Every learned parameter, including 600 per-pair modulation rows, is counted. Generic Fourier generation and interpreter code are free. Only width 8/16 have real residuals and complete containers. |

All six training widths retained initialization plus four distinct epoch checkpoints. Weighted
cross-entropy at epoch 4 fell from 0.255400 (width 8) to 0.230395 / 0.210225 / 0.192103 / 0.181901 /
0.178604 for widths 16/32/64/96/128. That training trend is not a byte result and does not fill the
unmeasured width-32/64/96 residual cells.

## Real-coder rows

Each full row races eight residual orderings and Brotli q11, zlib 9, and LZMA2 extreme. Each model
also races all three coders. The target in every row is **residual-equivalent <=36,858 B** and total
container **<=137,986 B**.

| row | model B | remaining residual B | residual-equiv B | vs 36,858 B | container B | vs 137,986 B | status |
|---|---:|---:|---:|---:|---:|---:|---|
| w8 b3.5 | 3,223 | 362,400 | 365,697 | +328,839 | 466,825 | +328,839 | built, packet-parsed |
| w8 b4.5 | 3,226 | 363,512 | 366,812 | +329,954 | 467,940 | +329,954 | built, packet-parsed |
| w8 b5.5 | 3,226 | 359,988 | 363,288 | +326,430 | 464,416 | +326,430 | built, packet-parsed |
| **w8 b6.5** | **3,223** | **359,176** | **362,473** | **+325,615** | **463,601** | **+325,615** | **receiver-closed exact** |
| w16 b3.5 | 4,070 | 366,088 | 370,232 | +333,374 | 471,360 | +333,374 | built, packet-parsed |
| w16 b4.5 | 4,057 | 371,576 | 375,707 | +338,849 | 476,835 | +338,849 | built, packet-parsed |
| w16 b5.5 | 4,057 | 362,972 | 367,103 | +330,245 | 468,231 | +330,245 | built, packet-parsed |
| w16 b6.5 | 4,057 | 360,184 | 364,315 | +327,457 | 465,443 | +327,457 | built, packet-parsed |

For the winner, the model race is Brotli 3,223 B / LZMA2 3,320 B / zlib 3,375 B. The selected
`tile64_time` residual race is LZMA2 359,176 B / Brotli 363,214 B / zlib 417,274 B. The model and
residual therefore choose different coders; a nominated-coder result would have been wrong.

| model-only row | counted model B | model coder | residual target | typed boundary |
|---|---:|---|---:|---|
| w32 b3.5 | 9,012 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w32 b4.5 | 9,001 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w32 b5.5 | 9,001 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w32 b6.5 | 9,001 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w64 b3.5 | 16,339 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w64 b4.5 | 16,339 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w64 b5.5 | 16,339 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w64 b6.5 | 16,339 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w96 b3.5 | 26,608 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w96 b4.5 | 26,632 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w96 b5.5 | 26,632 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w96 b6.5 | 26,632 | Brotli q11 | 36,858 | model-only; residual/container unmeasured |
| w128 b3.5 | 39,481 | Brotli q11 | 36,858 | model alone exceeds gate by 2,623 B |
| w128 b4.5 | 39,580 | Brotli q11 | 36,858 | model alone exceeds gate by 2,722 B |
| w128 b5.5 | 39,514 | Brotli q11 | 36,858 | model alone exceeds gate by 2,656 B |
| w128 b6.5 | 39,514 | Brotli q11 | 36,858 | model alone exceeds gate by 2,656 B |

Widths 32/64/96 are live unmeasured cells, not wins: even width 96 leaves only about 10.2 KB for
the residual and framing. Width 128 is closed under this serialization because the counted model
alone exceeds the complete residual budget.

## Receiver closure and both currencies

The best complete archive is 463,601 B, SHA-256
`665c1491306d5cfcf8fe90612ffed75ed9be07491f33282663e683d0c490add1`; its byte-identical repeat has
the same size and SHA. The 410,518 B HR3 packet SHA is
`2050a0bfec4e46bef7c3e6835bf05557ed61a5201538742d336eec9c464a3e93`. Archive parse-back and a second
direct receiver repeat both produce 117,964,800 bytes with categorical SHA
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`. The receiver-generated base
and implicit field also match their retained identities.

- Fixed current distortion: 463,601 B is **325,615 B over** 137,986 B, or **+0.216813678285 S** at
  `6.658590e-07 S/B` from `ddm_tx1_toolbox_crosswalk_20260819.md` section 0.
- Zero distortion: the current 180,368 B archive would need to shed 150 B to reach 180,218 B. HR3
  instead adds 283,233 B, leaving a **283,383 B shortfall** to that zero-distortion byte cap.

## Prediction adjudication

The preregistered prediction was that the residual would not reach 36,858 B and that the container
would close completely. The measured number is 362,473 B, but the disposition is
**UNADJUDICATED_SCOPE_REDUCED_AT_WIDTH_BOUND**. The width-8/16 formulation is a strong negative; the
full reference range through width 96 was not measured. ET1, HG1, and prior lossy rows remain their
own valid negatives, but HR3 does not supply the missing every-horn closure.

## RECALL EVIDENCE

Queries covered `implicit|residual|predictor.*correct|generator.*residual|evaluator.cell|per-class`,
the exact DX2/HG1 object hashes, and `procedural_predictor_plus_residual_correction_savings_v1`
across `.omx/research/`, local arm receipts, canonical research indexes, `sub015_DAG_*` FEED blocks,
design specs, task ledgers, and `src/tac/`; canonical equations were also listed with
`tools/list_canonical_equations.py --json` and filtered on this surface.

Beyond the charter seeds, the search found:

- `ddm_ig1_implicit_carriage_gestalt_20260821.md`, which refutes universal implicit dominance and
  requires complete receiver-closed routing among implicit, grammar, and hybrid forms. This changed
  HR3 from a model-byte comparison into complete model+residual+framing accounting.
- `procedural_predictor_plus_residual_correction_savings_v1`,
  `v8_geometric_rate_decomposition_v1`, and `fullstack_unique_home_assignment_v1`, which require all
  predictor, residual, and framing bytes to have one home. This produced `residual-equivalent bytes`
  rather than subtracting model savings from HG1 informally.
- Older residual-INR and JS3 counted-module work, which concerns different vehicles/objects and did
  not transfer a current-DX2 number. It changed those rows to context only.
- AE1/OE1's anti-predicted escape closure. HR3 is a different probability object, but the shadow
  made exact retained correction and receiver closure mandatory.

Did not find a current-DX2, receiver-closed learned carrier for HG1's exact residual in the bounded
research/index/DAG/task/code/HG1/ET1 scopes. This is scoped absence, not global nonexistence.

STORES CONSULTED: `.omx/research/`; `.omx/tmp/arm_receipts_local/`; canonical equation registry via
`tools/list_canonical_equations.py --json`; `CANONICAL_RESEARCH_INDEX*`; `sub015_DAG_*`; design specs;
task-ledger/state surfaces; `src/tac/`; `/Volumes/APDataStore/pact/ddm_hg1_heterogeneous_analytic_generator_gate/`
(read-only); `/Volumes/APDataStore/pact/ddm_et1_edge_topology_container_gate/` (read-only);
`/Volumes/APDataStore/pact/ddm_dx2/r7/` (read-only);
`/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation/measurement_v1/` (read-only);
Metal: none; Modal: none; scorer: none.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — disposition: resume width-32/64/96 full-field rendering, all eight
  residual orderings, all three real coders, and selected receiver closure; owner: MAIN-assigned HR3
  successor; consumer store:
  `/Volumes/APDataStore/pact/ddm_hr3_residual_implicit_carrier/manifest.json`; fire trigger:
  APDataStore has at least 6 GiB free with no competing bulk writer, then resume with the retained
  manifest and `--full-field-max-width 96` after adding an explicitly reviewed scope-expansion
  migration. No scorer fire unless a receiver-closed residual reaches <=36,858 B.

## LIVE-HYPOTHESES

- Widths 32/64/96 remain worth one retained test because their counted models fit under 36,858 B and
  training loss keeps falling; plausibility is limited by the width-8/16 negative and width 96's
  roughly 10.2 KB residual headroom.
- A sparse class/edge/temporal hybrid may beat this dense shared INR because 98.806% of residual
  incumbent cost lies in BL1's top 1%, 94.36% of lag-1 overlaps preserve target class, and 86.09% of
  events lie within eight pixels of a generated boundary.

## DEAD-ENDS

- Width-8/16 Fourier-coordinate, per-pair-FiLM residual-action INRs are closed on this exact
  formulation: all eight complete rows are worse than HG1 after model, residual, and framing bytes.
- Width 128 is byte-dead under this serialization: its smallest counted model is 39,481 B before one
  residual or framing byte, already above 36,858 B.
- A nominated coder is closed: Brotli wins the model while LZMA2 wins the residual; zlib loses both.
- Large float32 NumPy `matmul` on this environment is closed as a receiver path because it emitted
  non-finite arithmetic at >=128 rows; deterministic finite `einsum` replaced it and passed receiver
  identity.
- Complete container-family closure is not an admissible conclusion from this run because widths
  32/64/96 were not full-field priced.

Own-vehicle frontier unchanged: **dx2 — S `0.14821987563243377` @ `180,368 B` `[contest-CUDA T4, n600]`**, archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.
