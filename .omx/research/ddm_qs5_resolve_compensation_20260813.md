# DDM QS5 exact-object compensation repair and partial de-trim compile

## Outcome

QS5 produced one real, receiver-consumed, byte-closed candidate and a sealed fire
order for MAIN. It did **not** run SegNet, Modal, or `upstream/evaluate.py`; neither
frontier moved. The only complete-score quantity below is a pre-worker projection,
not a score.

| object | result | authority |
|---|---:|---|
| QS4 worker fact | 100 changed pixels; 34,970 -> 34,953 flips, net benefit 17 | `[contest-CUDA T4, n600] COMPONENT-ONLY` |
| exact QS4 B/H/W split | blocked; retained field is remote and Modal recovery returned `Could not connect to the Modal server` | bounded missing-input result |
| partial de-trim object | 45 token sites = 28 QS4 strict sites + 17 zero-net connective sites; six modeled-negative sites excluded | complete nearest-site model census; **not** a scorer measurement |
| partial de-trim model | 64 B / 7 H / 4 W; modeled B-H = 57 | `TOY-BRACKET` |
| fresh compensation | exact final objects on pairs 105, 517, 523; 98.33%, 99.94%, 99.93% leakage-energy cancellation | `[macOS-CPU advisory frozen CPU-torch PoseNet] NON-PROMOTABLE` |
| local pose delta | d_pose `-6.657906473377261e-9`; pose-score term `-4.012734865240994e-6` | advisory only; excluded from the fire projection |
| direct HP3/RC64 compile | 186,320 B, +68 B versus CP135; primary/repeat byte-identical | `[macOS-CPU exact byte/container]` |
| final Q2C1 overlay compile | **186,278 B**, +26 B; SHA-256 `0911cef621e7b77bfce058d11a72930dc0f284df611ffb822fbeed43453567a1`; repeat byte-identical | `[macOS-CPU exact byte/container and receiver parse-back]` |
| pre-worker projection | Delta S `-3.078193051674021e-5` = Seg model `-4.8319498697916664e-5` + 2x-QS2 pose budget `+2.252354e-7` + exact rate `+1.7312332781176455e-5` | `TOY-BRACKET`; unchanged T4 worker is the verdict |

The projection clears the charter's `-1.5e-5` fire bar. The disposition is
`QUEUED-WITH-A-FIRE-ORDER`, not admitted. Admission after harvest requires realized
matched-instrument Delta S < 0; naming a canonical row additionally requires
`abs(Delta S) >= 1e-5`.

## QS4 field recovery and bounded decomposition

The exact recovery rung was attempted and failed at the Modal API boundary. No scorer
rerun was substituted. The command is:

```text
.venv/bin/modal volume get comma-ddm-js1b-argmax-retained ddm_qs4_dual_axis_20260813_r1/retained/fields/candidate_argmax_n600.npy /Volumes/VertigoDataTier/pact/ddm_qs5_20260813/retained/inputs/qs4_candidate_argmax_n600.npy
```

The expected object is 117,964,928 B with SHA-256
`34a8dad9cb02f2abcfdd3b0f5cae882ec7269fd448ed10aa7a488b11890acfcc`.
Until those bytes are local, only `B-H=17` is measured from the worker receipt; B,
H, and W separately are unknown. The derived 40-flip shortfall is the difference
between the 57-flip nearest-site model and the 17-flip worker result. It does not say
whether trimming removed benefit or created new collateral. The resumable recovery
stage will validate `B+H+W=100` and `B-H=17` over all 100 changed pixels before writing
the exact per-pair JSONL.

## Exact final edit object

The final semantic object uses all three QS4 compile pairs. It keeps every strict
positive site, restores only sites with modeled `B==H` and `W==0`, and excludes all
modeled-negative sites.

| pair | proposal | strict | neutral restored | negative excluded | kept | model B / H / W |
|---:|---|---:|---:|---:|---:|---:|
| 105 | `js6_0000_9fbf75d81c43` | 6 | 14 | 2 | 20 | 8 / 0 / 0 |
| 517 | `js6_0004_06fc74e20d9e` | 10 | 1 | 1 | 11 | 17 / 1 / 0 |
| 523 | `js6_0001_da319a6b65d0` | 12 | 2 | 3 | 14 | 39 / 6 / 4 |

Restoring these 17 sites does not manufacture model value: the model remains 57 net
flips. It changes the exact receiver object so local connective support can survive.
Whether that repairs the 40-flip realization gap is unmeasured and is why the worker
order exists.

## Compensation is now bound to the compiled object

The Schur solve runs inside `compile_candidate` after the final semantic-token arrays
are retained and rendered through the receiver. Each pair gets a content fingerprint
over its pair index, exact semantic-token bytes, exact camera-uint8 master, and CP135
archive identity. The QS1 carrier compiler rechecks that fingerprint before changing
the code lattice. A changed token object with a missing binding, mismatched master,
or stale solve fingerprint now fails closed.

| pair | exact-object fingerprint | fresh int12 delta | local exact d_pose delta |
|---:|---|---|---:|
| 105 | `ce28230ac7b2e5f5d5ce0dd232c7faf63977d86b2412a4bc86b1530b4e329242` | `[2,0,0,1,-3,1,-1,0,-1,-1,1,0]` | `-1.919586422671143e-9` |
| 517 | `a907b7a5970c38ff3fa692ee80f7511bb5b530f0b7e48189498c53f56fddb397` | `[0,0,1,0,0,-1,0,0,0,0,0,2]` | `-4.379358364235235e-9` |
| 523 | `571ad7e2a4a545b581871f9f12fc075f0cf5e842a02207d1653172ed4c8585d7` | `[0,-1,-1,-3,-1,1,-1,-1,4,0,0,0]` | `-3.5896168647088344e-10` |

The old QS4 archive does not satisfy this rule because its frame-0 compensation was
solved for a different frame-1 token stream. Tests prove that changed tokens with a
stale solve are refused and that a matching fresh binding passes.

## Receiver closure and rate

The final code changes 19 int12 coordinates and fits the Q2C1 range. All 12 Brotli
quality candidates were retained; q11 won. The shipped runtime parses the 21-byte
overlay SHA-256
`5d0ea1a2215b763b7a7a30723b5dc100ff2d0b289dfbd1c5d1a30c7288d8d844`
and reproduces the intended 600x12 code lattice exactly. Its token payload SHA-256 is
`7bf4818a76d3b088bc188cfb3bb1cd220ed3e190dbe2c2196cb7aab2fec865d1`.

Both full HP3/RC64 encodes and shipped-decoder parse-backs reproduced the exact
117,964,800-symbol field. The final `archive.zip` and independently rebuilt
`archive.repeat.zip` are byte-identical at 186,278 B. This is 42 B smaller than the
direct variable-CAP1 form for the same exact compensation lattice.

## Projection boundary

The fire projection deliberately ignores the favorable local Pose result. It uses:

- 57 modeled net flips over the explicit 117,964,800-pixel denominator;
- a conservative `+2.252354e-7` score-unit Pose allowance, exactly twice QS2's
  measured `+1.126177e-7` leakage class;
- the exact +26 B archive cost at `25 / 37,545,489` score units per byte.

That yields `-3.078193051674021e-5`. At the same byte and pose allowance, 21 realized
net flips are sufficient for a negative delta and 33 are sufficient for the `1e-5`
super-band. These are derived thresholds, not predictions of the worker result.

## Retention, resumability, and custody

The governed store is `/Volumes/VertigoDataTier/pact/ddm_qs5_20260813/`. It contains
3,386 files and 8,928,382,873 logical file bytes; sparse-filesystem `du` reports 7.3
GiB. All semantic variants, masks, pre-R tensors, exact camera masters, scorer inputs,
Pose batches, Jacobian/cube/descent checkpoints, both full HP3/RC64 runs, every rate
candidate, adapted runtime, final archives, fire inputs, and manifests remain there.
Nothing was deleted or moved.

The runner has an SSD capacity preflight, exclusive lock, atomic stage checkpoints,
and `--resume-from` identity enforcement. A resumed final pass reused and hash-checked
the retained candidate instead of rebuilding or discarding it.

Key receipts:

- `FINAL_RESULT.json`: 88,294 B, SHA-256
  `d72fafec6a5d43789036b618a4f29012975fda90c50b63a12ecf339e4634f5f4`;
- `candidate/COMPILE_RESULT.json`: 71,457 B, SHA-256
  `741885af9b386fc03056f11bb3e3e572777fbdd8caf9eae6b9ba4d1ce667b5db`;
- `SEALED_FIRE_ORDER.json`: 1,871 B, SHA-256
  `ff01db055851a1f0ee736aca9b486a59d67f23628e862a7c132311ef6a9c33d7`.

## Sealed worker order

No scorer lane was claimed or fired. The unchanged dispatcher validated the sealed
request and all three inputs locally. The request is
`/Volumes/VertigoDataTier/pact/ddm_qs5_20260813/fire_order/SEALED_REQUEST.json`
(14,956 B, SHA-256
`e6900e884f5b246bf86045591a8696fd12a51ee9705ea962313af0f4ffbfc317`).

- disposition: `QUEUED-WITH-A-FIRE-ORDER`;
- owner: MAIN sole scorer-lane router;
- consumer store: `/Volumes/VertigoDataTier/pact/ddm_qs5_20260813`;
- fire trigger: MAIN confirms no active n600 scorer lane, the worker self-claims
  `ddm_qs5_resolve_compensation_n600_20260813`, and every sealed SHA verifies;
- scope/cost: one candidate, retained n600 T4 Seg field, official Pose first-six
  vectors, inputs/outputs, and deterministic repeat; approximately $0.16.

The canonical `evaluate.py` follow-on remains unnamed until the worker returns a
negative super-band result.

## RECALL EVIDENCE

Queries before implementation covered `exact object`, `quantize then compensate`,
`stale fit`, `cross-regime transfer`, `nearest collateral`, `same-frame
nonadditivity`, and `re-solve`. Stores consulted included the QS1/QS2/QS4 retained
stores, CP135/pr135 source custody, PZ4 pose-gauge receipts, the canonical research
index, and the live DAG/hot-state/lane surfaces.

Beyond the charter seeds:

- `ddm_rvs1_realization_survival_harvest_20260811.md` requires materializing the
  exact hard-rounded object and re-solving nearby degrees of freedom while holding
  that object fixed;
- `ddm_sf1_stale_fit_genus_sweep_and_structural_fix_20260802.md` classifies a moved
  partner with an old fit as non-transportable and makes re-solving the cure;
- `ddm_pz4p_pose_gauge_preproof_20260811.md` records the roughly 29-fold Pose loss
  from direct quantization and concludes that compensation is part of every hard
  object;
- the DAG precedents require nearest attribution, explicit collateral pricing, and a
  joint remeasurement for same-frame nonadditivity.

These sources changed the implementation: the final token stream is constructed
before the solve; every solve is content-bound and rechecked inside the compiler;
neutral support is only a candidate mechanism; and only one joint unchanged-worker
measurement can admit it. The canonical-equation search found no newer law that
supersedes the exact score marginal or rate denominator used here.

## Verification

- `py_compile` passed for the QS1 engine, QS5 runner, and QS5 tests.
- `ruff check` passed for all three files.
- Focused QS1/QS2/QS4/QS5 tests: **28 passed**.
- The payload-retention detector returned no findings for the changed QS1 engine and
  new QS5 runner.
- Primary/repeat archive SHA-256 values match; receiver parse-back reproduces the
  overlay, code lattice, and token stream.

## Frontier and authority boundary

The exact effective frontier remains CP135:
`S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`, archive SHA-256
`6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
QS5 did not move it. The own-vehicle frontier remains LC2
`S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`.

## NEXT_IF_RESUMED

- `FIRED_LOCALLY_BUT_BLOCKED_BY_MODAL_CONNECTIVITY` — owner: MAIN if connectivity remains unavailable to Codex; consumer store: `/Volumes/VertigoDataTier/pact/ddm_qs5_20260813`; fire trigger: Modal API connectivity is available; action: execute `QS4_FIELD_RECOVERY.json.exact_recovery_argv`, verify the expected bytes/SHA, resume QS5, and consume the exact all-100-pixel B/H/W decomposition without a scorer rerun.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/ddm_qs5_20260813`; fire trigger: no active n600 scorer lane, the QS5 worker self-claim is available, and every sealed request/input SHA verifies; action: execute `SEALED_FIRE_ORDER.json.exact_command_argv` once.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_qs5_20260813`; fire trigger: a complete QS5 worker receipt with retained Seg/Pose components arrives; action: compute realized matched-instrument Delta S, fold if nonnegative or sub-band, and name a canonical exact-eval order only if the negative `1e-5` super-band clears.

## LIVE-HYPOTHESES

- Restoring the 17 zero-priced connective sites may recover enough of QS4's missing
  realization to reach at least 33 net flips. This is plausible because QS4 found
  most harmful cells in neighboring rather than edited cells, while the strict trim
  may have broken local support topology; only the worker can test it.
- Fresh exact-object compensation will avoid QS4's large Pose regression on the T4
  worker. All three local exact solves canceled 98.33%-99.94% of leakage energy and
  locally improved d_pose, but host transfer is unmeasured.
- The exact QS4 B/H/W split may identify a narrower next support rule if this candidate
  misses: the retained worker already fixes B-H at 17 over exactly 100 changes, so
  separating removed benefit from new collateral would directly locate the lost 40.

## DEAD-ENDS

- Compensation repair alone is closed for this candidate: the charter's own prior-law
  estimate remains positive after rate, so QS5 did not fire a compensation-only row.
- Carrying compensation across changed edit objects is closed by construction: the
  compiler now rejects stale or absent content bindings before changing the carrier.
- The direct variable-CAP1 carrier is closed for this exact object: 186,320 B is 42 B
  worse than the receiver-equivalent 186,278 B Q2C1 overlay.
- Inferring exact QS4 B, H, or W from the scalar receipt is closed: only B-H=17 is
  identified without the retained field, and the bounded Modal failure is recorded.
- The favorable local Pose delta cannot promote the archive: macOS-CPU is advisory,
  Seg remains a toy projection, and the exact pointer is unchanged.
