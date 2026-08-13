# DDM QS2 compensation rate rung — 2026-08-13

## VERDICT

**QUEUED-WITH-A-FIRE-ORDER.** The exact QS1 six-pair compensation object now
costs **+34 archive bytes = 5.6666667 B/pair** instead of +77 B = 12.8333
B/pair. This is a real retained archive/container measurement on
`[macOS-CPU exact byte/container measurement]`, not a score. Holding QS1's
already measured matched T4 component deltas fixed because the semantic,
token, and signed-int12 code lattice are byte-/value-identical, the
pre-encoded complete-score change is **-4.374917893846169e-6 S** and the
realized-flip density is **32/34 = 0.941176 flips/B**, above the 0.785
breakeven law. A fresh T4 receiver run remains the verdict.

No Modal job was fired. No SegNet field was recomputed. The canonical pointer
did not move, and this arm did not establish a new exact score.

## MEASURED RATE CURVE

The denominator is 48 retained candidates: four dead-zone steps times all 12
Brotli qualities. Selection was minimum exact deterministic `archive.zip`
bytes, with lower quality breaking an exact byte tie. Every overlay, raw
carrier source, compressed carrier stream, split-model section, archive member,
and archive is retained under
`/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/retained/rate_race/`.

| compensation row | active pairs | nonzero coordinates | overlay bytes | best Brotli | archive bytes | delta vs CP135 | bytes/active pair | local Pose delta S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact step 1 | 6 | 24 | 31 | q11 | 186,286 | +34 | 5.6667 | +7.821817e-7 |
| dead-zone step 2 | 5 | 8 | 22 | q11 | 186,261 | +9 | 1.8000 | -1.099897e-6 |
| dead-zone step 3 | 2 | 2 | 12 | q11 | 186,251 | -1 | -0.5000 | +2.468821e-5 |
| dead-zone step 4 | 1 | 1 | 9 | q11 | 186,245 | -7 | -7.0000 | +2.895237e-5 |

The byte columns are `[macOS-CPU exact byte/container measurement]`. The Pose
column is `[macOS-CPU advisory frozen CPU-PoseNet, six changed pairs over
n600]`, computed against base `d_pose = 0.0001474653494795297`. Negative archive
deltas at steps 3 and 4 are container interactions, not negative payload sizes.

The coarsest locally Pose-admissible row is step 2. It was not compiled as the
fire candidate because it changes the code lattice and its favorable local Pose
result has no matched T4 Seg/Pose measurement. The local exact-step Pose result
also disagrees in sign/magnitude with the retained QS1 T4 component result
`+1.126177e-7 S`; that is direct evidence not to promote the local ordering.
The fired candidate is therefore exact step 1: it already clears the rate target
and preserves the only lattice with matched T4 components.

## COMPILED CANDIDATE AND ADMISSION

- Candidate: `/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/candidate/archive.zip`
  = **186,286 B**, SHA-256
  `0bb74f1d16e81138975a46c7d4f91f08e1d9fb574930255eac87ccd69f3f8b03`.
- Independent deterministic archive repeat is byte-identical.
- Joint sparse overlay: **31 B**, SHA-256
  `d36255e71057287eea16d9435b5bcd32968e85109c5e05c56f57041e48c04665`.
  It carries sorted 10-bit pair indices, 12-bit support masks, and the measured
  signed three-bit delta domain `[-3, 4]`; aliases, truncation, nonzero padding,
  out-of-domain codes, and signed-int12 overflow fail closed.
- Receiver-applied code lattice: SHA-256
  `f896bd977d0da0a03dc11c3ea1a9988ef5bed0755dc1eac80bd96e1087d380d3`.
  It equals the sealed QS1 code lattice exactly.
- Adapted runtime tree: SHA-256
  `5bff7f9d7194ca8c161633a0183a7be5f0ae3e7b7a5d57dd6e1d344423af447f`.
  Runtime parse-back recovered the same 31-byte overlay SHA while preserving
  the token, semantic, and HPAC stream hashes.
- Admission arithmetic, with no exact-score claim:
  `-2.712674e-5 Seg + 1.126177e-7 Pose + 34*(25/37545489) rate`
  `= -4.374917893846169e-6 S`. If the fresh receiver preserves the matched QS1
  components, the projected value is `0.16195076336034792`; this is a
  projection, not a measured row.

The candidate and runtime are receiver-closed through strict archive parsing
and exact code-lattice application. Full decode/render and both scorer
components remain unmeasured on CUDA because this arm did not own the scorer
lane.

## PER-PAIR POSTMORTEM AND REALIZATION LEVER

**BLOCKED_MISSING_RETAINED_FIELD_LOCAL_COPY.** The required candidate argmax
field exists in Modal volume `comma-ddm-js1b-argmax-retained` at
`ddm_qs1_dual_axis_20260813_r2/retained/fields/candidate_argmax_n600.npy`,
expected SHA-256
`ad1e3dcc0a57c53f0757773a018335924afc26992f398c23ec084eecace7ed20`.
Modal DNS was unavailable, and no byte-identical copy was found in the repo,
`/Volumes/VertigoDataTier/pact`, or `/Volumes/APDataStore/pact`.

Therefore per-pair realized flips, per-pair changed pixels, reverted-edit class,
calibrated waterfill order, and realization-efficiency engineering curve were
**not computed**. The retained whole-candidate denominator remains 189 changed
pixels, 32 net flips, 16.9312% realization efficiency, and 0.415584 flips/B at
the old +77-byte closure. Re-running SegNet to recreate the field was explicitly
forbidden and was not done. The typed blocker, exact download argv, source SHA,
and local destination are in
`/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/PER_PAIR_POSTMORTEM_BLOCKER.json`.
This is bounded absence in the searched local stores, not a claim that the field
does not exist.

## SEALED T4 HANDOFF

The only live order is
`/Volumes/VertigoDataTier/pact/ddm_qs2_20260813/SEALED_FIRE_ORDER.json`, SHA-256
`8c0c0c48647c949c3fcd72ea271e8b02123d7109d0b2067b3313067cafd6015b`.
It points to request `ddm_qs2_dual_axis_20260813_r2`, SHA-256
`aedcd98e49633d24e682724daae733c4536c90dbc15b8ac8747f2bef626b9949`.
The unchanged dispatcher accepted the sealed input census locally. The request
keeps the worker-required `local_pose_delta=0.0` and `pose_unmeasured=true`
transport placeholders; the retained advisory/prior screen is a separate input,
and the worker measures fresh Pose vectors. The dispatcher self-claims its lane.

The superseded R1 request at `fire_order/SEALED_REQUEST.json` is invalid for the
unchanged worker and MUST NOT be fired. Its durable tombstone is
`fire_order/DO_NOT_FIRE_R1.md`, SHA-256
`d1bdc84293b878250d378a55cae82ee2bb6641a57642ebd4cdd5cdca55948cc6`.

Exact argv:

```text
.venv/bin/modal run --detach experiments/ddm_qs1_modal_t4_dual_axis.py::main --sealed-request /Volumes/VertigoDataTier/pact/ddm_qs2_20260813/fire_order/SEALED_REQUEST_r2.json --fire-input-dir /Volumes/VertigoDataTier/pact/ddm_qs2_20260813/fire_order/fire_inputs --expected-request-sha256 aedcd98e49633d24e682724daae733c4536c90dbc15b8ac8747f2bef626b9949 --output-dir /Volumes/VertigoDataTier/pact/ddm_qs2_20260813/dispatch/ddm_qs2_dual_axis_20260813_r2 --detach --provider-detach-ack
```

## VERIFICATION AND CUSTODY

- `ruff`, `py_compile`, and the focused suite pass: **9 passed**.
- The source payload-retention detector reports no findings.
- A second `--resume-from` run reproduced the candidate archive, live request,
  fire order, and final-result SHA-256 values byte-identically.
- The SSD store is 95 MiB. All 48 rate archives and all locally materialized
  Pose inputs/outputs are retained; no generated candidate payload was
  discarded.
- Modal was not fired; `upstream/` and the protected files were not edited.

## RECALL EVIDENCE

### STORES CONSULTED

- Full `.omx/research/` content search, including arm receipts and memos, for
  `pose-null`, `Schur`, `compensation coding`, `realization efficiency`,
  `flips/B`, `shared codebook`, `dc0`, `expensive-cancellation tail`, `JS6B`,
  and `HV1`.
- `tools/list_canonical_equations.py --json`, including
  `pose_stack_exact_budget_v1`,
  `receiver_pose_semantic_preservation_ratio_v1`, and
  `receiver_lattice_leakage_exponent_v1`.
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED surfaces, current task
  ledgers, `.omx/state/main_hot_state.md`, the actual QS1/js6b dispatcher and
  worker, and the QS1/CP135 retained SSD stores.

Beyond the charter seeds, HV1's corrected candidate-specific stack-budget law
forbids spending one Pose budget per proposal; the receiver-preservation law
forbids inferring Pose from byte/determinism identity; and the receiver-lattice
law says continuous pose-null falloff need not survive uint8. JS6/JS6B close
only their unprojected formulation, not this coupled compensated object. Those
findings changed the plan by making complete-score admission candidate-specific,
keeping the exact QS1 lattice as the fire candidate despite step 2's favorable
local Pose row, and treating all coarser negative rows as instance-scoped.
No canonical equation was found that supersedes the measured 0.785 QS1
breakeven law.

## BOUNDARIES

- `[contest-CUDA T4 component instrument, n600]`: the QS1 exact-step prior
  components only; no new T4 measurement in this arm.
- `[macOS-CPU exact byte/container measurement]`: the 48-row archive race and
  all byte counts.
- `[macOS-CPU advisory frozen CPU-PoseNet, six changed pairs over n600]`: the
  four-row quantization curve only.
- `INSTANCE`: step 3 and step 4 fail the local Pose gate. This does not kill
  coarse compensation as a family.
- The 34-byte candidate is pre-admitted for one exact component run, not
  promotion-eligible and not an exact score.
- Goal status: the exact pointer remains `0.16195513827824176 @ 186,252 B
  [contest-CUDA T4, n600]`; sub-0.15 was not achieved.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN sole scorer-lane router; consumer
  store: `/Volumes/VertigoDataTier/pact/ddm_qs2_20260813`; fire trigger: no
  active n600 exact-eval/Modal lane and every R2 sealed-input SHA verifies, then
  execute the exact argv above and let the dispatcher self-claim.
- **QUEUED-AT-HARVEST** — owner: MAIN; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_qs2_20260813`; fire trigger: the R2 retained
  argmax field and matched base/GT fields are locally available, then compute
  the six-pair flips/changed-pixels/reversion decomposition and waterfill from
  those retained fields without a SegNet rerun.

## LIVE-HYPOTHESES

- The exact-step candidate will retain QS1's -32 net flips and +1.126177e-7 S
  Pose leakage on T4 because it preserves the semantic, token, and signed-int12
  code lattice exactly; only the container/receiver integration is newly
  compiled. This is plausible from strict parse-back but untested through the
  full CUDA renderer.
- Dead-zone step 2 may dominate exact step 1 after a matched T4 measurement: it
  costs only +9 bytes and improved the local Pose term. This is plausible but
  cannot be admitted from a nonmonotone, cross-axis local curve.
- Per-pair waterfilling or quantum-floor engineering may raise the realization
  efficiency above 16.9% because 189 changed pixels collapse to only 32 net
  flips and the six code supports are heterogeneous. The retained field is the
  missing evidence needed to localize that loss.

## DEAD-ENDS

- The original +77-byte QS1 closure is closed as an **INSTANCE**: 12.8333
  B/pair and 0.415584 flips/B miss the measured rate law.
- Dead-zone steps 3 and 4 are closed as **INSTANCES on the local Pose axis**:
  their +2.47e-5 and +2.90e-5 Pose-term changes erase their byte savings.
- Local archive equality, deterministic repeats, or local Pose ordering cannot
  stand in for matched T4 semantic preservation; the exact-step local/T4
  disagreement directly refutes that shortcut.
- Re-deriving the missing argmax field with another SegNet run is closed by the
  retained-field-only charter constraint; recover/harvest the retained bytes.
- `ddm_qs2_dual_axis_20260813_r1` is dead: its request violates the unchanged
  worker's Pose-unknown placeholder contract. Only R2 is live.
