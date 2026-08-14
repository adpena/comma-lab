# DDM FS1 fire-seal adapters — 2026-08-14

Both MAIN-blocked gates are now locally fireable. MC36 Variant C has a real
`ddm_qs1_t4_dual_axis_request.v1` seal accepted by the unchanged QS1
dispatcher, and the MT1 dispatcher now builds from the pre-mount Modal base so
its locked-venv build steps precede every runtime local mount. No Modal call,
scorer forward, remote worker, or score evaluation was run.

## MC36 dispatcher-conformant seal

`experiments/ddm_mc36_dual_axis_seal.py` consumes the already retained Variant-C
archive/runtime and local advisory receipt. It refuses drift from the pinned
object:

- archive: 186,269 B, SHA-256
  `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`;
- runtime: 238,713 B, SHA-256
  `64e4642d30b436e6393d5573efcb579a13f922726566790efad40bc2ca117545`;
- local advisory triple: 37 net Seg flips, +17 archive bytes, and
  `delta_dpose=-1.4632967835484165e-10` on the existing
  `[macOS-CPU advisory] NON-PROMOTABLE` surface.

The advisory values live inside `POSE_SCREEN_RESULT.json`. The worker-facing
request follows the corrected QS2/QS5 r2 placeholder law exactly:
`local_pose_delta=0.0`, `pose_unmeasured=true`, `retain_pose_vectors=true`,
`score_claim=false`, and `promotion_eligible=false`. Its input census is
exactly `{candidate_archive.zip, candidate_runtime.zip,
POSE_SCREEN_RESULT.json}`.

The builder called
`experiments.ddm_qs1_modal_t4_dual_axis.load_sealed_inputs` against the real
written request and copied inputs. Acceptance passed. The resulting request is
13,006 B with SHA-256
`2a311813637cfdafd66fcee3d154c7332400cb02b225e16066cdc13e3cbdd969` at:

`/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatcher_conformant_seal/SEALED_REQUEST.json`

The source arm's nonconformant order-metadata request remains preserved and was
not edited.

## MT1 Modal image closure

`experiments/ddm_mt1_modal_multitoken_sign_gate.py` no longer calls
`run_commands` on the shared, already locally mounted `eval_image`. It now:

1. derives from the shared `base_image`;
2. sets the image-level worker `PYTHONPATH`;
3. copies `upstream/` into the build layer with `copy=True`;
4. creates the same frozen upstream CUDA-group venv and installs the same
   pinned `pydantic==2.13.4` and `Brotli==1.2.0` tuple; and
5. adds `src/` plus every MT1 worker/source mount only after all build steps.

The change is limited to image construction. The sealed request, payloads,
run identity, worker behavior, lane claim path, retention, and harvest logic are
unchanged.

The retained MT1 request was reloaded through the edited dispatcher's real
`load_sealed` gate: all nine payload records passed. The sealed dispatcher
source hash is historical (`238c98b1...`) and the edited source hash is
`88dbb93e...`; the loader accepts this intentionally because it authenticates
the request SHA and every payload SHA, not dispatcher source. Therefore the
dispatcher-only image fix does not require or authorize resealing MT1.

## Exact MAIN fire order

Fire serially, MC36 first. MAIN must still perform the live global single-flight
check and lane claim; neither preparation step performed those actions.

### 1. MC36 Variant C

```bash
.venv/bin/modal run --detach experiments/ddm_qs1_modal_t4_dual_axis.py::main --sealed-request /Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatcher_conformant_seal/SEALED_REQUEST.json --fire-input-dir /Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatcher_conformant_seal/fire_inputs --expected-request-sha256 2a311813637cfdafd66fcee3d154c7332400cb02b225e16066cdc13e3cbdd969 --output-dir /Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatch/ddm_mc36_dual_axis_t4_r1 --detach --provider-detach-ack
```

Disposition: `QUEUED-WITH-A-FIRE-ORDER`. Owner: MAIN sole scorer-lane router.
Consumer store:
`/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatch/ddm_mc36_dual_axis_t4_r1`.
Fire trigger: no active full-n600 Modal scorer lane, MAIN has claimed
`ddm_mc36_dual_axis_t4_n600_20260814`, and the request plus all three input SHAs
still verify.

### 2. MT1 sign gate

```bash
.venv/bin/modal run --detach experiments/ddm_mt1_modal_multitoken_sign_gate.py::main --sealed-request /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/SEALED_REQUEST.json --fire-input-dir /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/fire_inputs --expected-request-sha256 c9d6d62c8115f6c209576a57d4cbf7e40c2191c542473fa0df33bc82af91dffc --output-dir /Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/dispatch --detach --provider-detach-ack
```

Disposition: `QUEUED-WITH-A-FIRE-ORDER`. Owner: MAIN sole Modal scorer-lane
router. Consumer store:
`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/`.
Fire trigger: MC36 is terminally harvested and its claim closed, the #978 T4
lane is free, MT1's local build claim is terminal, and the sealed request plus
all nine payload SHAs verify.

## Validation and boundaries

- 31 focused and precedent tests passed across the two new regression suites,
  RE1T preparation, QS1 preparation, and MT1 screen.
- The real MC36 archive and runtime passed `unzip -tqq`; their copied fire-input
  bytes and hashes equal the pinned source bytes.
- The MC36 real dispatcher loader accepted the written request. The MT1 real
  loader accepted the historical request under the edited dispatcher.
- Importing MT1 locally constructed the revised `modal.Image` without auth or
  a provider call. The focused AST regression proves no build step follows a
  non-`copy=True` local mount and that `add_local_python_source` is the final
  image operation.
- Ruff, Python compilation, and `git diff --check` passed on all changed Python
  files before review.
- All newly copied fire payloads remain on Vertigo with bytes and SHA-256; no
  payload was deleted, moved, measured-and-discarded, or reduced to scalars.
- No exact score was measured and no frontier pointer moved. These adapters are
  means for two imminent remote measurements, not goal progress by themselves.

## RECALL EVIDENCE

Searched before implementation:

- `.omx/research/`, arm final messages, `CANONICAL_RESEARCH_INDEX*`,
  `sub015_DAG_*`, live hot state, canonical task status, and task ledgers with
  `mc36|mt1|multitoken|dual-axis|fire order|sealed request|worker dependency|Modal image|add_local|build step`;
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for
  `modal|worker|seal|dispatch|dependency|payload|fire`;
- RE1 seal commit `9207d5eac0`, MC36 commit `2e4abc6210`, MT1 commit
  `af56d51c48`, the current QS1 dispatcher/RE1T worker, JS1B record helpers,
  MC36 retained receipts, and both QS2/QS5 r1/r2 seals and pose evidence.

Beyond the charter seeds, the corrected QS2 and QS5 r2 requests proved that
their first seals carried advisory Pose values in the top-level worker field
incorrectly; r2 moved that field to literal zero while preserving advisory
evidence in the screen payload. That changed MC36 to the r2 law. The live MT1
source inspection showed the failing `run_commands` was not after MT1's own
mounts; it followed mounts inherited from shared `eval_image`. That changed the
fix from merely reordering MT1's visible suffix to deriving from the pre-mount
base image.

The existing repository-wide Modal image audit was also inspected. It does not
track image ancestry across assigned variables, treats `add_local_python_source`
as a build step, and reports `copy=True` build inputs as violations. Making it a
sound `experiments/*_modal_*.py` recurring-census seed with a same-line waiver
requires a separate provenance-aware correction, so FS1 used the charter's
typed-worklist fallback instead of landing a false-positive gate.

Own-vehicle frontier remains `S=0.16959899569230852 @ 187,226 B [contest-CUDA
T4, n600]`; FS1 did not move it.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/dispatch/ddm_mc36_dual_axis_t4_r1`; fire trigger: no active full-n600 Modal scorer lane, MAIN claims `ddm_mc36_dual_axis_t4_n600_20260814`, and the MC36 request plus three input SHAs verify; action: run the MC36 command above and harvest every retained return.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN sole Modal scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/multitoken_978/ddm_mt1_20260814/optimal_form_r2/retained/t4_sign_gate_r1/`; fire trigger: MC36 is terminally harvested and closed, the #978 T4 lane is free, the local build claim is terminal, and all MT1 request/payload SHAs verify; action: run the MT1 command above and harvest `FINAL_RESULT.json` plus complete remote custody.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: DT1 recurring-census adapter maintainer; consumer store: canonical task-status/costate duty queue; fire trigger: the next recurring-genus adapter landing; action: normalize the existing Modal image-order audit into an ancestry-aware `experiments/*_modal_*.py` seed that honors `copy=True`, recognizes every local-mount method, supports a substantive same-line waiver, and starts warn-only.

## LIVE-HYPOTHESES

- MC36 Variant C may preserve its negative local component delta on T4 because
  the exact receiver bytes close, all local gates pass, and its affected base
  Seg fields matched the retained T4 base pixel-for-pixel. Candidate scorer
  transfer remains untested, which is why the fresh dual-axis row is necessary.
- MT1 may now reach the worker because the only reproduced provider failure was
  the Modal image phase-order refusal, and the revised chain places target-venv
  provisioning before every runtime mount while preserving its pinned
  dependency closure. GPU execution and scientific sign remain untested.
- A provenance-aware reuse of the existing Modal image audit can prevent this
  class without another repository scanner because its AST chain extraction is
  already close; the missing pieces are variable ancestry, `copy=True`
  semantics, complete local-mount classification, and waivers.

## DEAD-ENDS

- Reusing MC36's original `SEALED_REQUEST.json` is closed: it is fire-order
  metadata and fails the dispatcher's required schema and exact input census.
- Putting MC36's advisory Pose delta in top-level `local_pose_delta` is closed:
  the RE1T worker requires literal `0.0` with `pose_unmeasured=true`; corrected
  QS2/QS5 r2 seals confirm the same law.
- Reordering only MT1's visible local-mount suffix is closed: its late build
  step was already after mounts inherited from `eval_image`, so that edit would
  reproduce the provider failure.
- Resealing MT1 for a dispatcher-only edit is closed: the real load gate
  authenticated the unchanged request and all nine payload SHAs and does not
  bind dispatcher source.
- Promoting either local result is closed. MC36 is advisory and MT1 is a local
  n32 sign screen; neither is an exact n600 contest row.
