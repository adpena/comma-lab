# ddm_cp5v five n600-validated event composition

Tags: [no-triality] [p0-ledger-ok]  
Axis: [macOS-CPU scorer-free direct-token HP3/RC64 n600 reclose]  
Score claim: false

## RESULT

**READY_TO_FIRE.** The exact five-event CP5V object is byte-closed,
receiver-closed through two independent readers, deterministic across a full
second HP3/RC64 compose, and below the charter's `+3 B` falsifier. No scorer,
MPS, Metal, Modal, or other provider job ran in this arm.

| measured object or check | result |
|---|---:|
| primary `archive.zip` | **186,252 B**, SHA-256 `1c66e434ba60c6ff0be5f8634742eff3a85332bab89a12540050df852ea7a986` |
| delta from CP135 | **0 B** |
| independent archive repeat | same bytes and SHA, byte-identical |
| independent fresh RC64 repeat | byte-identical |
| adapted-runtime decoded token plane | 117,964,800 B, SHA-256 `9ab877de7e63d064624040c994368f83eb70da15a5ccd3e42f6a4364828340a5` |
| adapted canonical reader | `runtime.f26_inflate.decode_production_tokens`, `cpr1/`, Brotli 1.2.0, 465.939 s |
| adapted reader vs JO1 shipped RC64 backend | byte-identical full plane |
| base-to-candidate token diff | **exactly 5 cells**, the five requested events, no extras |
| validated singleton Seg gain | 6 flips across five events |
| sum of singleton n600 Pose deltas | `+6.539362330040836e-09` diagnostic, not a composed result |
| exact contest eval | **NOT RUN**; owned by MAIN |

The primary archive is retained at
`/Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/retained/candidates/validated_five/primary/objects/archive.zip`.
The independent repeat is under the adjacent `determinism_repeat/objects/`
tree. The adapted runtime uploaded with the row must be the copied, archive-
pinned runtime under the primary candidate root, not the read-only CP135 source
runtime.

## EXACT FIVE-CELL RECEIVER PROOF

The adapted-runtime canonical reader restored the following complete diff
against base token plane SHA-256
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`:

| proposal | frame | y | x | flat index | source to target | n600 singleton flips gained |
|---|---:|---:|---:|---:|---:|---:|
| `ec1_0003_fcb5ca3a4453` | 7 | 213 | 32 | 109088 | 2 to 0 | 1 |
| `ec1_0004_3bc2b69c706c` | 7 | 213 | 438 | 109494 | 3 to 0 | 1 |
| `ec1_0104_f4e219067530` | 73 | 282 | 318 | 144702 | 4 to 0 | 1 |
| `ec1_0164_3a4e239de5b9` | 96 | 293 | 20 | 150036 | 4 to 0 | 2 |
| `ec1_0168_818a3c77af51` | 96 | 297 | 3 | 152067 | 4 to 0 | 1 |

`30_TOKEN_DIFF_RESULT.json` records the expected and actual sets separately and
asserts equality. Five individual receipts under
`retained/per_event_diff_receipts/` pin the event payload, cell, source/target,
singleton pair, flip gain, Pose delta, and both value checks.

JO1's independent shipped-backend decode produced the same candidate plane
SHA-256. The adapted path compiled and retained the runtime's own
`runtime/entropy/rc64_backend.c` as `rc64_backend.so`, loaded the copied
runtime's `runtime/f26_inflate.py`, bootstrapped the exact `Brotli==1.2.0` pin
from `inflate.sh`, and decoded through its `cpr1/` reader. This is receiver
behavior on the candidate archive, not a source-file or constructed-plane
proxy.

## ARITHMETIC AND AUTHORITY BOUNDARY

The chartered arithmetic uses the measured zero-byte delta:

```text
S_prefilter
  = 0.16195513827824176
    - 5.086263020833333e-06
    + 25 * 0 / 37,545,489
  = 0.16195005201522092
```

This is a **projection, not a score**. It transfers the sum of the five n600
singleton Seg gains and charges the measured archive bytes, but it does not
measure joint Seg interaction or joint Pose interaction. The exact row's
pre-registered additivity readout is the realized composed delta versus the
sum of singleton deltas on affected pairs `[7, 18, 53, 73, 76, 96]`. The
prior JO1 six-event exact row harmed the base by `+0.000216`, so singleton
additivity must not be assumed even though these five were selected from n600
validation.

The effective floor therefore remains CP135 at **S = 0.16195513827824176 @
186,252 B [contest-CUDA T4, n600], ours**. The own-vehicle frontier remains LC2
at **S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]**. CP5V moved
neither frontier because its exact row is unmeasured.

## INPUT PINS AND RETENTION

- VD1 census final: 2,779 B, SHA-256
  `6c53628184f55722f87fcb7e3dadc8b6c9a70025a804e00cfcbecb6674004973`.
- VD1 event rows: 646,298 B, SHA-256
  `a97400d32878318d8eb657a36e62f523e4db48e402b292c09e611d2104b500b3`.
- CP135 adapted-runtime base archive: 186,252 B, SHA-256
  `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
- CP135 retained probability-object base archive has the same bytes and SHA.
- CP135 base token plane: 117,964,800 B, SHA-256
  `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.
- JO1 machinery is byte-identical to landed commit `3bc2cb557f`; runner
  SHA-256 `77990654e90f4c4cd0d2b068e0f039da171889d0ecc6ec418ac5900868d688a9`.
- Proposal store is the chartered 200-event JS5 store and JO1's landed default.

All materialized candidate payloads remain under
`/Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/`. The retained tree currently
uses 2.7 GiB across 2,636 files. It includes both spatial planes, both event-
order planes, every HP3 probability checkpoint, both fresh RC64 payloads and
decode outputs, both archives, the archive-pinned runtime, the canonical-reader
decoded plane, compiled receiver library, dependency receipt, safe-run
receipts, and per-event receipts. Nothing was deleted or moved.

The first canonical-reader call correctly refused because the required
`CPR1_RC64_LIBRARY` binding had not been installed. The driver now reuses VD1's
landed compile seam, retains the source and library, and scopes the environment
binding to the decode call. The successful 465.939-second decode completed and
persisted its payload before the sandbox wrapper disappeared; the next resume
verified and reused that retained terminal receipt. The driver also supports
atomically retained sparse-logit checkpoints for a future interrupted decode,
but the successful candidate did not need that replay path.

## CANONICAL FIRE ORDER

Claim the lane only after every live Modal/scorer claim is terminal:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id lane_ddm_cp5v_validated_five_contest_cuda_20260812 --platform modal --instance-job-id ddm_cp5v_validated_five_t4_20260812 --agent MAIN --status active_exact_eval_spawning --notes 'CP5V five n600-validated EC1 events; sole contest-CUDA row; validate 1c66e434ba60c6ff0be5f8634742eff3a85332bab89a12540050df852ea7a986 at 186252 B'
```

Then use the canonical submission chain, not a probe script:

```bash
.venv/bin/modal run --detach experiments/modal_auth_eval.py::main --archive /Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/retained/candidates/validated_five/primary/objects/archive.zip --output-dir /Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/main_t4 --expected-archive-sha256 1c66e434ba60c6ff0be5f8634742eff3a85332bab89a12540050df852ea7a986 --submission-dir /Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/retained/candidates/validated_five/primary/adapted_runtime --inflate-sh inflate.sh --gpu T4 --scorer-device cuda --expected-runtime-tree-sha256 auto --lane-id lane_ddm_cp5v_validated_five_contest_cuda_20260812 --instance-job-id ddm_cp5v_validated_five_t4_20260812 --claim-agent MAIN --claim-policy require_active --single-axis-waiver-reason 'CP135 F26 family is CUDA-locked; contest-CPU refused by vehicle precedent' --detach --provider-detach-ack
```

Harvest with:

```bash
.venv/bin/python tools/recover_modal_auth_eval.py --output-dir /Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/main_t4
```

The machine-readable twin is
`.omx/research/ddm_cp5v_compose_five_validated_events_20260812_t4_recipe.json`.
This arm did not claim the lane or execute any command above.

## VERIFICATION

```text
.venv/bin/python -m pytest -q \
  experiments/tests/test_ddm_cp5v_compose_five_validated_events.py \
  experiments/tests/test_ddm_jo1_joint_probability_object.py
11 passed

.venv/bin/ruff check \
  experiments/ddm_cp5v_compose_five_validated_events.py \
  experiments/tests/test_ddm_cp5v_compose_five_validated_events.py
PASS

.venv/bin/python -m py_compile \
  experiments/ddm_cp5v_compose_five_validated_events.py \
  experiments/tests/test_ddm_cp5v_compose_five_validated_events.py
PASS

git diff --check -- <CP5V owned files>
PASS
```

The focused payload-retention gate reports no measure-and-discard finding.
Final review-tracker and serializer receipts are recorded with the landing.

## RECALL EVIDENCE

The full local corpus query covered research, equations, memory, DAG, council,
task, and docs stores with these exact content queries:

- `five validated event compose additivity through exact compose`
- `HP3 RC64 direct event probability object receiver close`
- `VD1 event token cell exact diff adapted runtime cpr1`
- `canonical submission chain modal_auth_eval cp135 lc2 js7`

Direct bounded searches also covered `CANONICAL_RESEARCH_INDEX*`,
`sub015_DAG_*`, `.omx/state/canonical_task_status.jsonl`, the current hot
state, and the lane-claim ledger. No prior CP5V implementation or result was
found in those searched surfaces. The relevant recovered sources were the JO1
charter/memo/runner, VD1b charter/memo/worker, VD1 census verdict, T1R1
whole-container rehearsal, CP135/LC2/JS7 exact-row precedents, and JO1's
machine-readable T4 recipe.

Three canonical equation IDs were inspected directly:
`score_marginal_lagrange_multipliers_v1`,
`worldsheet_transport_residual_event_rate_v1`, and
`categorical_blahut_arimoto_rate_distortion_v1`. The first fixed the exact
rate denominator and score-unit conversion. The latter two reinforced event
atomicity and probability-object interpretation but did not replace the real
HP3 export, RC64 encode, archive stat, or receiver decode.

Recall changed the execution in four concrete ways: T1R1 ruled out gluing a
semantic stream without same-object HP3 re-encoding; VD1b supplied the exact
Brotli/`cpr1` loader and RC64 compile seam; JO1 supplied the direct-token
materializer, complete reclose, and independent shipped receiver; and the
CP135/LC2/JS7/JO1 recipes fixed the claim-first, active-lane-required canonical
`modal_auth_eval.py::main` command. The harvested JO1 six-event row also made
the non-additivity boundary load-bearing rather than boilerplate.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN exact contest-CUDA scorer owner. Consumer store: `/Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/main_t4`. Fire trigger: every live Modal/scorer claim is terminal, MAIN owns the sole scorer lane, and archive SHA-256 `1c66e434ba60c6ff0be5f8634742eff3a85332bab89a12540050df852ea7a986`, size 186,252 B, and the archive-pinned runtime still match; then claim and run the commands above exactly.**

## LIVE-HYPOTHESES

- The five n600-positive singleton changes may retain most or all six Seg flips
  jointly because they are five disjoint token cells and the exact receiver
  preserves only those edits. This is plausible but untested; JO1's six-event
  exact row proves interactions can reverse singleton expectations.
- Joint Pose damage may remain negligible because the five singleton global
  deltas sum to only `6.539362330040836e-09`. This is a diagnostic plausibility
  argument, not a composed Pose measurement.
- The exact row may improve CP135 without a rate penalty because the full
  candidate container is measured at exactly the same 186,252 B. Only the
  distortion terms remain uncertain.

## DEAD-ENDS

- Widening the `+3 B` budget is closed. The real compose costs 0 B.
- Shipping a separate EC1 sidecar is closed for this candidate. The five cells
  are carried directly inside the recomputed HP3/RC64 probability object.
- Treating the singleton sum or `0.16195005201522092` as an exact score is
  closed. JO1's exact six-event row was `+0.000216` worse than CP135 despite its
  optimistic singleton prefilter.
- Reusing the CP135 token stream without re-encoding is closed. The five-cell
  plane has a different event-order identity and was freshly HP3-exported and
  RC64-encoded.
- Calling the constructed spatial plane receiver evidence is closed. Both the
  shipped RC64 backend and the adapted runtime's canonical reader independently
  decode the exact archive to the same retained plane.
