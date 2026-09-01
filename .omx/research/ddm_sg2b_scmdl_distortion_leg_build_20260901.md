# SG2B — SCMDL distortion-leg compose build

Date: 2026-09-01

Task: #1374

Verdict: **BUILD-COMPLETE / QUEUED-WITH-A-MAIN-FIRE-ORDER**

Authority: `[macOS-CPU scorer-free exact field re-encode / no score claim]`

## Result

All four SFP1 fields are now canonical-advisory-ready as pin-consistent AFR1 runtime/archive
trees. The p00 inverse-coder control reproduced AFR1's 113,411-byte RC64 stream byte-for-byte over
all 600 frames. The three changed fields were then physically re-encoded through the same JG2
receiver-mirror, retained as real archives, staged beside copied AFR1 receivers, repinned from the
actual archive bytes, and accepted by `tools/fire_local_advisory.py --dry-run`.

No scorer ran. No Modal job ran. These are distortion vehicles, not score rows and not RXC1 rate
admissions. MAIN owns the one local scorer lane and the gate-2 join.

| proposal | verified field SHA-256 | changed sites / 117,964,800 | exact compose route | advisory archive |
|---|---|---:|---|---|
| p00 null | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` | **0 / 117,964,800** | byte-identical AFR1 runtime/archive plus full-n600 JG2 inverse-coder control | **180,002 B**, `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` |
| p01 | `75fe37daf8c3f615cd943a76697e9c6e8eabc56cb1c23d55a6b4251fc4553690` | **1,084 / 117,964,800** | JG2 lossless re-encode, archive splice, receiver repin | **179,833 B**, `83660e34cae84d620e74ad3fe2cf293ec72732a8866c29194fee14d046dcfb97` |
| p02 | `656bd0c5c102109c3327eccd0c6e3a606aac44cbce7d9144396f8c171e24b76e` | **2,831 / 117,964,800** | JG2 lossless re-encode, archive splice, receiver repin | **179,496 B**, `440244d45d9a884112c6a03e8cb4584df81f6928cbe4f2229d9b99dafe6aa9f6` |
| p03 | `fe6a9dd8ce770e308c7c3d1903ea1e40bee44938cc836188e486eefd408f527a` | **9,723 / 117,964,800** | JG2 lossless re-encode, archive splice, receiver repin | **178,014 B**, `9a77787c257403c65c1a1104ccd1289aee9f7719ba69153c40e0bf3e8ac96ae6` |

The archive sizes above are measured facts about the fixed-G compose vehicles. They are **not** the
SCMDL rate leg: SFP1 marks the proposals `refit_required=true`, and RXC1 owns the exact-delta API and
G/M refit. MAIN may join distortion only to RXC1's result after gate 1.

## Exact shipped render path

`fire_local_advisory.py` consumes a staged archive/runtime, not pre-rendered frames. The exact
AFR1 path is therefore pinned at source:

| role | exact file | bytes | SHA-256 |
|---|---|---:|---|
| RX1M container parser and riders | `/Volumes/APDataStore/pact/ddm_pq12/generation_7_afr1/runtime/residual_archive.py` | 30,622 | `aca361f3e94941f4f2800bacec79f5032335588e317e76ee1a306bbb5ba64530` |
| receiver orchestration and token decode | `/Volumes/APDataStore/pact/ddm_pq12/generation_7_afr1/runtime/f26_inflate.py` | 27,793 | `5d705f93c051b2b540845dad4140f73d7dd61c721e4de2ed33b2ad32170c35c4` |
| class-field renderer | `/Volumes/APDataStore/pact/ddm_pq12/generation_7_afr1/cpr1/inflate.py` | 13,792 | `ff446edd9237148bdc898be2f8f8c4782bf231a50cf3830c4b0b21a4474a736b` |

The archive member begins with the measured RX1M header
`magic=RX1M, version=1, codec=2, table_mode=0, reserved=0b11010,
hpac=13,515, semantic=30,856, carrier=22,010`. The route is:

```text
RX1M read_residual_archive
  -> restore the counted renderer/carrier and HPAC object
  -> decode_production_tokens to the 600x384x512 uint8 class field
  -> cpr1/inflate.py::render_video(tokens=that exact field)
  -> camera-resolution uint8 raw
  -> canonical advisory's frozen upstream R/scorers
```

This is the RX1M-container path resolved by #1333. `unpack_semantic_pose` is not used as a
substitute renderer or hand-composed container parser; the runtime's own RX1M riders execute first,
and `render_video` is the source-verified consumer of the decoded class field.

## Null identity control

The p00 field is byte-identical to AFR1's decoded field. The retained AFR1 full-receiver identity
already proves, on the exact same archive bytes, 600/600 pairs, 117,964,800/117,964,800 decoded
tokens, and 3,662,409,600/3,662,409,600 raw bytes with zero differing raw bytes. SG2B additionally
executed a fresh full-n600 inverse-coder control before any proposal:

| p00 control | result |
|---|---:|
| frames | **600 / 600** |
| dense tokens | **117,964,800 / 117,964,800** |
| emitted RC64 stream | **113,411 B**, `5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3` |
| equality to shipped AFR1 stream | **byte-identical** |
| receipt | `/Volumes/APDataStore/pact/ddm_sg2b_scmdl_distortion_leg_build/P00_CONTROL.json`, `410000237d34e255963f7c5a96c0a238f4a42f9d4bf7661fe46230580fb235d3` |

The scorer-free identity receipt is
`/Volumes/APDataStore/pact/ddm_sg2b_scmdl_distortion_leg_build/NULL_IDENTITY.json`, SHA-256
`208e30b6d6a8c26736d239e8f341f9e82f30103d050576f0c2b5e6c8354daadd`. It binds the exact AFR1
archive and retained full-receiver proof to p00. The known AFR1 components are
`d_seg=0.00020139` and `d_pose=6.37e-06`; a **fresh** advisory measurement was not run by this arm.
MAIN must execute p00 first and require exact numeric equality. Any deviation is
`STOP_INSTRUMENT_DEFECT`, not a changed candidate result.

## Composition custody

Each dense field was SHA-verified before it was read. Changed-site counts were re-derived from the
actual 117,964,800 bytes, not copied from SFP1 metadata. The pair-plane overlays parse back to the
same dense-field SHA:

| proposal | active pairs | retained overlay | overlay SHA-256 | JG2 receipt |
|---|---:|---:|---|---|
| p01 | 24 / 600 | 34,430 B | `9c8bcdc738bc9044e96155cae7c1ae1781b441b87cbda8c23afa0e8456e2a1f8` | `95704f7b2ece33ac0139d038e800124a60e8f1d0f0ff9b2459456ab47627882b` |
| p02 | 64 / 600 | 92,409 B | `9ae52f56797dbc24ae6d8f83aae53a442c3a26b0b1207308f0be93a35455de9d` | `b4e83046cdebdc3fc40dac258b873b2420d9451bbbb664ce8145064366aa6fee` |
| p03 | 600 / 600 | 837,020 B | `de5b85a690afb9c092a5fa860c6f0bf121020ce3b5ff88d6d55af65b8b13321a` | `b87768c91c0d90f373338bd4c9c038dbba8def79fef4b01c69f7fc23e71e6168` |

The receiver-mirror source was pinned at SHA-256
`e762bead28ab981980aa64161e9104bf1ef5e61c450888edf7777a550c3ac70d`. JG2's p00 control proves
the inverse on AFR1 byte-for-byte, and each candidate encode follows the same causal probability
trajectory while checking every emitted token plane against the target. Candidate streams,
per-frame ledgers, 25-frame crash checkpoints, archives, overlays, runtimes, and receipts remain
under `/Volumes/APDataStore/pact/ddm_sg2b_scmdl_distortion_leg_build/`.

The APDataStore build occupies 140 MiB. Full advisory outputs route to
`/Volumes/VertigoDataTier/pact/ddm_sg2b_scmdl_distortion_leg_build/fire_main/` because APDataStore
does not have the 3.66 GB raw-output payload plus reserve. Nothing was deleted or moved.

## Typed MAIN fire order

Disposition: **QUEUED-WITH-A-FIRE-ORDER**.

Owner: **MAIN sole scorer-lane router**.

Axis: `[macOS-CPU advisory]`, n600, non-authority.

Machine-readable order:
`/Volumes/APDataStore/pact/ddm_sg2b_scmdl_distortion_leg_build/MAIN_FIRE_ORDER.json`, SHA-256
`b3c44ac25298f331cf36db1cf99f1ec8aeb32093a7a6cc5cf161f2fcb1a43b12`.

Fire trigger: MAIN has a free scorer slot, no duplicate SG2B/local-advisory lane is active, and the
runtime/archive pin plus canonical dry-run still validate. Order is strict:

1. Fire p00 through `tools/fire_local_advisory.py`. Require exactly
   `d_seg=0.00020139` and `d_pose=6.37e-06`; otherwise stop as an instrument defect.
2. After p00 passes, fire p01 once.
3. Fire p02 once after p01 is terminal and harvested.
4. Fire p03 once after p02 is terminal and harvested.
5. MAIN joins each advisory's `delta d_seg` and `delta d_pose` to RXC1's exact-delta result only
   after gate 1, writing the gate-2 joint table against
   `/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/HANDOFF.json`.

Each row's exact argv, runtime, archive, consumer store, owner, disposition, and trigger are in the
fire-order JSON. Four canonical `fire_local_advisory.py --dry-run` manifests passed and are retained
under `retained/dry_runs/`.

## RECALL EVIDENCE

The recall searched `SCMDL`, `SFP1`, `RX1M`, `render path`, `unpack_semantic_pose`, `RC64`, `JG2`,
`receiver close`, `fire_local_advisory`, `#1333`, `#1370`, and `#1374` across `.omx/research/`, the
research index and sub-0.15 DAG, `.omx/state/main_hot_state.md`, the task/lane state, `experiments/`,
`src/`, `tools/`, and `.omx/state/canonical_equations_registry.jsonl`.

Beyond the charter seeds, recall found three load-bearing constraints:

- AFR1's retained identity proves the exact current archive decodes to the base field and produces
  the retained raw bytes; this made p00 an exact same-object identity receipt rather than a renderer
  approximation.
- BZ2D/#1333 established that the RX1M parser and its riders must execute before rendering; a
  plausible hand-composed semantic/carrier path is not the shipped object.
- The canonical advisory tool accepts only a staged archive/runtime and enforces receiver-pin
  consistency, so the initial frames-direct possibility was closed at source and the plan changed to
  one lossless archive splice per proposal.

The searched scopes did not contain another current SG2B distortion-compose harness or a
frames-direct entry in the canonical local advisory tool. The equation registry contained rate laws
for prior token-field/coder formulations, but no equation that could replace the exact RX1M
realization of these new fields. No fit or sampled surrogate was used: pins, counts, container
framing, RC64 composition, and score arithmetic are exact.

The storage preflight also changed routing: APDataStore had about 2.08 GB free at prepare time, which
is enough for the retained 140 MiB build but not one 3.66 GB advisory raw plus reserve. Build custody
remained on APDataStore as chartered; future scorer attempts route to Vertigo.

## Validation and review

- `python -m py_compile` passed for the harness and tests.
- `python -m pytest -q tests/test_ddm_sg2b_scmdl_distortion_compose.py` -> **9 passed**.
- `python -m ruff check ...` -> **all checks passed**.
- Two genuine full-file review passes were completed and all 22 harness entities were marked
  reviewed after each pass. The second pass found and fixed the sole Ruff finding.
- Independent disk reconciliation rehashed all four archives and matched proposal key, changed-site
  denominator, bytes, SHA, receiver pin, and canonical dry-run receipt.
- AFR1 score recomputed from components as
  `100*0.00020139 + sqrt(10*6.37e-06) + 25*180002/37545489 = 0.14797617125559104`.
- `BUILD_DONE.json` is `PASS`, `scorer_ran=false`, `modal_ran=false`; SHA-256
  `09aaab0ad066e7c0f7c34b84aca7026faf91296ef80a22b4119fc0eed4a04dd0`.
- Full build manifest:
  `/Volumes/APDataStore/pact/ddm_sg2b_scmdl_distortion_leg_build/MANIFEST.json`, SHA-256
  `fe99c26d789abf6d279393e13e78dfc88de3f4af7977ac9c877defcca192b133`.

## What is measured and what is not

Measured here: **600/600-frame p00 inverse-coder identity**; **117,964,800/117,964,800-site field
SHA/count checks per proposal**; physical RC64 streams and archive bytes for all four fixed-G
vehicles; pin consistency for 4/4 staged receivers; canonical advisory dry-run acceptance for 4/4
rows.

Reused measured evidence: AFR1 full receiver identity on the exact p00 archive and raw bytes; AFR1's
contest-CUDA components.

Not measured here: fresh p00 scorer components, p01/p02/p03 `d_seg`, p01/p02/p03 `d_pose`, RXC1
refit/exact-delta bytes, gate-2 joint score, contest-CPU, contest-CUDA, or frontier promotion. No score
claim is licensed.

## NEXT_IF_RESUMED

- **GATE_NULL_IDENTITY** — owner: `MAIN sole scorer-lane router`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sg2b_scmdl_distortion_leg_build/fire_main/p00`; fire trigger: MAIN free scorer slot, no duplicate active SG2B/local-advisory lane, and p00's runtime/archive pin plus dry-run revalidate; action: fire p00 once and require exact AFR1 `d_seg` and `d_pose`, otherwise stop as `INSTRUMENT_DEFECT`.
- **QUEUED_MAIN_ADVISORY** — owner: `MAIN sole scorer-lane router`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_sg2b_scmdl_distortion_leg_build/fire_main/{p01,p02,p03}`; fire trigger: p00 exact-identity PASS and each preceding row terminal/harvested; action: fire p01, p02, and p03 sequentially once each through the retained canonical commands.
- **ADMIT_OR_REFUTE_GATE_2** — owner: `MAIN`; consumer store: `/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/HANDOFF.json` plus RXC1's exact-delta store; fire trigger: each advisory is terminal and RXC1 publishes `GATE-1-PASSED` exact deltas; action: join measured distortion to exact refit bytes, recompute from components, and record the typed gate-2 verdict without promoting advisory evidence.

## LIVE-HYPOTHESES

- p01 may lower realized `d_seg` because it changes only 1,084 boundary-distance-1 sites selected by
  realized terminal argmax disagreement, limiting spill while concentrating the prior signal. This
  remains untested until MAIN scores the real render.
- At least one of p01/p02/p03 may have negative `delta d_seg`; all three use the one realized-cell
  selection signal absent from the closed token-GT families. The falsifier is all three measuring
  `delta d_seg >= 0`.
- The negative fixed-G vehicle byte deltas (-169/-506/-1,988 B) make it plausible that some rate
  credit survives RXC1's refit, but G/M coupling and prior non-additivity forbid transferring those
  values into the gate-2 join.
- p03 may expose the strongest benefit or the strongest spill because its 9,723 edits cover all 600
  pairs. Its width is why it remains last rather than why it is presumed better.

## DEAD-ENDS

- Frames-direct advisory input is closed for this task: the canonical tool accepts a staged
  archive/runtime and checks the receiver pin before launch.
- A hand-composed renderer or direct use of `cpr1.unpack_semantic_pose` as the container path is
  closed by #1333 and the source trace; AFR1's RX1M parser and riders must execute.
- Token-GT or label-space acceptance is closed by WWC1's broken realized-transfer evidence. Only the
  real render and frozen scorers can admit these fields.
- The first detached batch attempt did not survive this managed shell: PID 79550 exited before its
  first checkpoint, leaving zero-byte stdout/stderr and no payload. It was not retried. The completed
  fallback used four individually bounded (<30 minute), checkpointed, restartable steps, which is the
  charter's legal single-proposal form.
- The initial p00 fire-order path `runtimes/p00` was an instrument bug: the real staged path is
  `runtimes/p00_null`. Finalize refused before any fire, the resolver was fixed and regression-tested,
  and all four canonical dry-runs then passed.
- The fixed-G archive deltas are closed as rate admissions. They are vehicle facts only; RXC1's refit
  and exact-delta API own the rate leg.

[contest-CUDA T4 n600] own-vehicle frontier: AFR1 — S=0.14797617125559104, archive=180,002 B, d_seg=0.00020139, d_pose=6.37e-06, SHA-256=cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25 — UNMOVED by this arm.
