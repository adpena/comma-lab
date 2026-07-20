# R1b4 receiver actuator — L1/L2 implementation and measured smoke

`lane_id=lane_r1b4_receiver_actuator_20260720` · `[macOS-CPU capped hard-oracle advisory]` · `research_only=true` · pointer `0.1910828242 [contest-CPU] UNMOVED`

## Verdict

`MEASURED_RECEIVER_SMOKE_XI0_ACTUATOR_UNDERDETERMINED_L3_BLOCKED`

Verdict scope: exact prefix-n2 receiver-smoke archive on macOS CPU only. This is not an n600 row, not an R1b2 boundary candidate, not promotion evidence, and not a pose-actuator or boundary-family negative. No provider dispatch occurred.

The implementation authority was read directly from `r1b4_receiver_actuator_20260720T190615Z.wrapped.prompt.txt`, SHA-256 `b4d1f8156eb60441d0d28db2c22289e3489e20c07c16a9fcc0c250f1b7b59132`, followed by the committed build spec. The required lineage merge was already present; pinned C2 decoder and parser originals were not edited.

## L1 landed: versioned consuming receiver

New `src/tac/boundary_math/r1b4_section_receiver.py` implements:

- exact terminal-ZIP consumption; safe, canonical, ordered inherited C2 members followed by the four R1b2 members;
- byte-for-byte reconstruction and hash verification of the inherited C2 archive before delegating to the pinned decoder semantics;
- exact manifest, base-member, section-length, and SHA-256 custody, with canonical JSON and sealed final-output byte/hash assertions;
- strict `R1K1` compact replay grammar with canonical ordering, exact length/body hash/CRC/final-byte checks, zero receiver search, and direct camera-byte assignments;
- boundary-packet application to frame 1 followed by exact factor-2 uint8 realization and verification;
- xi0 application to frame 0 only through a deterministic integer horizontal translation with edge replication;
- atomic output and receipt promotion, overwrite refusal, output rollback if receipt promotion fails, and scratch cleanup only after the receiver receipt is fsynced;
- an unsealed discovery decode followed by archive output-hash sealing, then production decode refusal for any unsealed archive.

`src/tac/optimization/r1b2_mdl_xi0_compile.py` now refuses noncanonical replay/xi0 bytes and emits the versioned receiver policy/application order and an explicit unsealed n600 final-output assertion. This makes compiled-but-unsealed state literal instead of allowing a counted-but-inert production claim.

Focused tests prove strict parse/refusal for trailing, truncated, duplicate, reordered, unknown, and hash-drifted inputs; replay corruption and noncanonical ordering; deterministic double decode; manifest-policy, boundary, xi0, and replay byte causality; frame-local effects; zero-selection replay honesty; and zero receiver search.

## L2 landed and measured: real xi0 receiver smoke

Final source-bound receipt: `.omx/research/r1b4_receiver_actuator_20260720T195512Z.json`, SHA-256 `ae1c4314f4232f34512a363cf98c23cf64771738fd1f871098e83747e042a0b7`. It supersedes the earlier bounded smoke receipts at `19:37:13Z` and `19:40:49Z`; those remain historical provenance rather than final-source authority.

Sealed smoke archive:

- path: `/Volumes/VertigoDataTier/pact/evidence/r1b4_receiver_smoke_20260720T195512Z/r1b4_receiver_smoke_sealed.zip`
- bytes: `97,660`
- SHA-256: `94ac091aa1946958377ecf82f34468e2f5aec2ac784cb7fdb30d06a27d8c2b5f`
- delta versus the `94,344 B` C2 control: `+3,316 B`
- role: `receiver_smoke_only`; typed zero boundary and zero-selection replay; real banked `1,500 B` xi0 payload
- receiver search invocations: `0`
- sealed decode 1: `12.175791542045772 s`
- sealed decode 2: `12.189127916935831 s`
- two decoded SHA-256 values: equal; raw inputs were certified then deleted after the durable receipt fsync

Section effects in the measured prefix:

- manifest policy and final-output assertion: applied;
- boundary packet: exact factor-2 consumer executed, `0` changed bytes because this smoke intentionally uses the typed zero boundary;
- replay: strict consumer executed, zero selections honestly reported;
- xi0: all `600` values decoded; prefix shifts `[3,4]`; frame 0 changed for both pairs (`5,146,308` changed bytes total); frame 1 remained byte-identical.

Hard CPU-Torch oracle, pairwise prefix n2, seed 1234, one CPU thread:

| row | d_seg | d_pose | 100*d_seg | sqrt(10*d_pose) |
|---|---:|---:|---:|---:|
| exact C2 receiver control | 0.00333404541015625 | 158.02999877929688 | 0.333404541015625 | 39.752987155595854 |
| exact sealed R1b4 smoke | 0.00333404541015625 | 134.981689453125 | 0.333404541015625 | 36.73985430743092 |
| delta | 0.0 | -23.048309326171875 | 0.0 | -3.013132848164936 |

Both individual prefix pairs improved d_pose. This is a measured actuator signal, not an efficacy or n600 claim. The coordinate-to-warp calibration remains underdetermined: dim 0 and frame 0 are grounded, but an exact PoseNet-coordinate-to-camera-warp inverse is not custodied. Therefore the correct endpoint is `R1B4_XI0_TARGET_TO_FRAME0_POSE_ACTUATOR_UNDERDETERMINED`, not a fabricated contribution collapse.

The smoke carrier is `3,316 B`, above the currently registered `1,852.09 B` realization break-even anchor. No candidate was admitted and no new canonical equation was registered.

## Production carrier-contract correction

`R1B4_RECEIVER_CONTRACT_CARRIER_OVERHEAD_EXCEEDS_1852`

The typed receiver compile fixture, including production `384x512` scorer geometry, deterministically produces a `2,114 B` carrier delta under the production compiler, exceeding the exact production cap of `1,852 B` by `262 B`. The compiler therefore refuses fail-closed and leaves no candidate or staging artifact. No test-only cap override is permitted. This compile refusal does not invalidate the separate sealed receiver build/decode proof above; those receiver tests continue to prove deterministic construction and byte-identical double decode.

Carrier-gate receipt: `.omx/research/r1b4_receiver_carrier_gate_20260720T194049Z.json`, SHA-256 `f61419c1118fd41cb3b0495afe79b164d05d17cf3b2be7be251324cd1421ddf2`.

The inbox-provided fixed-C1 pointer-crossing precision warning is also consumed fail-closed: this compiler uses `216,222 B` as the safe cap, not the unreconciled `216,223 B` arm constant.

## L3/L4 fail-closed blocker

At the measurement milestone, the live read-only campaign re-audited as:

- status: `IN_PROGRESS_OR_SCOPED_BLOCKED`;
- completed: `565/600`;
- refused: `[11,245,277,482,514,532,574]`;
- total missing: `35` = seven refused plus 28 collateral/unresolved;
- blockers: `VJP_CAMPAIGN_NOT_TERMINAL_COMPLETE_N600`, `VJP_COMPLETED_PAIR_COUNT_565_NOT_600`, `VJP_MISSING_PAIR_IDS_PRESENT`, `VJP_REFUSED_PAIR_IDS_PRESENT`.

A later MAIN directive and the `launch_004` custody sharpened the mechanism: each refusal aborts its deterministic chunk, so retries cannot fill the 28 non-refused collateral pairs. The driver source lives at git `1b5507f758` in the separate probe worktree and is absent from this merged receiver lineage, so this branch refused a cross-worktree edit. Durable blocker: `.omx/research/r1b4_l3_chunk_abort_blocker_20260720T200024Z.json`, SHA-256 `887b9c05f8beb0041716c5e5de44c5cbe7496ba319054658abc7739e23ac69f0`.

No L3 assembly, refusal waiver, fresh-winner re-derivation, L4 compile, provider dispatch, or n600 score claim was attempted. P2 zero selection was not treated as an efficacy blocker. The exact cure is now two-part: the owning probe lineage must isolate refusal to one pair and continue each chunk, then fresh batch-16 native winners must be re-derived for the seven refused pairs.

## Verification

- `18 passed` across receiver, compiler, and measurement-tool focused pytest.
- Ruff clean on every touched Python surface.
- `py_compile` clean for receiver, compiler, and measurement CLI.
- JSON parse and exact SHA checks clean.
- Certified success-only raw cleanup verified; durable archives and decode receipts remain on the SSD.
- Final measurement source custody matches the current receiver, compiler, xi0 codec, and measurement-tool bytes exactly.

## Stores consulted

- exact delegated wrapped prompt and committed R1b4 build spec;
- settled C2 control receipt/archive and pinned decoder;
- R1b2 compiler and R1b3 xi0 codec/producer receipts;
- real banked xi0 payload;
- frozen scorer/cache helpers used by the settled C2 hard-oracle path;
- live read-only VJP campaign receipt;
- main inbox update at `2026-07-20T19:22:34Z`.
- main inbox mechanism update at `2026-07-20T19:55:39Z` plus read-only `launch_004` custody.

MAIN review should independently inspect the strict member grammar, output-sealing transaction, section-causality tests, frame-0-only xi0 isolation, compiler unsealed-state contract, exact receipt authority, and rerun the focused test command before serializer commit.
