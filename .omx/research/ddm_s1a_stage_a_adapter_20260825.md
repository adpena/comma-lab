# ddm_s1a Stage-A adapter verdict — apparatus complete, MAIN gates pending

**Date:** 2026-08-25  
**Axis:** `[macOS-CPU scorer-free exact byte/container apparatus]`  
**Verdict:** `APPARATUS_COMPLETE_MAIN_GATES_PENDING`  
**Seal:** `/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter/S1A_CHAIN_SEAL.json`, SHA-256 `bf8424fd931adb7ffcacef91df3eab1941f3b8d1c304aa4e02b22baa75f604cc`  
**Score claim:** false. **Frontier moved:** false. **Training / scorer / Metal / Modal invocations:** `0 / 0 / 0 / 0`.

Stage A is now a real WD3 path over the exact GB1 body. It admits both registered seeds, strictly
loads the errata-resolved RJ1 Film-W96 initializer, preserves every non-renderer GB1 section, runs
the RJ2 carrier encoder and compensation algebra against this body, and retains two byte-close-loadable
scorer-free births plus three typed launch requests. MAIN still owns every live-compute gate.

## RECALL EVIDENCE

**Sources searched.** I read the charter and common contract; `PROGRAM.md`; the governing repo files;
`docs/operating_manual_craft_handoff.md`; `main_hot_state.md`; the S1 compiler, memo, and interface
audit; RJ1/RJ2 source, memos, receipts, and custody trees; WD3 source, receiver, receipts, and tests;
DS1 source, tests, R0/R1 memo, and retained R0 receipts; JG2's moved-field encoder; WA1/FB1/W72/NY1/QS5;
the canonical research indexes, `sub015_DAG_*` FEED surface, task/queue ledgers, and the canonical
equation registry from `tools/list_canonical_equations.py --json`.

**Queries.** Bounded content searches included `#1270`, `S1`, `Stage A`, `Film-W96`, `GB1`,
`renderer_only_mutable`, `RJ2`, `compensation`, `carrier re-solve`, `cheap-to-shrink`, `uniform`,
`waterfill`, `20260815`, `20260816`, `seed`, `resume`, `93.23`, `moved token`, `JG2`, and `Pose6`.

**Beyond-charter findings and plan changes.** The DS1 objective already existed but was not imported or
called by WD3. DS1 R0 further forbids using sensitivity-waterfill alone as this treatment's ladder:
the retained component gate shipped uniform int4 and the proxy did not establish a reliable ordering.
I therefore wired the actual uniform `4 -> 3 -> 2` packet ladder, with real serialized byte checks,
instead of rebuilding DS1 or transferring the obsolete ceiling proposal. The same recall proved WD3
had never varied its seed, so separate seed-specific births were required; changing a seed after resume
would be overwritten by restored RNG. No canonical-equation entry supplied a current GB1/W96 compile
contract. The live board still routes #1270 S1 as the remaining sub-0.12 path.

## Executed binding re-proofs

| binding | executed result | durable receipt |
|---|---|---|
| RJ1 custody | `PASS`; current verified payload records numerator `189`, current payload records denominator `189`; three AppleDouble metadata rows voided; verified payload bytes `5,363,215` | `RJ1_CUSTODY_REPROOF.json`, SHA `5df83b6a122eee218fad96a2028a82ca441b6646d65f447726d9cd0ea29b1b23` |
| Film-W96 initializer | strict load into WD3 W96; `253,955 B`, SHA `e74ba046af251808ef105cf0a2295f6133efa194360148f3110762765b9db434` | same RJ1 receipt plus both retained birth checkpoints |
| exact GB1 body | `180,215 B`, SHA `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`; runtime archive equal; deterministic ZIP rebuild equal | `GB1_IDENTITY_REPROOF.json`, SHA `6a93850e0f1e20f74f4430e644f44b96d7d118b1ac01e7a6050e9a5820abb484` |
| WD3 receiver/container | fresh-process WD3 packet parse and repack exact; candidate primary/repeat equal; source archive absent from candidate runtime | `retained/initializer_candidate/PARSEBACK_TRANSCRIPT.txt` and `SECTION_PRESERVATION.json` |
| RJ2 carrier | CAP1 -> DX2 -> RR5 -> Brotli q9/lgwin16 reproduced the exact GB1 carrier stream; stream and state-code verdicts exact | `RJ2_ADAPTER_REPROOF.json`, SHA `8fc5ce3da78c11f5ee5e023ec79d30ec01b636965bcf42883b9266835dd7e15c` |
| RJ2 compensation | retained solve replay maximum absolute float delta numerator `5.577040332216399e-6`, tolerance denominator `1e-5`; proposed int12 code lattice exact | same RJ2 adapter receipt; retained update and proposed-code payloads |
| seed variation | initializer tensors identical numerator `32`, initializer tensors denominator `32`; optimizer state identical; generator RNG differs; no post-knob overwrite | `SEED_VARIATION_REPROOF.json`, SHA `6e0aa594b763f860c9f813de68e77735d571d740b5d37f78b00dcfb69f6ea087` |

The compensation float replay is deliberately not called bit-exact: RJ2 retained `trained_pose` as
float32 although its local solve consumed the live tensor. The tolerance passes and the shipped int12
decision is exact.

## GB1 section preservation

Only the semantic/renderer section changed. These source and candidate records are byte-identical:

| untouched section | bytes | SHA-256 |
|---|---:|---|
| HPAC | 13,515 | `602115b323b0e403d08287af9b273a2d4fb23e026d83c1f6e4609ed77ef98f98` |
| carrier | 22,010 | `932b979f5181b331a9099162c6f392f558860b7998c62a36f38c2c99629c9b12` |
| fixed residual | 96 | `8ab2fe748ab7d69d2102ba2292289e22bd7ea503f8ae29938e0854ec46ca3da1` |
| token stream | 113,624 | `a2e3bfc056fba17c387751866105c2c0568437a8ca211e9a6a3714cad0ed782a` |

RX1 magic/version/codec/table-mode/reserved, section framing other than the required semantic-length
field, tail length, and ZIP metadata also match. The initializer specimen changed the semantic stream
from `30,856 B` to `30,807 B`; its retained archive is `180,166 B`, SHA
`85ed7be3eb421a8527d8a42b0c43981c8f5601ab280f17d1a17b6ccfd36dcb83`. This is a receiver/container
specimen, not an admitted score candidate and not a frontier row.

## Per-seed readiness and sealed MAIN launch order

| run | treatment | birth/request custody | readiness |
|---|---|---|---|
| seed `20260815` | OFF | birth `533,061 B`, SHA `97f2f6d5...544c8b`; request config SHA `11b83768...fea83` | apparatus complete; MAIN gates pending |
| seed `20260816` | OFF | birth `533,061 B`, SHA `3a624389...bd272`; request config SHA `bd5085f3...66017` | apparatus complete; MAIN gates pending |
| seed `20260815` | ON, sampled real uniform int3/int2 rungs | shares the exact OFF-15 birth; request config SHA `163cb5d5...d577` | queued only after the two-seed OFF floor is reviewed |

Order: `off_seed_20260815 -> off_seed_20260816 -> on_seed_20260815`, sequential only. Every run retains
all evaluated packet/archive payloads, resumes from disk, saves at most every five epochs, and preserves
its stage-end checkpoint. The only blockers are MAIN-owned: fresh distinct scorer and Metal lane claims,
`launch_authorized=true`, and verified exit of r5 PID 63183. Apparatus blockers are empty.

`MAIN_LAUNCH_ORDER.json` is SHA
`708eae6a93684a077fe93aae202f0743b414f9ad36b52889bc76213bfe0833f4`.

Memory preflight measured available-memory bytes numerator `100,645,699,584` against planned launcher
limit bytes denominator `68,719,476,736`, ratio `1.4645876884`. The prior F64 peak RSS MiB numerator is
`7,511.219` against planned launcher limit MiB denominator `65,536`, ratio `0.1146121063`. From the
measured `93.23 s/epoch`, 65 epochs/run, two OFF runs, and an ON scorer-pass factor projection of two,
the projected total is `6.7332777778 h`. Receipt: `MEMORY_WALL_PREFLIGHT.json`, SHA
`bc8084b9f402cc6f7804a0fdb3e21a78689e66214453ccffd991fbe793ac096b`.

## Stage-B fingerprint contract

`STAGE_B_FINGERPRINT_CONTRACT.json` (SHA
`cfc19f84f6ae23b060dc5c33405828b135ec88c828aa77d608d1c09ca63fd925`) requires one selected
Stage-A window to supply an absolute moved-runtime path plus tree SHA, exact archive bytes/SHA, a
receiver-realized frame-1 field (`600 x 3 x 874 x 1164`, uint8), and the receiver-consumed token field
(`600 x 384 x 512`, uint8), all under one fingerprint. JG2 must first reproduce all 600 receiver-consumed
tokens byte-identically in its control before an edited delta is trusted. Pose6 custody was parsed, not
assumed: `gt_poses.npy`, shape `600 x 6`, dtype `<f8`, member SHA
`f73ec194b379a7c04ecf208ac80ab3b1855fe7466ea6eeb7366edafcd824f6a2` inside the pinned GT cache.

## Verification and retention

- Ruff format/check and Python compilation passed for all five changed Python files.
- Focused suite: `108 passed` across the S1A receipt tests, DS1 objective/variation tests, and WD3 tests.
- The payload-retention detector found `0` findings in the three implementation modules.
- Two fresh review-tracker passes cover every entity in all five Python files:
  `s1a-final-pass1-invariants` and `s1a-final-pass2-retention-provenance`.
- The interrupted atomic runtime copy remains preserved at
  `retained/incomplete_runtime_copy_interrupted_20260825/` (non-AppleDouble files numerator `21`,
  retained bytes numerator `202,710`). The first failed parse-back transcript, the legacy-root birth
  preflight config, and both superseded pre-review seals also remain retained. Nothing from GB1, DX2,
  RJ1, Vertigo, or the shipped runtime was mutated.

## Conclusions and boundaries

- Stage A's previously missing interfaces are implemented and sealed; MAIN can compile and fire the
  requests after taking the live lanes and authorization gates.
- No training, frozen-scorer evaluation, Metal run, Modal dispatch, or contest evaluation occurred.
  Consequently there is no distortion result, no score row, and no negative verdict at instance,
  formulation, or family scope.
- Stage A alone cannot establish sub-0.12. Its selected moved object is an input to Stage B, then the
  exact-object compensation/admission chain.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER / STAGE-A-OFF-FLOOR`; **owner:** MAIN; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter/training/`; **fire trigger:** MAIN verifies r5 exit, claims fresh distinct scorer and Metal lanes, compiles the two OFF requests without source drift, sets `launch_authorized=true`, and runs them sequentially.
- **Disposition:** `QUEUED-BEHIND-OFF-FLOOR / STAGE-A-ON`; **owner:** MAIN; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter/training/on_seed_20260815/`; **fire trigger:** both OFF seeds finish with retained receipts and MAIN reviews the measured seed floor plus its unresolvable delta-S before explicitly authorizing the ON request.
- **Disposition:** `QUEUED-BEHIND-STAGE-A / MOVED-FIELD-AND-JG2`; **owner:** MAIN-designated Stage-B moved-field producer; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_b/`; **fire trigger:** one Stage-A seed/window is selected and its moved runtime, archive, realized frame-1 field, and receiver-consumed token field satisfy the emitted fingerprint contract.
- **Disposition:** `QUEUED-BEHIND-STAGE-B / EXACT-OBJECT-COMPENSATION-THEN-ADMISSION`; **owner:** MAIN-designated RJ2/QS5 implementer and MAIN scorer-lane router; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_c/` then `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/admission/`; **fire trigger:** Stage B is mirror-consistent and receiver-closed on one exact moved object, after which compensation is re-solved on that object and the complete A-C chain passes repeat-identity, payload, arithmetic, and single-flight gates.

## LIVE-HYPOTHESES

- Multi-pair trained Film-W96 with in-compile compensation may recover the majority of the pose damage
  that killed untrained rungs. This remains plausible because RJ2 reduced d_pose from `0.0061619673`
  before compensation to `0.0033970401` after one update in
  `/Volumes/APDataStore/pact/ddm_rj2_joint_renderer_object_change/reviewed_replay_r1/SMOKE_RESULT.json`,
  and QS5 established the exact-object compensation law; neither result has been measured on the
  Stage-A trained GB1 object.
- The real uniform cheap-to-shrink treatment may retain lower-bit operating points without paying the
  prior catastrophic distortion. It is plausible because the real WD3 packet ladder and scorer-native
  loss are now in-loop, but the two-seed OFF floor and the ON effect remain unmeasured.
- JG2 on a genuinely moved receiver-consumed field may produce the first negative composed delta. W72's
  token field never moved, so it did not test this condition; Stage A has not yet produced the field.

## DEAD-ENDS

- Sensitivity-waterfill alone is closed as this launch's rung definition: R0 did not establish a reliable
  ordering and uniform int4 is the measured shipped baseline. This is not an allocator-family verdict.
- A single-seed Stage-A verdict is forbidden: WD3 had never sampled seed variance, and restored birth RNG
  makes a post-resume seed edit ineffective.
- Reusing a runtime copy that contains the source archive is closed: the adapter atomically excludes it,
  binds only the candidate archive, and refuses a remaining GB1 archive SHA.
- Carried compensation is closed: the solve must be rerun on the exact object; the integer decision was
  re-proved here and must be rerun again after Stage B moves the object.
- SVD truncation and other mechanism reductions remain closed by NY1; Stage A is the trained W96 path.
- Carrier Brotli q11 and q9/lgwin24 are closed for exact RJ2 stream reproduction; q9/lgwin16 is the
  byte-identical production binding.

**Own-vehicle frontier: UNMOVED — gb1, S `0.14811799921260607` at `180,215 B` `[contest-CUDA T4, n600]`, archive SHA `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`.**
