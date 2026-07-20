# Codex findings — Einstein–Kolmogorov crux v3 same-packet closure

**UTC:** 2026-07-20  
**Lane:** `lane_einstein_kolmogorov_crux_v3_20260720`  
**Authority:** `[macOS-CPU advisory] NON-PROMOTABLE`  
**Deliverable classification:** **A — completed and measured**  
**Frontier pointer:** **UNMOVED**

## Verdict first

The undefined `_CP_XI_FX/_CP_XI_CX/_CP_XI_CY/_CP_XI_D` receiver blocker is
closed. The receiver now emits one canonical camera-contract JSON string, binds
that exact string by SHA-256 at receiver startup, then materializes the four
runtime constants. The regenerated packet is byte-for-byte the retained packet:

- **MEASURED** `archive.zip`: **91,062 B**
- **MEASURED** SHA-256:
  `3555bafcccac0827225a87f07dc5b093381de3188560cb7002f2bf9ac2b37c6a`
- **MEASURED** retained-vs-repaired archive identity: exact `cmp`, same SHA-256
- **MEASURED** strict n24 shipped-receiver gate: **bit-exact**, 4 frames
  compared, 0 differing frames, maximum absolute uint8 difference **0**

The exact same packet was then decoded at n600 and scored through the frozen
hard CPU-Torch SegNet/PoseNet oracle:

| Quantity | Value | Status |
|---|---:|---|
| archive bytes | 91,062 B | MEASURED |
| `d_seg` | 0.003555730183919271 | MEASURED, 600 pairs |
| `d_pose` | 126.30360158587386 | MEASURED, 600 pairs |
| Seg term | 0.35557301839192706 | DERIVED by `tac.contest_score` |
| Pose term | 35.53921799728771 | DERIVED by `tac.contest_score` |
| Rate term | 0.060634447989211165 | DERIVED by `tac.contest_score` |
| projected local total `S` | **35.955425463668846** | DERIVED from measured components |
| delta versus 0.1910828242 bank | **+35.76434263946884** | DERIVED; worse is positive |
| headroom versus 264,320 B box | **173,258 B** | DERIVED |

This packet is comfortably inside the requested byte box but is decisively not
a score candidate. Its measured Pose term dominates. The result closes the
measurement obligation; it does not close the xi/pose-carrier family.

## Exact receiver repair

`tools/levelset_byte_close_and_eval.py` now:

1. resolves the four camera values once from the canonical clip profile (with
   the already-documented standalone fallback);
2. serializes them as sorted compact JSON;
3. emits that JSON and its SHA-256 into `inflate.py`;
4. verifies the hash before JSON parsing; and
5. defines `_CP_XI_FX`, `_CP_XI_CX`, `_CP_XI_CY`, and `_CP_XI_D` inside the
   shipped receiver's own namespace.

The JSON contract is generic receiver mechanism and contributes **0 counted
payload bytes**. The archive stays identical because `archive.zip` contains the
same counted `0.bin`; the repaired receiver source has its own measured hash:
`039f474795485ea2aaf6cf8e2a52eb4389acfb7697f1ebb76ffbba6884466811`
(65,459 B).

Tampering only the emitted camera JSON from `fx_native=910.0` to `911.0` while
leaving its digest unchanged is regression-tested to fail closed with
`xi camera contract hash mismatch`.

## Decode and score custody

- Counted `0.bin`: **91,916 B**, SHA-256
  `3f96832dd7e224125d723c654aa920dd1d9457f666c6063c4882754c10921a7e`
- Full raw: **3,662,409,600 B**, SHA-256
  `e161990ed1c0071a156b3a071758f25b5a7f56bfd1f6c9da69ff1fc47b09e8e6`
- GT cache: **5,078,017,610 B**, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`
- Scorer: `tools.levelset_byte_close_and_eval.parity_on_inflated`, CPU,
  Torch 2.12.1, 6 Torch threads
- Raw generation: the exact shipped `inflate.py` from the repaired packet,
  1,200 frames at `874x1164x3 uint8`

The run was checkpointed at packet, full-raw, and score boundaries. The full raw
was hashed and its rebuild command/source hashes were written before deletion;
after the score receipt landed, the 3.66 GB rebuildable raw was automatically
removed. The small packet and all machine-readable custody receipts remain.

## Adversarial note: two different bit-identity checks

The requested **strict n24 gate is green**. A separate post-hoc
`pose_carrier_confirm` field in the capped n24 report says
`frame0_decode_bit_exact=false` with maximum difference 68. These are not being
silently conflated.

**DERIVED implementation diagnosis pending the required external review:** the
capped `run_inflate(max_pairs=24)` path re-quantizes and writes a capped extracted
blob, but `run()` passes the original full blob and full pose-carrier bytes to
the later confirmation helper. The strict gate compares the shipped capped blob
against an oracle dequantized from that same capped blob, so it is the correctly
matched receiver/oracle test. The post-hoc confirmation appears to compare the
capped raw to an oracle regenerated from a different blob. This is a
**capped-confirmation formulation finding**, not evidence that the full n600 hard
score was borrowed: the n600 measurement used `max_pairs=None` and the hard
oracle read the actual full shipped raw directly.

This diagnosis must be confirmed or falsified by the one required high-effort
external review before MAIN landing. If confirmed, the narrow fix is to pass the
exact extracted/scored blob and its pose-carrier section into
`pose_carrier_confirm`; no packet or score bytes change.

## Measured / derived / assumed boundary

- **MEASURED:** archive/receiver/raw bytes and hashes; strict n24 zero-diff gate;
  all-600-pair `d_seg` and `d_pose`; cleanup state.
- **DERIVED:** canonical score terms, projected total `S`, bank delta, byte-box
  headroom, and the capped-confirmation code-path diagnosis above.
- **SETTLED INPUT:** comparison score 0.1910828242 and byte box 264,320 B.
- **ASSUMED:** none.

No contest-CPU Linux x86_64 or contest-CUDA score is claimed. No provider job,
paid dispatch, optimizer, or frontier-pointer update occurred.

## Triality and system wire-in

- **DSL:** the packet configuration is unchanged; the existing typed xi bridge
  contract supplied the retained donor/generator/cache bindings.
- **DAG:** this v3 task consumes the existing v2 blocker edge and lands the
  measured terminal receipt; it does not create a new optimization branch.
- **Equations:** score composition is exclusively
  `tac.contest_score.compute_contest_score`, with no hand-rolled authority path.
- **Continual learning:** the negative result is encoded as machine-readable
  measured components plus a formulation-scoped warning; it does not close the
  xi family or authorize a pointer move.

## Verification and artifacts

- Repair commit: `80842be655` (`fix: bind xi receiver camera constants`)
- Measurement harness/config commit: `87cf1bda86`
- Repair tests: direct template execution, emitted-source presence, unhashed
  mutation refusal, strict tiny store-nothing bit-exact gate, clip-profile value
  binding, Python compile, and focused Ruff safety rules all green.
- Python review tracker: two clean passes on every changed Python file.
- Machine receipt:
  `.omx/research/einstein_kolmogorov_crux_v3_20260720.json`
- Full decode stage:
  `.omx/research/einstein_kolmogorov_crux_v3_n600_inflate_stage_20260720.json`
- Hard-oracle score stage:
  `.omx/research/einstein_kolmogorov_crux_v3_n600_score_stage_20260720.json`
- Reproduction config:
  `.omx/research/einstein_kolmogorov_crux_v3_measurement_config_20260720.json`

## MAIN landing requirement

This work is only on
`codexwt/einstein_kolmogorov_crux_v3_20260720T075336Z`; it is **not on MAIN**.
MAIN must review the immutable commits, the external adversarial review, the
strict-gate/post-hoc-confirmation distinction, and the three measurement
receipts before merging. The pointer remains unchanged after any merge unless a
separate contest-axis evaluation supplies promotion-grade evidence.

## STORES CONSULTED

- delegated authority file and its verified SHA-256/byte count
- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and craft handoff manual
- v7.5 and v10 canonical specs
- v2 findings, failure receipt, retained packet, runtime snapshot, and launch
  manifest
- `reports/latest.md`, lane registry, subagent progress ledger, broadcast/per-arm
  inboxes, current MAIN head, and forensic Einstein implementation branch
- exact generator checkpoint, donor R1 checkpoint, GT cache, shipped packet,
  decoded raw, and hard scorer modules
