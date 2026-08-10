# DDM-CP2 composition receiver and harness

## Outcome

DDM-CP2 produced real one-member PR130 archives, not arithmetic projections. The
SM3R low-rank/VQ receiver is wired into the actual temporal-capable
`inflate.sh` runtime, and all three composed archives double-build
byte-identically and parse back through the shipped outer receiver.

The replacement interaction is not uniformly additive. Low-rank plus temporal
and SD1 plus temporal are exactly additive. VQ32 plus temporal costs four more
ZIP bytes than the sum of the independently measured deltas. This is an
INSTANCE-level container interaction, not a distortion verdict.

| exact composition | summed individual delta | additive expectation | actual archive | actual delta | interaction gap | archive SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| pointwise low-rank r32 + temporal reversion | -8,688 B | 182,364 B | **182,364 B** | **-8,688 B** | **0 B** | `2080215145b8958de5ec79f9d2f02777eb14370836989cf4acac2646b5f3f0e4` |
| joint vector/scale VQ32 + temporal reversion | -7,064 B | 183,988 B | **183,992 B** | **-7,060 B** | **+4 B** | `d8bf1fa5f87913bd028693dd9971c841f251437114f05cb93d1edc6c0779c11b` |
| SD1 selected mixed q3/q4 + temporal reversion | -3,264 B | 187,788 B | **187,788 B** | **-3,264 B** | **0 B** | `fc283b6a6a538c74f2f0c30d7cf8ace93bd9496a96a18701e7eb9ed130dc8320` |

Axis for every number in the table: **[scorer-free exact `archive.zip` bytes;
real shipped receiver parse-back]**. The corresponding rate-only changes are
-0.0057849825847254245, -0.00470096420904253, and
-0.0021733636229907673 score units. They are not full-score changes.

## Receiver result

The receiver has exactly the chartered three cases:

1. An absent `SM3R` magic returns `None` and executes the existing legacy/SD1M
   loader unchanged. The temporal legacy-q4 control rebuilt byte-identically at
   188,636 B, and its decoded semantic state SHA is the pinned legacy state SHA
   `23ef13ea9f00217d3b09250096a6f4cb14a0312612be3b50fe62f5d05bdd7933`.
2. `SM3R` version 1 modes 1-3 (vector VQ, scale VQ, joint VQ) and mode 4
   (pointwise low-rank) use the new decoder. The low-rank and joint-VQ retained
   fields decode tensor-for-tensor identically to SM3's packer states.
3. A present `SM3R` field with an unknown version, unknown mode, truncated
   body, invalid selection mask, or trailing bytes raises `SM3RFormatError`.
   SM3's unselected row-prune mode is therefore refused, not guessed.

The CP2 `inflate.py` is a formatted fork of AI1's retained temporal runtime.
Against a formatted copy of that source, its semantic change is limited to two
imports, magic-first width dispatch, and the SM3R-or-legacy loader selection.
All codebooks, factors, scales, indices, and quantized values remain inside the
counted `archive.zip`; no video-derived data moved into free runtime code.

## Real receiver execution

The rank-1 archive completed the real `inflate.sh` path with return code 0 in
**1,086.320886 seconds**, inside the 1,800-second inflate limit. The exact
182,364-byte archive produced a retained **3,662,409,600-byte** RAW with SHA-256
`46ca24e7004c5a3ea42a118981a4fdf6a523e9d5b56cf6baff4444a062176f32`.
The token stage retained its 117,967,988-byte final checkpoint, its
117,967,974-byte progress checkpoint, and its finish receipt. Dependency
closure used the retained PR130 runtime environment (`constriction 0.5.0`,
NumPy 2.3.4, Torch 2.10.0); CPU was selected and no dependency gate was waived.
The sealed receipt is
`/Volumes/VertigoDataTier/pact/ddm_cp2_20260810/pointwise_lowrank_r32__temporal_reversion/receiver_parseback/inflate_receipt.json`
(SHA-256 `5a1efedeec8bdbab3d04d05387290aecac765ddd61839bdba0d2006c50873795`).

Two prior attempts failed closed before decoding and are retained rather than
hidden:

- attempt 1, exit 68: resolving the venv symlink selected the system Python,
  hiding NumPy and Torch under `PYTHONNOUSERSITE=1`;
- attempt 2, exit 2: the repo venv exposed `constriction 0.4.2`, correctly
  failing the shipped `0.5.0` requirement, after which the sandboxed home cache
  prevented bootstrap.

The harness now preserves the venv path without resolving its symlink, pins the
already-retained SSD runtime venv as the default, and records that environment
in the success receipt. No dependency gate was waived.

The governed runner's RSS sampler reported 0 MiB because sandboxed `ps` was not
observable; this is not a valid peak-RSS measurement and is not claimed as one.
The system admission gate ran, the process completed without a kill action, and
the runtime/RAW receipt remains the execution authority.

## Payload custody

Each composition receipt enumerates 20 retained payload records, including both
archive builds, both packed members, both compressed model fields, all source
fields, the composed raw model section, preserved suffix, temporal sidecar,
semantic state, carrier tensors, and staged submission copies. A bounded
rehash verified **60/60 retained payload records** and **27/27 runtime-file
records** across the three builds. All bulk evidence is under
`/Volumes/VertigoDataTier/pact/ddm_cp2_20260810/`.

`tac.payload_retention_gate.scan_paths` examined 4/4 CP2 Python files and found
0 measure-and-discard sites. The first detector pass found one false shape in
`state_wire`: bytes were appended into the persisted state wire, but the local
`tobytes()` plus `len()` shape hid that reachability from the detector. The
implementation now streams a memoryview into the persisted wire, so the cure
clears the detector without a waiver.

## What was not measured

- No scorer ran. `d_seg` and `d_pose` remain unmeasured for every composed
  archive.
- No full S is claimed. Rate-only deltas cannot move the PR130 base.
- No Modal or remote execution ran.
- The full local receiver row is `[macOS-CPU scorer-free]`, not contest
  authority. Any later local scorer result is `[macOS-CPU advisory]`.
- Only the rank-1 composition receives the charter-required full RAW replay in
  this arm. Rank 2 and rank 3 require their own scorer-free full replay before
  their queued scorer jobs fire.

**BASE UNMOVED:** PR130 CPR1 S = 0.172141297491896447 @ 191,052 B
`[contest-CUDA, DALI GT, n600]`.

## Borrowed-substrate accounting

- Base archive/gauge: commit `113b52fdb1`, archive SHA
  `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
- SM3 packer, retained fields, decoded states, and double-build protocol:
  commit `d3650d6c68`.
- Existing SD1M magic-first loader: commit `58f62cd22f`.
- Landed AI1 ANS receiver/resume base: commits `46c7b85219` and `caa8eef4d8`.
- AI1 temporal input was an append-only sibling SSD artifact, pinned by archive
  SHA `0f5a797f...5c84` and by per-runtime-file SHA in every CP2 build receipt.
  It was not edited. CP2 lands its own exact temporal-capable receiver fork.

## RECALL EVIDENCE

The recall was full-corpus rather than charter-only:

- Research memos/receipts were searched by content with
  `PR130|semantic quant|mixed precision|receiver|parse-back|composition|section replacement|temporal_reversion|SM3R|SD1M`.
- `tools/list_canonical_equations.py --json` was searched for composition,
  archive, receiver, and section surfaces.
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`,
  `.omx/state/canonical_task_status.jsonl`, the live hot state, and lane claims
  were searched with the same content surface.

Beyond the charter seeds, recall found SR1's full rendered legacy identity
proof; `ddm_v4b_composed_gate_instrument_fidelity_v1`, which imposes a standing
exact-evaluator re-anchor duty at each composed grammar; and task #827's
measured counterexample where a favorable rate-plus-seg composition concealed
catastrophic pose loss. These changed the handoff: CP2 does not transfer any
semantic-only or rate-only number into a score, omits the prefix screen, and
queues full n600 `d_seg` plus `d_pose` for each exact archive.

## Verification

- 3 real composed archive builds, each double-built byte-identically.
- 3/3 outer receiver model fields and token fields byte-identical on parse-back.
- 3/3 semantic decoded states equal their pinned packer states.
- Focused suite: **4 passed** in 8.05 seconds, including the conditional real
  `inflate.sh` receipt and exact 3,662,409,600-byte RAW geometry check. Pytest
  emitted one environment warning for the disabled plugin's `timeout` config;
  it did not skip or alter a test.
- Ruff and Python compilation pass on all four CP2 Python files.
- `git diff --check` passes on all CP2 source/report paths.

## Follow-on dispositions

The exact ranked fire order is in `SM3_SCORER_QUEUE.json`. Every row is
`QUEUED_WITH_A_FIRE_ORDER`, owned by MAIN after `ddm_ai1` releases the scorer
slot, and names an SSD consumer store and exact archive SHA. No scorer follow-on
was silently fired by this arm.

## LIVE-HYPOTHESES

- Low-rank plus temporal is the strongest first scorer row because its two
  independent replacements compose exactly and bank the largest rate reduction,
  but its low-rank semantic distortion and temporal scorer effects are untested.
- VQ32 can still beat low-rank if its semantic tensors preserve the evaluator
  partition enough to recover the 1,628-byte rate disadvantage; its +4-byte ZIP
  interaction is too small to settle that race.
- SD1 plus temporal is the useful ancestor control because SD1 has a prior
  semantic-leg measurement, but that number cannot transfer across the temporal
  token replacement and its pose leg remains open.

## DEAD-ENDS

- Transferring the charter's -20-byte original-section superadditivity estimate
  is closed: the actual replacements measured 0 B, +4 B, and 0 B interactions.
- Concatenating section byte deltas is closed as an archive measurement: VQ32
  proves it can be wrong even when the sections are logically disjoint.
- Selecting a composed winner from bytes or a prior semantic-only result is
  closed by task #827's omitted-pose counterexample; both scorer legs are
  mandatory.
- The repo `.venv` is closed for this receiver replay because it has
  `constriction 0.4.2`; the pinned retained runtime has the required 0.5.0.
- Unselected SM3R row-prune modes are closed for this receiver revision: unknown
  modes fail closed and may enter only after their own measured scorer win.
