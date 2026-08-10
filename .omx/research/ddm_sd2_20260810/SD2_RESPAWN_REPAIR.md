# ddm_sd2 respawn repair — Range/ANS checkpoint capability binding

## Outcome

The runner is repaired and scorer-free preflight is green. Both retained fire inputs are Range
archives according to the exact materialized receiver at commit
`58f62cd22ff07562c0534c999d705fb9edfe5279`. The runner now leaves
`PR130_TOKEN_CACHE` and `PR130_TOKEN_RECEIPT` unset for those archives, so the receiver performs
the valid sequential Range decode. ANS archives still receive the paired cache variables and keep
their periodic stack checkpoint path.

The receiver guard was not changed. Range still has no compatible intra-decode checkpoint in this
unchanged archive. A crash during Range decode costs a full replay of that decode; the 60-pair
scorer chunks remain independently resumable. Failed receiver attempts remain retained and future
failure receipts state the correct replay boundary.

No full receiver decode, SegNet, PoseNet, or contest evaluator ran during this repair. No d_seg,
d_pose, edge matrix, S value, or pointer movement was measured. `score_claim=false`.

## Real retained-payload proof

| Candidate | Retained member | Bytes | SHA-256 | Pinned receiver result | Checkpoint env |
|---|---|---:|---|---|---|
| PR130 q4 control | `retained/candidates/pr130_q4_control/decode_input/p` | 190,952 | `fcc6a3c242106350077d3c328a2c07c8994c86d19b1a361f8507917de6ba3d84` | `token_codec=range`, `model_codec=legacy_lzma` | none |
| mixed q3/q4 n600 | `retained/candidates/sd1_selected_mixed_q3q4_n600/decode_input/p` | 190,104 | `590af0611f7fdb40a7c6efcaf28be5fe1a0d808ff532915d729cdf2fd6bc3037` | `token_codec=range`, `model_codec=legacy_lzma` | none |

The first failed attempt remains immutable at
`/Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600/retained/decode/pr130_q4_control/attempts/attempt_0001/`.
Its 1,490-byte log SHA-256 is
`8bb9bc56f4bd8ce86d99104b493dbeb5fa1f0bc2bd8bf7d4aa464dbd1d84e54f`. Its historical failure
receipt says a token-checkpoint retry was safe; that sentence is the diagnosed bug and is not
reused as truth. New failure receipts distinguish ANS resume from Range replay.

The existing progress receipt contains only `retention_preflight`, no completed decode, no scorer
chunks, and no active chunk. Because the repair changes the runner hash, full fire records a
runner-fingerprint migration only under exactly that pre-decode state. Any other fingerprint
change, completed decode, retained chunk, or active chunk refuses migration.

## Validation

- `pytest`: 15 passed.
- Ruff, Python compilation, and `git diff --check`: PASS.
- `tac.payload_retention_gate`: 8 files scanned, zero findings.
- Live writer probes: atomic byte, NumPy payload, byte range, and JSON receipt PASS on
  APDataStore.
- Scorer-free `--plan-only`: `READY_FOR_MAIN_FIRE`, receipt
  `/Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600/SD2_RETENTION_PREFLIGHT.json`,
  17,938 bytes, SHA-256
  `b90fc80d5be447df980df80ce759c373aabedef1902da9e1cc390b61044c02dd`.
  The receipt itself records both real inputs as Range and zero checkpoint environment keys.
- Free bytes at preflight: `1,070,822,326,272`. Conservative remaining requirement:
  `27,858,793,811` bytes, including a 3,662,409,600-byte failed-decode contingency and a
  5,000,000,000-byte reserve. Full paired n600 retention fits.

Machine-readable proof:
`.omx/research/ddm_sd2_20260810/SD2_RESPAWN_REPAIR_RECEIPT.json`.

## Exact MAIN fire command

```bash
/Users/adpena/Projects/pact/.venv/bin/python /Users/adpena/Projects/pact/experiments/ddm_sd2_pr130_seg_decomposition_runner.py --out-dir /Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600 --resume-from /Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600/progress.json --queue /Users/adpena/Projects/pact/.omx/research/ddm_sg2_20260810/SG2_SCORER_QUEUE.json --base-archive /Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip --candidate-archive /Volumes/VertigoDataTier/pact/ddm_sd1_semantic_20260809/cpu_screen/archives/selected_mixed_n600.zip --challenge-root /Users/adpena/Projects/pact/upstream --video-names-file /Users/adpena/Projects/pact/upstream/public_test_video_names.txt --uncompressed-dir /Users/adpena/Projects/pact/upstream/videos --device mps --decode-device cpu --batch-size 4 --chunk-pairs 60 --pair-count 600 --seed 20260810 --cpu-threads 6 --num-threads 2 --prefetch-queue-depth 4 --minimum-free-bytes 5000000000
```

Disposition: `QUEUED-WITH-A-FIRE-ORDER`. Owner: MAIN scorer owner. Consumer store:
`/Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600`. Fire trigger: the serialized repair
commit is present and MAIN's sole full-n600 scorer slot is free. Lane
`lane_ddm_sd2_seg_decomposition_20260810` remains MAIN-claimed and is reused.

Projected runtime remains approximately 45–90 minutes on MAIN Metal: two serial CPU receiver
passes plus one unmeasured paired target/base/candidate MPS scorer pass. The prior same-family
single n600 decode measured about 1,011 seconds; this runner's total remains unmeasured.

## RECALL EVIDENCE

Searched the full `.omx/research/` corpus by content for `ddm_sd2`, `PR130`, `argmax retention`,
`seg decomposition`, `token_codec`, `progress checkpoint`, `Range`, and `ANS`; queried the canonical
equations registry; searched `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, specifications, task and
lane state, and inspected the exact materialized receiver.

Beyond the original charter seeds, the search found:

- `src/tac/pr130_runtime/ddm_hp3_runtime/inflate_hp3.py` resumes Range only because HP3 changes the
  counted token stream to include an explicit seek checkpoint. That mechanism cannot be borrowed
  for an unchanged PR130 archive, so the repair does not pretend Range is intra-decode resumable.
- `experiments/ddm_cp2_composition_receiver_and_harness.py` sets the token-cache environment for an
  ANS composition archive. That confirmed the environment is a codec capability request, not a
  generic receiver-resume switch.
- The pinned `receiver.split_payload` already returns `token_codec`; the retained PR130 and mixed
  q3/q4 members both parse as Range. This changed the patch from candidate-name or header guessing
  to source-verified receiver parsing.
- No canonical equation changes the capability boundary. The relevant invariant is byte/object
  identity: unchanged Range bytes cannot acquire HP3's counted seek state for free.

## Boundaries and frontier

The queued MPS result will be `[macOS-CPU advisory]`/diagnostic, never contest authority. The PR130
base remains `S=0.172141297491896447 @ 191,052 B [contest-CUDA, DALI GT, n600]`. This repair did not
move the exact pointer.
