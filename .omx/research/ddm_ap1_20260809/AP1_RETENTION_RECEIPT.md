# DDM AP1 retained-ANS custody receipt

**Disposition:** `REAL OBJECT RETAINED AND ROUND-TRIP VERIFIED; AP1 GOVERNED RELAUNCH BLOCKED`

**Axis:** `[macOS-CPU advisory, scorer-free]`  
**Score claim:** `false`  
**Pointer movement:** none

AP1 has a durable 114,860-byte ANS token payload and the 116,980-byte shipped Range token
payload under its required SSD root.  The DT1 producer regenerated Range and ANS from the full n600
int16 lattice at constriction 0.5.0, then replayed both through the real receiver.  Both reconstructed
all 117,964,800 symbols exactly with decoded SHA-256
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`; the ANS replay also passed
the receiver's fail-closed final-state check.

AP1 did not launch a second child.  The charter's provenance pin does not reproduce: commit
`5de03569ad` contains the required Range/ANS and model-codec bitfield dispatch, but it does not
contain the charter's stated `absent -> legacy`, `r7_smevr_v1 -> SMEVR`, other -> refused grammar.
The charter says a nonreproducing pin is a STOP (`verdict_scope=INSTANCE`: this charter mapping at
this commit, not the ANS family).  The DT1 producer also used a bare shell
launch rather than `tools/launch_detached_process.py`, so its result cannot be represented as an
AP1 governed-launch receipt.

## Retained objects

| object | AP1 path | exact bytes | SHA-256 | status |
|---|---|---:|---|---|
| ANS token payload | `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/retained/ans_n600.bin` | 114,860 | `a0b18dc0803ef541d3eb265bba5380f7aa067593f6af584b0891ded5bdd74488` | retained and n600 round-trip verified |
| Range token payload | `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/retained/range_n600.bin` | 116,980 | `948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb` | retained shipped oracle; directly archive-equal |
| AP1 chunk manifest | `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/retained/chunk_manifest.json` | 16,813 | `ad2993eb9aa0856e152849361349c4549e7abeb3b8eede9e95c10f5bf241ce86` | complete, AP1 paths |
| coder checkpoints | `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/retained/coder_checkpoints.json` | 4,496 | `21c199649276e4ef15f3d9db7602dbe58b965ccc9111cf32cfcbeb9af0d69983` | 29 boundary states covering 28 intervals |
| terminal n600 receipt | `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/retained/retained_n600_result.json` | 15,828 | `5c15f38ab68df68c09a5859d17d19e4247f90e76457282edccbc8a34d060916c` | complete; hard-linked source receipt |
| AP1 custody receipt | `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/AP1_CUSTODY_RECEIPT.json` | 4,678 | `47a9f0866449f57f054a701e6d4dcbe9ef6a832c2e6ce2ce86514fac1a16892d` | machine-readable handoff and blockers |

The 28 chunk pairs are hard-linked into AP1 custody from the DT1 producer, so this adoption adds no
duplicate 1.30 GB allocation and survives deletion of either directory entry.  The manifest covers
frames 0 through 600 contiguously, 117,964,800 symbols, 117,968,384 symbol-file bytes, and
1,179,651,584 int16-code-file bytes.  Every chunk row carries both file hashes.  Maximum checkpoint
span is 24 frames.

The source DT1 manifest is
`/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/chunk_manifest.json`, SHA-256
`23089d6f627e1da56a3f947900727e94ee4a99d1a2ce30fd582aeeac3130caea`.  AP1 changed only the
absolute custody paths in its copied manifest; all 56 `.npy` files were verified to share the source
inodes.

## Identity proof status

### Range byte identity

**Closed, with lineage kept explicit.**  AP1's `range_n600.bin` is the pre-existing recorded/shipped
oracle, not a separately retained encoder output.  A direct pinned-receiver split of archive member
`p` measured a 116,980-byte token field with the same SHA-256 and byte-for-byte equality to this AP1
file.  Separately, the DT1 encoder rebuilt the Range
stream from every retained chunk and raises unless the resulting bytes equal the archive's shipped
token field.  It performs this exact comparison before it constructs or atomically writes
`ans_n600.bin`.  The terminal receipt records `streams.range.byte_identical_to_shipped=true`,
116,980 bytes, and SHA-256 `948379...15eb`.  The regenerated in-memory blob was not written as a
second file; AP1 retains the proven-equal oracle sequence.  The earlier length-only ANS experiment
is not evidence for this byte identity.

### ANS symbol identity and terminal state

**Closed on the retained bytes.**  The terminal receipt records `complete=true`, 600 frames,
117,964,800 tokens, `all_tokens_reconstructed=true`, and `exact_target_equality=true` for both Range
and ANS.  Both decoded tensors have SHA-256
`c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.  Before constructing either
decode result, the producer calls `receiver.finish_token_decode`; pinned receiver SHA
`d968...ac4f` raises if ANS has retained state.  Therefore the terminal ANS result also proves final
decoder exhaustion.  Measured wall times were 741.953 seconds for Range and 804.876 seconds for ANS
on this `[macOS-CPU advisory, scorer-free]` surface.

## Provenance pins

- Prior length receipt verified:
  `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/ans_n600/ans_vs_range_n600_result.json`,
  184 bytes, SHA-256
  `8816f91afcc21060753a6612cda4e1b7f3b483a7aa073cbfa1b9b5d7e520d451`.
- Source archive verified:
  `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`, 191,052
  bytes, SHA-256
  `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
- Actual receiver at `5de03569ad` verified: `receiver.py` SHA-256
  `d9689091430b31b37f5f12d2eaa8025187f7f08899ae1b99ba43a30480b7ac4f`.  It uses bit 31 for
  Range/ANS, bits 29-30 for legacy/split-Brotli/split-LZMA2 model codecs, refuses model selector 3,
  and rejects a nonempty final ANS state.
- Charter-described SMEVR selector pin: **does not reproduce at `5de03569ad`; STOP**.  That grammar
  belongs to a different runtime surface.
- The producer interpreter is CPython 3.11.15 with constriction 0.5.0, NumPy 2.3.4, and Torch 2.10.0
  on arm64.  Repository `.venv` has constriction 0.4.2 and is not an admissible AP1 runtime.
- AP1 preserved the exact producer script at
  `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/provenance/ddm_dt1_ans_decode_wallclock.py`,
  SHA-256 `29fe64180f08bf3406e9f9122d95e00fef560ac828d4b18e740a98492bd6d466`.
- AP1 also extracted the eight runtime files from `5de03569ad` under
  `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/provenance/runtime_5de03569ad/`; the preserved
  `inflate.py` SHA-256 is
  `48218c360fa26f9d3d7fec76db54be0e7652b53e8f9cbd48245e488d7494770f`.

## Launch and resumability boundary

The producer command was:

```text
/Volumes/VertigoDataTier/pact/ddm_pq1_runtime_20260809/venv/bin/python \
  experiments/ddm_dt1_ans_decode_wallclock.py --mode retain --torch-threads 4 \
  --output /Volumes/VertigoDataTier/pact/ddm_dt1_20260809
```

It was not launched through the charter-required detached launcher and has no detached launch
manifest or watcher `.done` receipt.  Its table stage is genuinely crash-resumable: complete hashed
chunks are verified and skipped on restart.  Its encode checkpoints are emitted only after both
streams finish, and its decode progress is observability rather than restorable entropy/model state.
Those latter two limitations prevent an honest claim that the whole job is crash-resumable
(`verdict_scope=INSTANCE`: this producer and invocation, not retained entropy coding generally).

The producer also did not call the already-built
`receiver.encode_ans_code_chunks_reverse` helper.  It reused the receiver's exact int16-table
rehydration but implemented the reverse row/batch loop locally.  The terminal equality result proves
this loop's output is valid; it does not close the charter's helper-reuse requirement or provide a
full-n600 byte comparison between the local loop and the helper.

AP1 therefore preserved the completed chunk lattice and payloads but did not manufacture a launcher
receipt after the fact.  A corrected pin plus a landed resumable verifier is required before a new
AP1 child may fire.

## Optimal-form assessment

The scientific object reached the reference scale: full n600, both Range and ANS, real PR130
conditional tables, pinned constriction 0.5.0, retained payloads, and exact receiver replays.  It did
not reach full charter operating form because the invocation was not governed, the entire encode and
decode sequence was not crash-resumable, the named reverse-chunk helper was not used directly, and
the selector pin is false as written.  These are custody/operating-contract gaps; they do not reverse
the measured byte or symbol identities.

## CX2 handoff

CX2's live input validator now reads the producer's actual fields:
`streams.range.byte_identical_to_shipped`, `ans_decode.exact_target_equality`,
`ans_decode.all_tokens_reconstructed`, and `provenance.host.constriction`.  Its accepted-input store
is `/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/stages/01_inputs.json`.

Evidence-only handoff is `QUEUED-WITH-A-FIRE-ORDER`: owner `ddm_cx2`; consumer checkpoint
`/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/stages/01_inputs.json`; fire by running CX2's own
input validator against the source pair
`/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/ans_n600.bin` and
`/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/retained_n600_result.json`.  Its data trigger
is now satisfied: the receipt is complete, both decode proofs are exact, constriction is 0.5.0, and
the payload stat/hash matches `streams.ans`.  AP1 and DT1 copies of both the payload and terminal
receipt were verified as same-inode hard links.  The full CX2 fire trigger is not satisfied because
its validator remains sibling-owned and untracked; it must be reviewed and landed before it writes
the accepted-input checkpoint.  This evidence-only consumption is not a waiver of AP1's false
charter pin or governed-launch debt.  No composed archive or evaluator score is claimed here.

## Follow-on dispositions

- `FIRED` — owner `ddm_dt1`; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_dt1_20260809/retained/retained_n600_result.json`; fire trigger was
  the measured affordable timing gate; terminal trigger is satisfied with both exact n600 decodes.
- `QUEUED-WITH-A-FIRE-ORDER` — owner `ddm_cx2`; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_cx2_20260809/stages/01_inputs.json`; fire trigger is a reviewed,
  landed CX2 validator consuming and rehashing the now-valid DT1 payload/receipt pair above.
- `QUEUED-WITH-A-FIRE-ORDER` — owner `MAIN/charter author`; consumer store
  `.omx/research/charters/ddm_ap1_ans_payload_retention.md` or a durable signed erratum beside it;
  fire trigger is an explicit correction or waiver of the false selector pin.
- `QUEUED-WITH-A-FIRE-ORDER` — owner AP1; consumer stores
  `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/run/launch_manifest.json` and
  `/Volumes/VertigoDataTier/pact/ddm_ap1_20260809/retained/AP1_GOVERNED_VERIFIER_RESULT.json`; fire
  trigger is the charter correction/waiver plus a landed verifier that directly reuses the canonical
  reverse-chunk helper with restorable encode and decode stage state.

## RECALL EVIDENCE

Searched the full research corpus, not only the charter seeds, with these content queries:

- `stack.AnsCoder|encode_reverse|AnsCoder|ANS payload|range coder|114860|116980|2120|HPAC`
  across `.omx/research`, `src`, `experiments`, and `.omx/state`;
- `ddm_ap1|ddm_rc1|ddm_dt1|ans payload|ans token|ans retain` across the canonical research indexes,
  `sub015_DAG_*`, SPECs, designs, and task stores;
- `ddm_r7_token|SMEVR|order-1 ANS|CONSTRICTION_ORDER1_ANS|r7_token_coder` across the DAG and index;
- all 429 rows from `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for ANS,
  entropy-coder, Range-coder, archive-byte, and rate-score surfaces;
- exact AP1/DT1/CX2 ownership terms across the arm queue, task status, lane registry, active-dispatch
  claims, P0 ledger, probe outcomes, and the dated harness bridge.

Findings beyond the charter seeds changed the plan materially:

1. DT1 had already implemented and fired the exact full-n600 retention mechanism.  AP1 adopted its
   immutable chunks instead of rematerializing roughly 1.30 GB.
2. The shipped Range oracle already existed with exact path and SHA, and independent archive-tail
   comparison had closed its identity.
3. CX2 briefly expected nonexistent DT1 JSON fields; the sibling owner corrected that interface, so
   AP1 did not add an adapter.
4. Prior R7 static-rANS and SMEVR negatives apply to different token representations and do not
   transfer to PR130's per-symbol HPAC tables.
5. The canonical equation registry has no AP1 retained-object equation.  The nearby static-packet
   byte-delta equation supplies arithmetic only, not payload or receiver custody.
6. The live board overstates the old Range proof and says the receiver is unbuilt.  Primary receipts
   show the old proof was length-only and commit `5de03569ad` built the receiver.  AP1 did not edit
   the managed board while sibling work was active.
7. The charter's selector description is from another runtime and fails its own pin rule.  That
   changed AP1 from a planned governed rerun to a STOP plus durable custody handoff.

## Measurement boundary

Measured here: exact file sizes and SHA-256 values, archive and receipt pins, complete chunk coverage,
hard-link identity, pinned environment versions, Range/archive byte identity, retained-ANS decode
identity, and the fail-closed ANS terminal state.  The local decoder timing delta was +62.924 seconds
(ANS/Range ratio 1.08481), within the producer's measured macOS inflate-only headroom; this is not a
contest-host timing claim.  Not measured here: SegNet, PoseNet, `upstream/evaluate.py`, contest
CPU/CUDA, a composed archive, or a new score.  The PR130 baseline remains
`S=0.172141297491896447` at 191,052 bytes `[contest-CUDA, DALI GT, n600]`; AP1 does not move it.
