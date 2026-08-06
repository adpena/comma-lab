# ddm_hb1 2026-08-06 receipt

## Result

HB1 did not land a PR130 HPAC byte row on OUR tq1c or GT label payloads. The
best measured incumbent remains PP1 KT temporal context-arith:

| payload | best measured incumbent | bytes | S_rate | HPAC total on OUR payload | HPAC status |
|---|---|---:|---:|---:|---|
| tq1c parent argmax labels | PP1 KT temporal context-arith | 142001 | 0.094552637 | not measured | BLOCKED locally: no CUDA and no trained OUR-label HPAC checkpoint |
| GT `lstars` labels | PP1 KT temporal context-arith | 173617 | 0.115604434 | not measured | BLOCKED locally: no CUDA and no trained OUR-label HPAC checkpoint |

No adoption, no exact score claim, and no pointer move. This was byte-only and
scorer-free: `upstream/evaluate.py` was not run and no SegNet/PoseNet forwards
were run.

## Race Table

Primary table: `.omx/research/ddm_hb1_20260806/BYTE_RACE_TABLE.md`.

Summary against each incumbent:

| payload | coder | bytes | S_rate | delta bytes vs incumbent | delta S_rate vs incumbent | equality |
|---|---|---:|---:|---:|---:|---|
| tq1c parent argmax labels | PP1 KT temporal context-arith | 142001 | 0.094552637 | 0 | 0.000000000 | n600 closed-form plus n=6 range proof |
| tq1c parent argmax labels | Brotli q11 raw uint8 | 368760 | 0.245542148 | 226759 | 0.150989510 | whole-stream PASS |
| tq1c parent argmax labels | LZMA1 preset 9 extreme raw uint8 | 354900 | 0.236313342 | 212899 | 0.141760705 | whole-stream PASS |
| tq1c parent argmax labels | PR130 HPAC | not measured | not measured | not measured | not measured | not run |
| tq1c parent argmax labels | SMEVR | not measured | not measured | not measured | not measured | not run |
| GT `lstars` labels | PP1 KT temporal context-arith | 173617 | 0.115604434 | 0 | 0.000000000 | n600 closed-form plus n=6 range proof |
| GT `lstars` labels | Brotli q11 raw uint8 | 424728 | 0.282808941 | 251111 | 0.167204508 | whole-stream PASS |
| GT `lstars` labels | LZMA1 preset 9 extreme raw uint8 | 409989 | 0.272994846 | 236372 | 0.157390412 | whole-stream PASS |
| GT `lstars` labels | PR130 HPAC | not measured | not measured | not measured | not measured | not run |
| GT `lstars` labels | SMEVR | not measured | not measured | not measured | not measured | not run |

The imported TK1 semantic-stream race reports aggregate wall-clock
`171.40160866687074` seconds for the completed KT/generic work. It does not
store per-cell encode/decode timing for each imported cell, so HB1 does not
invent per-cell timings.

## Source Proof

Payloads:

| payload | shape | dtype | raw sha256 | file sha256 | path |
|---|---:|---|---|---|---|
| tq1c parent argmax labels | 600x384x512 | uint8 | `a7dd6f4271eedfa877f6499348de5f9dae2d97311f9e98f4f534908eb66e044e` | `764a244c4890b22a67c4dbe95a959e970c29328778d41ffe4deb85f5b650eee6` | `/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy` |
| GT `lstars` labels | 600x384x512 | uint8 | `f2c8be94774780bda718adf337900403a8533b6ffa1352b5aae19e200a005557` | `b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d` | `/Volumes/VertigoDataTier/pact/ddm_ph1_lstars_u8.npy` |

PR130 external anchor, verified but not adopted as an OUR-payload result:

| artifact | bytes | sha256 |
|---|---:|---|
| retained PR130 token stream | 116980 | `948379872ff81a4e5d948ec301c143be00ebd0033544c8abdfb4af0f4c4a15eb` |
| retained PR130 packed HPAC model blob | 15164 | `ef8bb9d59bdd3916fb77713c11cdcb85e029f01d80b82472a40ab28f7e56a9ee` |
| retained PR130 HPAC checkpoint | 177041 | `0f4775920aeb2fb419555cc4d68703dd90b88be9d24c82466a99fddc1b1f1aa7` |

The PR130 intake reports the packed model as the shipped artifact and the raw
HPAC model as 20179 bytes inside the raw model bundle. The 116980 token bytes
plus 15164 packed model bytes equal 132144 bytes for the PR130 payload only.

## Execution Boundary

Local hardware check:

```json
{
  "cuda_available": false,
  "cuda_device_count": 0,
  "mps_available": false,
  "platform": "macOS-26.4-arm64-arm-64bit-Mach-O",
  "torch": "2.12.1"
}
```

Reason HPAC was not run: the charter requires PR130 optimal-form HPAC on full
n600 payloads. Running the retained PR130 training recipe here would require
changing the compute boundary away from CUDA or running a long CPU-only
substitute. That would not be the requested PR130 optimal-form row.

Reason SMEVR was not run: the in-tree R7 SMEVR token coder is scoped to compact
token tensors and enforces a value-count cap below the full `[600,384,512,1]`
label-map object. HB1 did not create a new SMEVR label-map implementation during
this unit, because that would be a new coder surface rather than the requested
existing race cell.

## Recall Evidence

Read or checked in this unit:

| source | use |
|---|---|
| `.omx/tmp/codex_runs/hb1_prompt.md` | charter and required outputs |
| `.omx/tmp/codex_runs/_common_contract.md` | scorer-free boundary, protected files, receipt/commit requirements |
| `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md` | governing Pact constraints |
| `.omx/state/main_hot_state.md` | live queue/frontier authority; hb1 scorer-free is compatible with occupied scorer lanes |
| `.omx/research/ddm_eh1_20260806/EUREKA_TABLE.md` | row 3 HPAC target plus rows 4/6 stretch target |
| `.omx/research/ddm_eh1_20260806/CURE_TABLE.md` | HPAC and IX2TOK01 context |
| `.omx/research/ddm_tk1_20260806/RECEIPT.md` | existing byte-only semantic stream ladder for the exact same label payloads |
| `.omx/research/ddm_tk1_20260806/semantic_stream_race.json` | source digests, completed generic rows, aggregate wall-clock |
| `.omx/research/ddm_pp1_direct_partition_pricing_20260728.md` | PP1 KT context-arith provenance |
| `.omx/research/pr86_pr130_fullstack_intake_20260728.md` | PR130 archive/member/model/token accounting |
| `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/scripts/e2e.py` | stage-41 to stage-44 retained HPAC command chain |
| `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/scripts/train.sh` | compact retained HPAC self-compress, pack, and token commands |
| `experiments/ddm_r7_token_coder.py` | SMEVR implementation boundary |
| canonical equations/search over `.omx/research` | found adjacent rate laws and prior PP1/TK1 rows, but no measured HPAC-on-OUR-label row |

## Follow-On State

Follow-on file: `.omx/research/ddm_hb1_20260806/NEXT_IF_RESUMED.md`.

Rows 4 and 6 stretch work was not run. Both are queued behind a completed
OUR-payload HPAC model with exact token decode:

| stretch | HB1 status |
|---|---|
| CPR1-style Huffman/Rice repack on low-rank carrier | queued, blocked on base HPAC row |
| bit-depth self-compression of HPAC model | queued, blocked on base HPAC model; require max logit diff `0.0` |

## Own-Vehicle Frontier

Own-vehicle frontier unchanged. Live macOS-CPU advisory row remains tq1c at
`S = 0.7534578126155775 @ 357837 B`; contest-CPU pointer remains the borrowed
`0.19108` family. HB1 produced no byte-closed archive and no public-evaluator
row.

