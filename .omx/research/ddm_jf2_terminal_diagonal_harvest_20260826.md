# DDM JF2 — the terminal trained diagonal is byte-alive

JF2 falsified the pre-registered all-negative prediction. Three of the seven epoch-60 joint field-and-model refits are below the `127,292 B` token-subsystem gate after the real JF1 model pack, native RC64 stream encode, and exact production-token decode:

- `k002500`: `127,124 B` (`-168 B`)
- `k040000`: `126,350 B` (`-942 B`)
- `k060000`: `125,716 B` (`-1,576 B`, best)

This is a byte-leg verdict only. JF2 ran no SegNet, PoseNet, Modal job, scorer, or `upstream/evaluate.py`; `d_seg`, `d_pose`, and complete `S` are unmeasured and are not projected. The axis for every row below is `[macOS-CPU advisory / scorer-free exact model-pack and RC64 measurement]`, population `n600`.

## Terminal physical rows

`vs E0002` is terminal total minus the same arm's epoch-2 total. E0002 did not contain `k005000` or `k020000`, so those two comparisons are explicitly absent.

| arm | model B | stream B | total B | vs 127,292 B | E0002 total B | vs E0002 B | exact token decode | byte disposition |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `null` | 13,463 | 114,143 | 127,606 | +314 | 134,679 | -7,073 | yes | byte-negative control |
| `k002500` | 13,487 | 113,637 | 127,124 | **-168** | 133,907 | -6,783 | yes | **BYTE-WINNER** |
| `k005000` | 13,442 | 113,889 | 127,331 | +39 | not measured | not available | yes | negative by 39 B |
| `k010000` | 13,442 | 114,236 | 127,678 | +386 | 134,993 | -7,315 | yes | negative |
| `k020000` | 13,440 | 114,632 | 128,072 | +780 | not measured | not available | yes | negative |
| `k040000` | 13,438 | 112,912 | 126,350 | **-942** | 132,292 | -5,942 | yes | **BYTE-WINNER** |
| `k060000` | 13,398 | 112,318 | 125,716 | **-1,576** | 130,007 | -4,291 | yes | **BYTE-WINNER / best** |

The null positive control improved by `7,073 B` from E0002 but still misses the subsystem gate by `314 B`; its refit stream alone is `366 B` larger than the shipped null stream. The successful trained-diagonal falsifier therefore comes from the non-null fitted fields, not a repaired null control.

## Machine receipts and custody

- Terminal receipt: `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/BYTE_DIAGONAL_TERMINAL.json`, SHA-256 `0a1cc640ad237374fec2bd4ce8d523d37ccf4584d9bd417a4b79b104f4b4dba5`. It is byte-identical to the native JF1 finalizer output `BYTE_DIAGONAL.json`.
- Fixed comparator, unchanged: `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/BYTE_DIAGONAL_SCOPE_E0002.json`, SHA-256 `ac48c6005eedf4e5459b9ab6e0228dc808b9cc295a74aa6a1fa515e9e0cf15e0`.
- Storage preflight: `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/JF2_STORAGE_PREFLIGHT.json`, SHA-256 `d091910b58e0ba832cde979ca4297f543bd9178d44b18efc6c6c893b6679f653`; APDataStore was selected with `39,929,643,008 B` free, `10,737,418,240 B` requested, and a `4 GiB` reserve.
- Durable bulk root: `/Volumes/APDataStore/pact/ddm_jf2_terminal_diagonal_harvest/retained/`. Every arm has a complete `216M` row tree containing the result JSON, every q0-q11 model candidate, selected model, native stream, archive/runtime, resume state, decoded `117,964,800`-byte token field, and five class masks.
- Receipts mirror: `/Volumes/APDataStore/pact/ddm_jf2_terminal_diagonal_harvest/receipts/`. The terminal and E0002 JSON hashes match the source hashes above.
- Verification: recursive `diff -qr` passed for all seven source/destination row trees after deleting only copy-created AppleDouble `._*` sidecars under the new JF2 destination and disabling their recreation. No research payload was deleted or moved; local and APDataStore copies both remain.

## Payload hashes

The table records the selected physical model payload, native RC64 stream, measurement archive, and exact decoded field for every arm. All hashes are SHA-256.

| arm | selected model | native stream | candidate archive | decoded field |
|---|---|---|---|---|
| `null` | `af8bde55040a9d2843a8db3416f5facd6bf3e3aa45c592a7615869d7e24df177` | `07aa0a49f63cb2ceb9ceae16cbe44fcf99868ef91858f598aba927fced386473` | `f7b075662d5486382403a3a8a1afa0ff810ecd0b8ca44ba43e1ce0d6be06bb27` | `cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb` |
| `k002500` | `e40d9b8f30efe52bab4f9866f0e8859b35e96a326afd27a16848958900e20043` | `e3e32ae96018910460987f4d97ee1f6c2188143d28929f1b4e1b9dcae0c0c216` | `32b30b835321e5661613f11756dafba4b0639efafdab4720917bda20f120b152` | `c45979acb7a87bdae41fe23d67c9efd10661d5320e5e0c84f9d863a743b3831e` |
| `k005000` | `1a317b4ac7bf8f9b97f8c6cd4b7d0a897b670327c9b1bccdd344a3080d31c2c0` | `1341c26a96e55452a119931e3d4a025e25e6988d50f2572bbef598d8d8f56526` | `56447c871b8f7d33f2bfb2edb0e290a24420763b7e15a8b8c4be24a1721c30fb` | `6c210dd19eefb2b67dad5c5f93ee8008a625b8aea50e685553ee5335f179f000` |
| `k010000` | `b628feccf37b66b5007b4cfd13040b2bd313ea1e16d429b0f1b9891c99e86bc9` | `d8b4568cd7550d99c98d5b6b46ca8a5b1b7304730a00db272a769ec866738b5d` | `a06021ba9ca6c4abb893023b664b56d0e063ecfadda08059d8c39b7a93736970` | `297cee64f3e1438b985f9b242d6405ad5521b5cf320865390bc0ca105fe8351d` |
| `k020000` | `6ebf74adf9f4180c79c2d7431783b0c87cde19b1433a34b503eec961d09e2bd3` | `09c12284374b26c2ef688232f579f8b5489d94c6cfe5fbd1b41e2091420c66c6` | `db07e110b4b8f774b9e25811ab9643f39f5b5ec15f504bb31d11bd65c514df7f` | `7251367a078796a12c2302d726d2d5b1941c9d35d5755745fc664f29de0344fb` |
| `k040000` | `04a065b95ee4260f08081d34b15e35f1ddd449fd79cb262111bcde6cee29ecda` | `c6ddb1d8546343e5946b72e7715c6f1fcd4fbe71126914e914c8edd40eb77452` | `31d99f0beab5d0d665b76cdde66e3e5fb795183b7ac729385af6acb2a1ee4122` | `03ce7bd8a8498ea2a1fc61a0191d0c9eeab3e5ff729e7d522dc07f64add08093` |
| `k060000` | `98b96ee585f16250b14a05c2202c67541f7717e01c63dfbf068f0af7a714ddc0` | `90d85c19df03d35c055aaf68e559910104485de9257c9662b1a764067f4aa424` | `59428f07e6344129d2c5e37ffac84ec19f8e609b2b5951d0d970fb694b88c54a` | `15018481bd8007dd9099d1b67d5e8014283465d062a34ba3f06b3450758b5878` |

Each decoded-field hash equals that arm's retained target-field hash. The terminal checkpoint hashes are preserved inside each row receipt; all seven source launcher status receipts were `ok` before the pack was fired.

## RECALL EVIDENCE

JF2 searched beyond the charter seeds before adjudication:

- Full-text queries over `.omx/research/`, arm receipts, design/SPEC surfaces, and the task ledger for `joint field`, `model refit`, `byte diagonal`, `terminal`, `127292`, `1215`, `1221`, `1239`, `WJ1`, `SY2`, and `trained renderer diagonal`.
- Canonical equations via `.venv/bin/python tools/list_canonical_equations.py --json`, plus `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` FEED blocks.
- The actual JF1 instrument and all seven terminal training/status trees, not just memo prose.

Findings beyond the seeds changed the plan in three ways:

1. `ddm_s1e_stage_a_off_floor_verdict_20260825.md` appears at first to populate cell `#1215`, but it measures the WD3 trained-renderer width family, not JF1's coupled retained field plus terminal HPAC model refit. JF2 therefore did not misclassify S1E as a duplicate physical closure.
2. Canonical equation `hpac_mc36_joint_descent_law_v1` contains run-scoped terminal byte estimates, but its own authority boundary requires the real CPU IHS1 pack and receiver identity. JF2 used it only as recall context and ran the physical coder instead of promoting the estimate.
3. WJ1 already retained a `28,523`-position target list at SHA-256 `bb1c42698e38deb94d9bee8edbdf44261a40a95554defef38d6088730be5da7d`. That converted the byte-win branch from a vague suggestion into the explicit consumer and scorer fire order below.
4. D3B completed concurrently while JF2's long exact decodes ran. Its best receiver-closed lossless Lane factorization is `127,499 B`, `+207 B` against the same subsystem gate. JF2 did not absorb or rerun that work; the terminal `k060000` row is `1,783 B` smaller at `125,716 B`, while both results retain their distinct mechanism and score-authority boundaries.

## Scorer-order and queued exits

Original JF1 scorer order `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/SCORER_FIRE_ORDER.json` is pinned at SHA-256 `70e12f6b793c54b92bbe8ed9d66874dc23480cd27670a0077bb8a559d0d1d4b4`. JF2 did not own the sole full-n600 scorer lane, so it fired no scorer. It is explicitly `HONORED-AND-NARROWED-TO-NULL-PLUS-THREE-BYTE-WINNERS` in:

`.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/wj1_target_consumer/JF2_FIRE_ORDER.json`

SHA-256 `f7607371791441bd6ccc1eeb6d73b5c2bb863f33de75b81e955331b4432af9a3`, also mirrored under the APDataStore receipts root.

- `WJ1_TARGET_CONSUMER_REFIT` — `QUEUED-WITH-A-FIRE-ORDER`; owner `MAIN / JF1 byte-model successor`; consumer store `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/wj1_target_consumer/`; fire when the WJ1 COMPLETE receipt and position-list hash remain pinned and the terminal receipt hash above remains current.
- `JF2_TERMINAL_WINNERS_REALIZED_COMPONENTS_N600` — `QUEUED-WITH-A-FIRE-ORDER`; owner `MAIN / exclusive full-n600 scorer-lane custodian`; consumer store `/Volumes/APDataStore/pact/ddm_jf2_terminal_diagonal_harvest/scorer/`; fire only after MAIN grants the sole idle lane, no competing n600 scorer exists, the APDataStore rows revalidate, and a fresh output-storage preflight passes. Measure the null control plus `k002500`, `k040000`, and `k060000` serially in chunks at most 120, retaining all raw/component outputs.

## Ledger receipts

The canonical CLI, actor `ddm_jf2`, session `ddm_jf2_20260826`, appended these lifecycle events:

- HV2 `ITEM_1` (`JF1 terminal byte harvest`): `in_progress` at `2026-08-26T18:19:21.774041Z`, then `completed`, test `green`, at `2026-08-26T18:19:21.811871Z`.
- HV2 `ITEM_3` (`SY2 terminal JF1 threshold harvest`): `in_progress` at `2026-08-26T18:19:21.850153Z`, then `completed`, test `green`, at `2026-08-26T18:19:21.888571Z`.
- HV2 `ITEM_4` (`WJ1 target-consumer refit`) remains `pending`; it is not falsely resolved because the byte falsifier fired. Its exact execution and component-measurement exits are the two queued actions above.

The shared canonical ledger acquired unrelated OR1/D3B appends while JF2's long decode ran. JF2 used the locked canonical writer for its own four rows but excludes the shared ledger from this landing to avoid absorbing concurrent work.

## Landing custody

The required serializer was invoked with this memo as its sole explicit file, `base=new`, its post-edit SHA-256, `[no-triality] [p0-ledger-ok]`, Markdown-only review override, label `ddm_jf2`, and `--no-co-author`. Git refused before staging with `unable to create temporary file: Operation not permitted` and `failed to insert into database`; the shared staged index remained empty.

The chartered fallback is retained under `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/jf2_commit_fallback/` and mirrored to `/Volumes/APDataStore/pact/ddm_jf2_terminal_diagonal_harvest/receipts/jf2_commit_fallback/`. It contains the final memo intent patch, exact serializer failure/MAIN landing instructions, and a SHA-pinned bundle manifest. MAIN must land that exact patch through the serializer; JF2 makes no commit claim.

## Verdict and boundaries

Verdict scope: `INSTANCE` for each measured row and `FAMILY-ALIVE AT TERMINAL BYTE-LEG` for the seven-arm trained diagonal. The family is not score-admitted: three rows passed the physical model-plus-stream gate, but none has measured realized d_seg or d_pose. The E0002 and SOLVE-entry negatives remain valid at their own scopes; they no longer support closing the terminal trained entry.

`GESTALT-DELTA: terminal training changed the diagonal from an epoch-2 all-negative diagnostic into a measured three-winner byte family; the remaining uncertainty is no longer whether the coupled representation fits under 127,292 B, but whether k002500/k040000/k060000 preserve enough realized Seg/Pose quality to beat the null control and the complete-score frontier.`

`[contest-CUDA T4 n600] own-vehicle frontier: GB1 — S=0.14811799921260607, archive=180,215 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4; JF2 measured no score and did not move the pointer.`
