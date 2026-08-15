# DDM MZ1 model-section rate race — receipt

## Verdict

MZ1 did **not** create a smaller archive and did **not** move the frontier. On the exact e480b RX2 object, the claimed 52,566-byte model-serialization gap is an attribution error: `estimated_model_bytes=17,991` describes the HPAC `IHS1` object, while the 70,557-byte `RX1M` wrapper also contains the frozen semantic renderer and carrier. The exact identity is:

`70,557 = 13,619 HPAC + 34,763 semantic + 22,161 carrier + 14 wrapper`.

The 17,996-byte raw HPAC object is only 5 bytes above the trainer estimate and ships at 13,619 bytes, 4,372 bytes **below** that estimate. The existing split-Brotli `q10/q11/q11` model section won the complete lossless race at 70,557 bytes. Exact savings are 0 bytes; the 15,153-byte sub-0.15 rate rung was missed by 15,153 bytes. No scorer or evaluator ran, and no duplicate T4 row is authorized.

Axis: `[macOS-CPU advisory, scorer-free lossless model-section race]`. Verdict scope: `INSTANCE: exact e480b RX2 model section; lossless same-decoded-section forms`.

## Exact full-section race

Every row below is a complete framed model section on the real 70,557-byte object, not an entropy estimate or subset. Every row parsed back to the three exact decoded section hashes. Non-current formats closed through the MZ1 measurement decoder; the winner is closed through the shipping RX1 receiver.

| candidate | exact bytes | delta vs current | exact savings | disposition |
|---|---:|---:|---:|---|
| current RX1M, split Brotli q10/q11/q11 | 70,557 | 0 | 0 | winner; shipping-receiver-closed |
| per-section Brotli q11 | 70,598 | +41 | -41 | FOLDED |
| per-section raw LZMA2 | 70,688 | +131 | -131 | FOLDED |
| per-section adaptive RC64 | 83,753 | +13,196 | -13,196 | FOLDED |
| per-section SMEVR r7 | 72,611 | +2,054 | -2,054 | FOLDED |
| whole-section Brotli q11 | 70,978 | +421 | -421 | FOLDED |
| whole-section raw LZMA2 | 70,759 | +202 | -202 | FOLDED |
| per-tensor mixed identity/Brotli/LZMA/RC64/byte-map | 71,038 | +481 | -481 | FOLDED in the measured MZC1 framing |

Candidate denominator: 8/8 full-section rows. Required lossless saving: 15,153 bytes. Best measured saving: 0 bytes. The score associated with any hypothetical byte saving is projection-only; for the measured winner it remains the already-authoritative RX2 value because the archive bytes are unchanged.

## Section autopsy and derive-don't-ship result

The raw decoded objects are:

| object | raw bytes | SHA-256 | shipped compressed bytes |
|---|---:|---|---:|
| HPAC IHS1 | 17,996 | `94526d667a9c8b98f1e3ef8d39fe8769d6cc6721cb9a102629ad47f26016460d` | 13,619 |
| semantic physical section | 36,040 | `b0d41ec904aca82f93f3c8bc68d0e48896ba08efdaa7a4a2ee204f002fc28ec8` | 34,763 |
| carrier physical section | 22,219 | `065fce08fc3d44e49d29ad624561cbef86d01282cc73dcd32533b5d63115bd9f` | 22,161 |

The IHS1 object consists of 4 magic bytes, 259 bytes of deployed-depth metadata for 517 depths, 91,052 packed weight bits (11,382 bytes), and 6,351 bytes of fixed parameters across 19 tensors. It ships no shape/index metadata. The nine fp32 training-only bit-depth buffers are absent already, so stripping them is a dead end. The 259-byte deployed-depth vector is learned, consumed by decode, and not reconstructible from config or seed; moving it into receiver code would be hide-data-in-code, not derive-don't-ship.

The charter's “XZ container” description is also false for this byte object. The winner is a 14-byte RX1M header plus independently Brotli-compressed HPAC q10, semantic q11, and carrier q11 sections. The 1,099,767-byte file is the training checkpoint, not the raw serialized IHS1 payload.

## Rebuild, receiver identity, and custody

The selected model section was rebuilt with the exact residual and token payload, then packaged twice with the deterministic shipping archive builder. Both retained archives are byte-identical to RX2:

- archive: 183,502 bytes, SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`
- member `p`: 183,402 bytes, SHA-256 `30c0165ec56dd9327ca4dcda477c34c25f7664622ac37ec8ed171114267d1b58`
- model: 70,557 bytes, SHA-256 `7cf390160189e8708faf3a7b09a76fc18cee85e45fdc7f71d30f725014417411`
- token: 112,749 bytes, SHA-256 `b981b8399f184795da7cd99b8ee44416bd672c8c4ed1672f1252b32a64c10627`
- residual: 96 bytes, SHA-256 `64bbf9dfd88d6eb50d111f72d968ab7e8f8dc0ab00fb675d8ed2ee8a410b73ac`

Because the rebuilt archive is the same byte object already exercised by the existing CPU identity receipt, that receipt transfers without inference across different bytes: decoded-token identity is true and the decoded raw output is 3,662,409,600 bytes, SHA-256 `e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9`. Receipt: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/retained/cpu_decode/best_rx2/receipts/CPU_DECODE_RESULT.json`, SHA-256 `a9d8b4ae9a8108052dc867eeda173c3ccfef0994f3320266cc71be8d959c9a5e`.

All materialized coder outputs, including every per-tensor trial, are retained under `/Volumes/VertigoDataTier/pact/ddm_mz1_model_section_rate_race/`. The retention inventory covers 230 files and 2,456,776 bytes before the final receipt. Principal receipts:

| receipt | SHA-256 |
|---|---|
| `PREFLIGHT.json` | `8c8f68885718a2683ce2727d7d0b1208d2c059a44f502c96bc271200e1d5129d` |
| `AUTOPSY_RESULT.json` | `a86e50e3f2919403abf23d70e655b716d3512a1a6fff9f56b848eecf47a32881` |
| `RACE_RESULT.json` | `5d307cdbe57df35cf4a2eb270b52115a9f16671917051c02595ea7efc7ccac73` |
| `BUILD_RESULT.json` | `f11364670a998a0d5c7941867dc0198c2debc025b93ff385edb0d5af54450cd1` |
| `T4_FIRE_ORDER.json` | `c1bde4f4e8ef002e67f94079e8ea8c06a3890adc324856cfcdde413bef1aad2c` |
| `RETENTION_INVENTORY.json` | `e7f0afa4d162d4c2faa7b5d72db6bbaa29438a5351d1e20d0a9ba89857e1861b` |
| `FINAL_RESULT.json` | `d852b8eb1769a4e3b74af6cd0f3d17474a201e36fa6a70d78fca039be6f08e60` |

Reproduction commands, all using the default retained SSD root:

```text
.venv/bin/python experiments/ddm_mz1_model_section_rate_race.py autopsy
.venv/bin/python experiments/ddm_mz1_model_section_rate_race.py race
.venv/bin/python experiments/ddm_mz1_model_section_rate_race.py build
.venv/bin/python experiments/ddm_mz1_model_section_rate_race.py finalize
```

## T4 disposition and boundaries

`T4_FIRE_ORDER.json` is `FOLDED`: MZ1 produced no new archive byte object, so dispatching the same 183,502-byte archive would only duplicate authority call `fc-01M02QMN3SQ9SNHXZMRWXYEJEW`. MAIN may fire only when a strictly smaller archive is under custody, passes the shipping receiver, and proves exact token-stream plus decoded-raw identity before dispatch.

Measured here: exact serialized decomposition, eight complete lossless coder rows, parse-back identity, deterministic archive equality, and transfer of the existing CPU identity proof to the identical bytes. Not measured here: Seg/Pose components, a new exact score, CUDA behavior on a new object, or any lossy/model-changing representation. This is not a FAMILY verdict against representation changes; it closes the tested same-decoded-section coder family on this exact e480b instance.

Own-vehicle frontier remains `S=0.1600920261571558 @ 183,502 B [contest-CUDA T4, n600]`, archive SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`. MZ1 did not move it.

## RECALL EVIDENCE

Before the race, recall searched the full `.omx/research/` corpus, code, task/lane stores, canonical research indices, `sub015_DAG_*` FEED blocks, and the canonical equations registry. Content queries included `model section`, `estimated_model_bytes`, `IHS1`, `self-compress`, `per-tensor`, `lzma raw`, `brotli q11`, `rc64`, and `SMEVR`; equations were enumerated with `.venv/bin/python tools/list_canonical_equations.py --json`.

Beyond the charter seeds, recall found:

- CP135 had already found split q10/q11/q11 best on the frozen semantic/carrier physical sections and SMEVR worse. This changed the plan by treating section-wise Brotli as the real incumbent and requiring the new race to beat complete current framing, not an alleged XZ wrapper.
- `.omx/research/ddm_pr130_reproduce_20260809/SEMANTIC_SECTION_NO_MEMORYLESS_SLACK.md` had closed memoryless coder slack on the PR130 semantic/carrier sections. Their bytes and hashes match RX2's frozen sections. This changed the plan by retaining those controls but concentrating the autopsy on whether HPAC attribution or tensor framing supplied a new opening.
- Canonical equation `hpac_mc36_joint_descent_law_v1` defines `estimated_model_bytes` as an estimated HPAC-model term and requires shipped bytes to be established through pack/identity. This changed the plan by testing the trainer number against raw and compressed IHS1 before accepting the charter's subtraction.
- The HB2 packing receipt showed that training-only `.bit_depth` buffers are not serialized as fp32 tensors; deployed depths are explicitly packed. This changed rung (a) from “strip them” to a byte-level presence test, which confirmed they were already absent.

## LIVE-HYPOTHESES

- A representation-changing attack on the frozen semantic/carrier content remains plausible because those two sections plus wrapper occupy 56,938 bytes, while this receipt closes only lossless recoding of their current physical representation.
- A different HPAC architecture or joint token/model retraining could lower total bytes because fixed decoded tokens were a hard constraint here; this race never tested a different model or token stream.
- Tensor-native shared-context framing may recover a small number of bytes because some individual tensor trials locally beat identity, but the measured MZC1 directory erased those gains. This is plausible as a cleanup lever, not as evidence for the 15,153-byte sub-0.15 rung.

## DEAD-ENDS

- The 52,566-byte “serialization gap” is closed as a false attribution: it subtracts an HPAC-only estimate from a wrapper containing HPAC, semantic, and carrier objects.
- Stripping nine training-only fp32 bit-depth buffers is closed: none of those buffers is shipped.
- Deriving the 259-byte deployed-depth vector from config or seed is closed: it is learned video-derived content required by decode.
- Optimizing an alleged XZ wrapper is closed: the exact winner uses split Brotli q10/q11/q11, not XZ.
- The measured same-decoded-section coder alternatives are closed on this exact instance: all seven alternatives were larger than current by 41 to 13,196 bytes.
- Re-firing T4 on the rebuilt winner is closed: the archive is byte-identical to the already-authoritative archive.
