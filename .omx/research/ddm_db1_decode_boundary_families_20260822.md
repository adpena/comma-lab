# DDM DB1 decode-boundary families — 2026-08-22

Disposition: **CLOSED(FORMULATION) for the two measured Family-A metadata-boundary variants; Family B is QUEUED-WITH-A-FIRE-ORDER under NR1; Family C remains folded into B.**

The receiver cannot make dc1s's position or per-block-width information free merely by moving it across a packet boundary. On the complete retained n600 FX5 object, fixed-grid addressing removes the separate 227.4 KB Elias-Fano position field but carries the same video-derived support in a dense width field; its best real-coded complete payload is **372,049 B**. One counted width per group removes all per-block width bits, but the wider hash questions alone cost 2,112,915 bits; its best complete payload is **477,133 B**. Both are larger than the **113,777 B** token member they would replace and far above the **71,395 B** same-distortion replacement ceiling implied by the 42,382 B archive demand.

This is a scorer-free retained-payload result on `[macOS-CPU advisory / scorer-free retained-fx5 n600 rate measurement]`. No receiver was edited, no archive candidate was built, no scorer or Modal job ran, and no score or frontier movement is claimed.

## Measured result

All rows cover all 600 frames, all 190 HPAC groups, and all 221,374 retained non-MAP blocks. “Distance to demand” is `113,777 - payload_bytes - 42,382`; a negative value is the remaining byte shortfall. Complete coded payloads include DB1's 46-byte coder wrapper when a non-raw coder wins.

| formulation | boundary semantics | raw packet B | winning real coder | complete payload B | savings vs 113,777 B | distance to 42,382 B demand | result |
|---|---|---:|---|---:|---:|---:|---|
| Family-A source control | explicit sparse positions and source-chosen individual/uniform widths | 388,326 | Brotli q11 | **341,855** | -228,078 | **-270,460** | fails |
| dense-width addressing | fixed-grid slot order; zero width means MAP, nonzero width supplies support and question width | 21,256,883 | raw LZMA2 extreme | **372,049** | -258,272 | **-300,654** | fails |
| group-uniform width | explicit sparse positions; one counted five-bit width per group; zero per-block width bits | 492,680 | Brotli q11 | **477,133** | -363,356 | **-405,738** | fails |

The 71,395 B ceiling is a derived one-member replacement bar, not an allocation or a predicted stream size. None of the rows reaches it. Even the real-coded Family-A control is 270,460 B above it.

The complete coder races were:

| formulation | raw | Brotli q1 | Brotli q6 | Brotli q9 | Brotli q11 | zlib9 | raw LZMA2 extreme |
|---|---:|---:|---:|---:|---:|---:|---:|
| Family-A source control | 388,326 | 378,064 | 362,826 | 362,888 | **341,855** | 370,204 | 359,425 |
| dense-width addressing | 21,256,883 | 505,299 | 488,532 | 458,784 | 396,715 | 438,743 | **372,049** |
| group-uniform width | 492,680 | 492,731 | 492,731 | 492,731 | **477,133** | 484,804 | 479,239 |

Each winning coder was independently rerun, retained to a distinct path, and reproduced the winner byte-for-byte:

- Family A: SHA-256 `48fe2c489c2983504eca58dca02d065c2b019bec172441b4e10f493b076e09e2`.
- Dense-width: SHA-256 `6c93bccafe728e5f18c3b9d4b95ebc97c44ef2b6f28e59993a7f985df4b9d71b`.
- Group-uniform: SHA-256 `ad9f3d8982cd253da38f6729c219775d041611b30c0096b6689af37d636d448d`.

## What the two derivability tests establish

### Position derivation

The dense variant gives the receiver a deterministic traversal of every fixed-grid slot and therefore consumes no separate sparse-position list. It does **not** derive video-specific support for free: zero/nonzero values in the counted 21,200,400-byte dense width field still identify the 221,374 non-MAP slots. The question field remains 450,759 bits / 56,345 B. A real outer-coder race reduces the highly sparse raw packet to 372,049 B, but that remains 258,272 B larger than the token member.

Conclusion: **CLOSED(FORMULATION: fixed-grid dense-width addressing).** This is support relocation, not a rule-118-free derivation. A shared task-cell certificate that genuinely determines support is a different Family-B representation and remains open.

### Width derivation

The group-uniform variant consumes one counted five-bit maximum width per group and derives every block's width from that shared value. It therefore contains zero per-block width bits. The price moves into longer questions: 2,112,915 question bits = 264,114.375 B, already 150,337.375 B larger than the entire 113,777 B token member before positions, headers, or framing. Sparse positions still cost 1,819,325 bits.

Conclusion: **CLOSED(FORMULATION: one uniform question width per group).** Combining this width rule with a different support representation cannot beat the token member because its question bits alone exceed the member.

### Decode proof

The decoder parsed each actual coder winner, regenerated every question answer from the retained receiver-known HPAC conditional rows and transmitted prefix, and never passed the target into the question decoder. Each variant decoded 221,374 non-MAP blocks and traversed an expected 618,965 ordered candidates. The three 3,151,202-byte answer transcripts are byte-identical, SHA-256 `f82dbd47144e012a83732ff8d2dd233e8263eb17e38c04c9da216f038cca68aa`. Local decode times were 10.76 s, 10.83 s, and 18.46 s respectively; these timings are advisory and are not score terms.

The existing FX5 adaptive walk was not rerun. Its conditional rows, group sizes, and targets were consumed from all 190 retained dc1s bundles, each content-checked against the pinned full-n600 result. This is the pre-existing receiver context for the boundary test, not a claim that DB1 built a new shipped receiver. A retained one-bit mutation of the dense-width field was refused on its SHA-256 check.

## Prior-law verdict

The charter predicted that moving the metadata boundary should change the 388,326 B arithmetic by more than dc1s's 1,434 B within-family block-size gain. It did: the largest absolute complete-payload change versus 388,326 B was 88,807 B. Therefore the registered falsifier, “every derivable-metadata variant remains within about 1,434 B,” **did not fire**.

The direction was nevertheless adverse. Boundary placement materially changes the rate, but these two honest receiver paths do not turn that change into a saving. This closes only the named formulations; it does not promote a broad nonexistence claim for every possible support code or for a true scorer-cell quotient.

## Families B and C, from DC1 rather than relabeling

DC1 Family B ships a counted scorer-equivalence-cell constraint plus every solve seed/index, learned or video-derived parameter, and framing, then deterministically returns any candidate proved to lie in that cell. Its ideal sample length is `L_C = -log2 P(C)`, compared with exact-answer length `L_x = -log2 P(x)`; the possible saving is `L_x - L_C = -log2 P(x|C)`. The missing object is a compact receiver-checkable cell constraint. The explicit raw constraint measured by DC1 is about 44.244 MB, and no current proof lets a scorer-free receiver certify every output in the desired Seg/Pose cell.

DC1 Family C is relative-entropy / bits-back coding with a receiver-runnable posterior `Q`. When `Q=P(.|C)`, its ideal price is `KL(Q||P)=-log2 P(C)`, algebraically the same quotient price as Family B. A delta posterior on the current exact field refunds zero bits. Family C therefore has no independent live branch until an actual runnable quotient exists.

DB1 did not rename either dense-width or group-uniform metadata as B/C. The older score-quotient functional is a structural contract with no fitted receiver-closed result. NR1 is explicitly specification-only, has no executable command, and forbids that fixture as a shortcut. Thus **Family B remains unmeasured and unimplemented**, and **Family C remains folded into it**.

## RECALL EVIDENCE

The original recall was broader than the charter seeds:

- Research memos and receipts: content searches under `.omx/research/` for `Family B`, `Family C`, `task-cell`, `task cell`, `score quotient`, `quotient functional`, `hash-constrained`, `hash question`, `sparse-grid`, `bits-back`, `REC`, `relative entropy`, `receiver-close`, `113777`, and `42,382`.
- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for `quotient`, `rate`, `MDL`, `task RD`, `receiver`, and `sufficient statistic`.
- Graph memory: content searches over `.omx/research/CANONICAL_RESEARCH_INDEX*` and `.omx/research/sub015_DAG_*` for `FEED-603`, `DC1`, `NR1`, `score quotient`, `task_rd`, and `worldsheet`.
- Live/task stores: searches for `dc1`, `db1`, `nr1`, `quotient`, `113777`, and `42382` in `.omx/state/main_hot_state.md`, `.omx/state/canonical_task_status.jsonl`, and related task-ledger surfaces.
- Source and custody: `experiments/ddm_dc1s_sparse_grid_sweep.py`, its full-n600 result, source packet, and all 190 group bundles; the RB1 and TL1 provenance pins; the actual decoder/parser definitions.

Beyond the charter seeds, recall found the July score-quotient functional contract and FEED-603, but both are design/structure surfaces without an executable or empirical quotient. It also found NR1's newer exact boundary: a real quotient must be born against the frozen endpoint, retain `QPARAM + QCTX + QPAIR + QEVENT`, and fail at 113,777 B actual-coded quotient bytes or the dynamic whole-archive ceiling. TL1 and live state supersede the common contract's older frontier line with the exact DX2 row and retain the 42,382 B fixed-distortion demand.

That changed the plan in three ways: DB1 refused to pass metadata variants off as B/C; it priced Family A and both boundary tests through complete real coder wrappers; and it folded the only honest B follow-on into NR1's existing queue rather than creating a duplicate fire row. No cheaper operational task-cell quotient payload was found in the searched corpus.

## Custody, retention, and verification

Implementation: `experiments/ddm_db1_decode_boundary_families.py`, SHA-256 `94ad08825a859fd1d037f3e56f4ba01faaf92c78588f2d8142dc8d4179cd6cba`.

Current retained store: `/Volumes/APDataStore/pact/ddm_db1_decode_boundary_families/retained/`, 111,471,910 logical bytes across 2,400 files at the final audit. The manifest content-addresses every current stage artifact; prior stage receipts were copied into `prior_attempts/` before replacement.

Primary receipts:

- `result.json`: 73,314 B, SHA-256 `48cb8adae36ce09a96777ef1296cb624b1876fa8198838449e906305124a5d96`.
- `stage_1_inventory.json`: 41,978 B, SHA-256 `783748efda472d5dd805ac311c8c5e6bc261d43041f9ccf5b077971fd625a754`.
- `stage_2_coder_races.json`: 12,490 B, SHA-256 `9275b4ce8a8a20edfd1ceb059b9afe9580feaa23cbe47f81717d4bff4c1043b8`.
- `stage_3_decode_verification.json`: 184,489 B, SHA-256 `561f72a5b3a882c9648137de77eb6b8bfefe61a78d4ca4652bfee96bddc780a2`.
- `manifest.json`: 192,415 B, SHA-256 `95437c8787460960eba6a53daa0a755c843138ba3343a60c54e94dbe630d7347`.

The final preflight passed with more than the 1 GiB estimate plus 8 GiB reserve and pinned 195 sources: the source script, source packet, source result, RB1, TL1, and all 190 group bundles. The final run is stage-resumable. Every raw packet, every coder candidate, every genuinely re-encoded winner repeat, all per-group decode outputs, final transcripts, the mutation control, and prior receipts remain retained.

### P0 retention incident and repair

An early development decoder used Python 3.12+'s compensated `sum()` rather than dc1s's explicit sequential float64 accumulation. At a 1e-15-class product-law tie it returned the wrong answer at group 129, block 94. That attempt had accumulated the prior transcript only in memory and discarded it on failure. This was an **ALWAYS KEEP THE PAYLOAD P0 violation**, not a harmless debug detail.

Before banking any result, DB1 repaired the decoder to match dc1s's exact accumulation, added per-group write-through checkpoints with fsync and success/failure attempt receipts, and reconstructed the exact discarded prefix from the proven successful traversal: complete groups 0–128 plus the five records preceding block 94 in group 129. The recovered 2,404,863-byte payload is retained at `recovered_incidents/family_a_pre_group129_block94_failure_prefix.answers.bin`, SHA-256 `4215467cbc09dcf75fa2ca5419ac2e5c195af2a726dde4c4986157a164a891df`. The first five group-129 indices are below 94 and the sixth is 94; independent concatenation reproduced the retained recovery byte-for-byte.

The final implementation's targeted payload-retention audit examined one candidate Python file and found zero measure-and-discard findings. `py_compile` and Ruff passed. Independent verification rehashed all 195 source pins, decoded every coder winner to its raw packet, recomputed the winner and demand arithmetic, concatenated and rehashed all 570 current group checkpoints, reproduced the incident recovery, and reconfirmed mutation refusal. Two review-tracker passes were completed after the final code content.

## Boundary-scoped disposition

- **FOLDED / CLOSED(FORMULATION):** fixed-grid dense-width addressing. Consumer: this memo and retained DB1 result. Reason: support remains counted and the best complete payload is 372,049 B.
- **FOLDED / CLOSED(FORMULATION):** one uniform question width per group. Consumer: this memo and retained DB1 result. Reason: question bits alone exceed the token member; the best complete payload is 477,133 B.
- **FOLDED:** Family C into Family B. Reason: without a receiver-runnable non-delta `Q`, it has no distinct price or executable branch.
- **QUEUED-WITH-A-FIRE-ORDER:** Family B under NR1, not DB1. It remains a genuinely different representation whose counted task-cell constraint determines outputs rather than relocating exact-token metadata.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / NR1; consumer store: `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/retained/`; fire trigger: r9 is terminal, the frozen endpoint and provenance are harvested, and a reviewed implementation provides a real receiver-checkable counted quotient with an executable command; action: build and actual-code `QPARAM + QCTX + QPAIR + QEVENT`, then test the 113,777 B quotient falsifier and the candidate's recomputed whole-archive ceiling. Family C may be priced only if that implementation also supplies a receiver-runnable non-delta posterior `Q`.

## LIVE-HYPOTHESES

- A compact implicit task-cell body can beat exact 113,777 B token reproduction because scorer-cell equivalence is a weaker requirement than exact token identity; the 42,382 B demand fits inside the token member's mass, but this remains unmeasured until a receiver-closed quotient exists.
- A real quotient may make sparse surprise support implicit in shared counted structure rather than storing or relocating one support decision per fixed-grid slot; this is plausible because dc1s measured position metadata as the largest raw debt, while DB1 proved that fixed-grid relocation alone is insufficient.
- REC/bits-back may become useful only after the Family-B quotient exists and exposes a receiver-runnable non-delta posterior; then its finite model/state/index costs can be measured instead of assumed.

## DEAD-ENDS

- Family A with explicit sparse positions, source width modes, and SHA-prefix questions: closed at the existing full-n600 scope; even a new outer-coder race leaves 341,855 B.
- Fixed-grid dense-width addressing: closed at formulation scope because video-specific support is still counted and the best complete payload is 372,049 B.
- One uniform width per group: closed at formulation scope because widened questions alone cost 264,114.375 B and the complete payload is 477,133 B.
- Treating the July score-quotient contract or NR1 specification as an executable Family-B shortcut: closed by missing fitted receiver-closed task output and NR1's explicit no-shortcut boundary.
- Current-field bits-back with a deterministic delta posterior: closed because it refunds zero bits and reduces to ordinary exact-answer coding.

Own-vehicle frontier remains **S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**, DX2 archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; DB1 did not move it.
