# ddm_hp1 receipt - learned-AR-prior race on tq1c IX2 token stream

Axis: `[macOS-CPU byte-only scorer-free]`. `score_claim=false`. No scorer forwards, no
`upstream/evaluate.py`, no receiver edit, no archive candidate promoted.

## Byte Ladder

| row | counted object | bytes | vs shipped | dS_rate vs shipped |
|---:|---|---:|---:|---:|
| 1 | shipped `IX2TOK01` bulk, Brotli-q11 blocks | 341,296 | 0 | 0.000000000000 |
| 2 | forced `IX2TOK01` LZMA1 blocks | 349,811 | +8,515 | +0.005669788986 |
| 3 | HP1 learned prior + counted model | 456,166 | +114,870 | +0.076487217945 |

Verdict: the pre-registered net-byte falsifier fires. The best HP1 learned-prior
frame is byte-negative on the exact tq1c token stream.

## What Was Measured

- Source archive: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes`
- Source archive custody: 357,837 B, sha256 `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`.
- ZIP shape: one stored member `0.bin`, 357,729 B, sha256 `1bd7b7186978e7d96316a43e297588208411b91df91e4834be6357128abc839b`.
- Token stream: receiver-input `IX2TOK01` lattice, not 5-class semantic labels; shape `(600, 24, 32, 4)`, `uint8`, alphabet `{0..15}`, 1,843,200 symbols, sha256 `1a46a51909b150bc1fc320cb6f66f52cc53472e6f830c911c2ea7bbec2bbdcc3`.
- Shipped token bulk: 341,296 B, sha256 `a2d3ad05074ea23a77d8111553da63b738d89705e61ec0cc76d31be2d8467183`; it re-encoded byte-identically through `tac.optimization.ddm_ix2_archive_container.encode_token_frame`.
- HP1 frame artifact: `/Volumes/VertigoDataTier/pact/ddm_hp1_20260806/hp1_learned_prior.hp1`, 456,166 B, sha256 `575d7acafd562e75ead009bcb81634a0681859f93e5e1510aebc537c0fc641fc`.
- Machine-readable receipt: `.omx/research/ddm_hp1_20260806/hp1_results.json`, sha256 `c4760f8cea0f99753f622736520ae3134e24c0730783d3a8fe9d0afad39f838e`.

## Entropy Checks

| conditioning | active contexts | bits/symbol | ideal bytes |
|---|---:|---:|---:|
| order-0 | 16 | 3.457480934629 | 796,603.607 |
| spatial: left/up/channel-prev | 12,690 | 2.455426600903 | 565,730.289 |
| previous-pair same cell | 68 | 2.003773349763 | 461,669.380 |
| previous-pair plus spatial | 126,222 | 1.254670827724 | 289,076.159 |

Temporal enters as conditioning, not diff-coding. The high-cardinality
prev+spatial empirical table shows headroom in principle, but the counted <=10K
model family measured here cannot realize it economically.

## Learned-Prior Race

All HP1 rows use exact range coding, counted model bytes, exact decode equality,
and canonical re-encode equality. Generic context maps were raced; no scorer data
or hidden per-position table was used.

| context mode | model bytes | range stream B | frame B | result |
|---|---:|---:|---:|---|
| prev_k | 1,088 | 466,159 | 467,309 | loses |
| prev_left | 4,624 | 457,399 | 462,085 | loses |
| prev_up | 4,624 | 451,511 | 456,197 | loses |
| prev_chan | 4,624 | 451,480 | 456,166 | best, loses |
| hash_prev_spatial | 10,000 | 692,418 | 702,480 | loses |

Best decode feasibility: 3.497536833 s on this host including canonical
re-encode; 1,843,200 scalar range updates; 48 patch groups per pair; 3,072
cell-channel temporal streams. This is inside the 30-minute envelope, but it is
not a PR130 94-step decode claim.

## #918 Scope Re-Grade

Scope correction filed here: `#918 "RATE CODING IS CLOSED"` is instrument-scoped.
It raced explicit token LZ/rank/basis/base-rule families; it did not measure the
learned-conditional-prior family on the live tq1c stream. The LZ/rank/basis
closure stands. HP1 adds the measured result for a small counted learned
conditional-prior formulation on this exact stream: negative, because the best
net row is +114,870 B after counted model bytes. Verdict scope: FAMILY for the
tested <=10K static-context learned-prior family on tq1c `IX2TOK01`; not a
global theorem against all future neural priors.

## Recall Evidence

| source searched | query / read | what changed |
|---|---|---|
| Governing files | `hp1_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, operating manual, `main_hot_state.md` | Kept HP1 scorer-free; used tq1c `b35e7568` as the live parent; preserved scorer slot and protected files. |
| Memory registry | `rg -n "pr86|PR130|eu2|#918|sv2|IX2TOK01|adaptive-map|tq1c|pa2|learned conditional prior" MEMORY.md` | No direct HP1 memory hit; reused only scoped real-coder/context reminders and kept absence bounded. |
| tq1c receipt | `.omx/research/ddm_tq1_20260805/tq1c/RECEIPT.md` | Located the exact final archive and confirmed saturation scope. |
| SV2 receipt | `.omx/research/ddm_sv2_smevr_base_rule_race_20260803.md` | Identified `IX2TOK01` as the live token field and match-structure as the current shipped win. |
| EU2 receipt | `.omx/research/ddm_eu2_20260805/EU2_RECEIPT.md` | Set the <=10K int8 admission scale and the PA2-projected context floor. |
| PA2 receipt | `.omx/research/ddm_pa2_20260805/PA2_RECEIPT.md` | Treated shared context as modest evidence, not as a token-stream adoption result. |
| BO1 / NA3 | `#918` sections in `ddm_bo1...` and `ddm_na3_negative_audit_20260804.md` | Re-graded #918 as explicit-coder scoped, not a blanket learned-prior closure. |
| XO1 receipt | `.omx/research/ddm_xo1_20260805/XO1_RECEIPT.md` | Confirmed current IX2 same-coder/orderer controls were already negative and public-data student remained data-blocked. |
| Canonical equations / DAG/index | `tools/list_canonical_equations.py --json` filtered for token/context/entropy/coder/rate and `rg` over `CANONICAL_RESEARCH_INDEX*`/`sub015_DAG_*` | Found adjacent entropy/rule-118/rate context, but no measured current-vehicle learned-prior row overriding HP1. |

## Boundaries

Measured: byte-only coder rows, exact token decode equality, model bytes counted,
and decode-time feasibility for the HP1 Python prototype.

Not measured: `d_seg`, `d_pose`, receiver integration, full archive recomposition,
contest-CPU, contest-CUDA, or any exact score movement.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`;
contest pointer remains borrowed/unmoved at `S = 0.1910828242`.
