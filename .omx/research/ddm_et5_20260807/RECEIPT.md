# ddm_et5 carriage-recode receipt

Axis: [macOS-CPU advisory]. Score claim: false. Promotion eligible: false. Pointer moved: false.

## Curve table

| restriction | best coder | kept nnz | dropped collateral | proj bytes | B/full-flip | B/proxy-flip | net dS full-retain | net dS proxy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base_flip_r0 | split_lzma1 | 452061 | 0.1468 | 7100737.5 | 84.476 | 98.751 | 4.656834 | 4.667135 |
| base_flip_r1 | split_lzma1 | 500936 | 0.0546 | 7803675.0 | 92.839 | 98.085 | 5.124891 | 5.128703 |
| base_flip_r2 | split_lzma1 | 519764 | 0.0191 | 8075700.0 | 96.075 | 97.945 | 5.306022 | 5.307382 |
| phase_target_r0 | split_lzma1 | 529870 | 0.0000 | 8229018.8 | 97.899 | 97.899 | 5.408110 | 5.408110 |
| phase_target_r1 | split_lzma1 | 529870 | 0.0000 | 8229018.8 | 97.899 | 97.899 | 5.408110 | 5.408110 |
| phase_target_r2 | split_lzma1 | 529870 | 0.0000 | 8229018.8 | 97.899 | 97.899 | 5.408110 | 5.408110 |

## Verdict

No measured restricted-patch point can be promoted, because every priced point is rate-dead before realization. The best optimistic description-side point is `base_flip_r0` with `split_lzma1`: 84.476 B/full-ET4-flip and projected net dS 4.656834 if all banked ET4 flips survive restriction. Its support-proxy net dS is 4.667135. The honest disposition is `FOLDED`: no measured restriction+coder point goes net-negative.

Waterfilled subset under W (optimistic full-flip retention): 0 / 32 sample pairs, projected bytes 0.0, projected flips 0.0, projected net dS 0.000000.

Follow-on disposition: FOLDED. Do not materialize or spend the scorer slot on this ET5 priced family. Reopen only if a new coder/restriction measures <= W on a stratified n>=32 sample; then the owed leg is all-600 materialization plus exact CPU-torch restricted-patch argmax validation.

## Custody

- ET4 rows: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_solve_within_cvp_rows.jsonl` sha256 `183695482076af8da33aa7fd48f8850696b19f38fc92dc73830ad3952e85cec0`.
- ET4 byteclose receipt: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/byteclose_archive_receipt.json` sha256 `9375636abacc8ea7250a72b75fc5b4fb3fdcd371c2a6de8e503abb2a6598b9f6`.
- ET4 summary: `/Volumes/VertigoDataTier/pact/ddm_et4_20260806/et4_solve_within_cvp_summary.json` sha256 `204c3c200e605a0b54d2a9cb9535a8a3b34ae824c6a729bcc31aa84974c9e46e`.
- Re-encoded all 600 patch records with ET4 Brotli Q11 and matched receipt raw/compressed bytes and shas.
- Per-pair delta index/value shas checked: 600.

## Recall evidence

- Read charter .omx/tmp/codex_runs/et5_prompt.md and common contract .omx/tmp/codex_runs/_common_contract.md.
- Read PROGRAM.md, CLAUDE.md/AGENTS.md byte-identical contract, docs/operating_manual_craft_handoff.md, and .omx/state/main_hot_state.md.
- Read ET4 adjudication and byteclose receipt; ET4 row is rate-dominated but solver leg fixed 78,302 net flips through full byte-close.
- Searched MEMORY.md for et4/et5/carriage/SMEVR/#939/#984/waterfill; only RL1/R7 adjacent coder lessons found, no prior ET5 pricing.
- Searched .omx/research and state for et4/et5/carriage/SMEVR/1.273108/description-vs-realization/#984; found R7 API, SE3/RL1 coder-race precedents, and main_hot_state ET5 route.
- Ran tools/list_canonical_equations.py --json sampling; consumed registered byte/flip waterline discipline rather than minting a new equation.

## Boundaries

- No SegNet/PoseNet scorer call was run by ET5.
- No archive.zip was built.
- All net-dS values are projections using banked ET4 full-patch flip counts and measured restricted bytes.
- Verdict scope: INSTANCE, et4 correction field on tq1c parent, stratified n32 description pricing.

Own-vehicle frontier line: S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]; ET5 did not move it.
