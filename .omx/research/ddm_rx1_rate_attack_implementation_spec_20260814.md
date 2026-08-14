# DDM RX1 implementation spec — tq1c probability-object transfer onto MC36

## Objective

Add one scorer-free, stage-checkpointed runner that measures whether the already-custodied tq1c IntegerHPAC probability object can losslessly encode MC36's exact n600 token field more cheaply than MC36. The runner must retain every model representation, probability-code frame, entropy payload, decoded-token payload, candidate archive, deterministic repeat, and machine-readable receipt under `/Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814/`.

This is a rate-only experiment. Admission requires byte-identical decoded event-order symbols and byte-identical decoded spatial tokens. It does not run a scorer and must never claim an exact contest score.

## Owned source surface

- Add `experiments/ddm_rx1_rate_representation_attack.py`.
- Add narrow tests in `experiments/tests/test_ddm_rx1_rate_representation_attack.py` if that test directory is the repository convention; otherwise use the nearest existing experiment-test convention.
- Do not modify any other repository file.

## Immutable inputs and pins

- MC36 archive: `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/micro35_candidate/archive.zip`, 186269 bytes, SHA-256 `f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de`.
- MC36 adapted runtime: `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/micro35_candidate/adapted_runtime`.
- MC36 event-order source manifest: `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532/compile_workspace/retained/candidates/qs1_combined_unique_pairs/primary/chunk_manifest.json`.
- Expected decoded event-order SHA-256: `f4149ab66096e9de8771d5cf9be1058c543177acc0041fed6c361b73e0820be8`.
- Expected decoded spatial-token SHA-256: `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52`.
- tq1c packed IHS1: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/artifacts/tq1c/hpac.bin.xz`, 14116 bytes, SHA-256 `6c44216e8f79bd7d04e998b898d5bf0dc16bae6e3763f8bc19ce4ec8ebdabb40`.
- tq1c training checkpoint for provenance only: `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/tq1c/hpac_selfcompress_e60.pt`, 177614 bytes, SHA-256 `2a907f06cc5d278e1df12eac6cd575fb3dcb32477446f0da842bb92a14d05ddc`.
- ExperimentBook RC64 source: `/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book`.
- Reuse the exact algorithms in `experiments/ddm_cp135_rate_compose.py` for frame-wise probability export and disk-resumable RC64 state snapshots; do not invent a proxy coder.

## Required stages

Expose explicit CLI stages and make every stage idempotently resumable from durable receipts:

1. `preflight`: validate every pin, validate the source manifest and both decoded-token digests, confirm the output resides on the SSD, record free-space and a conservative byte budget, and fail closed if insufficient.
2. `prepare`: persist the tq1c packed payload and exact decompressed IHS1 bytes; race only lossless model representations (at minimum custodied XZ and Brotli qualities 0..11), retaining every materialized stream and parse-back receipt. Parse MC36 once and retain its semantic/carrier sections and residual table. Build an adapted runtime copy that recognizes an explicit RX1 model container whose HPAC section restores canonical IHS1; learned/video-derived bytes stay inside `archive.zip`.
3. `export`: export exact int16 probability-code checkpoints one frame at a time for two variants: `tq1c_table_on` (MC36's exact residual correction table applied) and `tq1c_table_off` (no correction). Existing complete frame checkpoints must be custody-verified and reused. Do not keep probability codes only in memory.
4. `encode`: encode one requested variant with the native RC64 backend, snapshotting exact encoder state at least every 24 frames, resuming only when the snapshot is byte-bound to the probability identity receipt. Persist the final RC64 payload before measuring it. Decode it fully and persist both event-order and spatial token fields; reject unless both expected hashes match.
5. `build`: build and retain every viable whole-container archive over the lossless model-representation race for a requested variant, plus byte-identical deterministic repeats. For `table_off`, ship a receiver-readable neutral residual table (or an explicit table-absent marker) so shipped parsing uses the same probabilities that were encoded. Every candidate must parse back through its copied shipped receiver, recover the exact tq1c IHS1, and prove token identity. Record whole-archive bytes, SHA-256, delta vs MC36, rate-only projected score using MC36's exact distortion terms, and `score_claim=false`.
6. `finalize`: produce one JSON result with per-lever measured bytes, winner, retention inventory, receiver parse-back proof, zero decoded-token/raw-token distortion, and a sealed MAIN T4 fire order. Do not launch an evaluator.

## Receiver and candidate rules

- The RX1 container must be self-delimiting and reject malformed/trailing model fields.
- The adapted runtime is copied under RX1 output; do not edit the source MC36 runtime.
- Receiver code is free under the contest rules, but all learned tq1c bytes remain in the stored member `p`.
- Preserve MC36 semantic and carrier bytes exactly after their existing decode representation; only the entropy probability object and residual-table choice may change.
- Use deterministic ZIP metadata and retain `archive.zip`, `archive.repeat.zip`, member `p`, model payload, residual payload, token payload, and receipts for every candidate.
- Never materialize a payload whose bytes are not persisted.

## Resource and authority constraints

- CPU-only, scorer-free, no Modal, no GPU, no `upstream/evaluate.py`.
- The full n600 export is long; it must be frame-checkpointed. RC64 must be state-checkpointed. No non-resumable long phase.
- Use at most four Torch CPU threads and one interop thread.
- Do not render the 3.66 GB RGB raw in this runner. Exact decoded token identity is the zero-distortion admission gate; the separate charter owner may run the existing lifted CPU renderer afterward for the final raw-output receipt.
- Axis label: `[macOS-CPU advisory, scorer-free lossless composition]`.

## Tests and acceptance

Tests must cover, with toy fixtures and no custodied input dependency:

- deterministic ZIP equality and stored-member parse-back;
- RX1 container encode/decode and malformed/trailing rejection;
- neutral-table encoding/decoding if used;
- probability conversion and spatial/event-order permutation identity;
- RC64 state header round-trip or a narrow mocked checkpoint custody test;
- no scalar-only payload retention violation in the owned runner.

Acceptance commands:

```bash
python -m pytest -q experiments/tests/test_ddm_rx1_rate_representation_attack.py
python experiments/ddm_rx1_rate_representation_attack.py --help
python -m tac.preflight.check_no_measure_and_discard_payload --paths experiments/ddm_rx1_rate_representation_attack.py
```

## Do not touch

- `.omx/research/ddm_cr1_composition_row_827_20260801.md`
- `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md`
- `src/tac/optimization/direct_description_carrier_compose.py`
- Any staged index entry or unrelated dirty worktree path.
- `upstream/` and every immutable/custodied input above.

## Execution addendum

After both whole-container arms proved larger than MC36, the same runner gained a
`cpu-decode` stage so the required final raw-output identity receipt would not be
left to an unowned handoff. It reuses the already-custodied F26P four-thread CPU
lift, copies and pins that runtime inside RX1 custody, preserves its durable token
checkpoint and interrupted-render path, and retains the complete raw output. It
does not invoke a scorer. `finalize` now fails closed until that CPU receipt is
present and byte-identical to MC36's retained CPU raw SHA-256.
