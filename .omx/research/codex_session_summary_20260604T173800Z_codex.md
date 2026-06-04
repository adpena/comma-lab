# Codex Session Summary - SNeRV/HiNeRV MLX Recovery 2026-06-04T173800Z

## Scope

- Operator directive: recover and continue all local CPU/MLX work; Modal only for exact auth eval of true frontier candidates.
- Priority stack: SNeRV first, HiNeRV second; PR95 remains the baseline to beat.

## Concrete Landings

- Tightened SNeRV official receiver runtime authority: `receiver_runtime_decode_authority` now requires both the official receiver runtime decode contract and selected-packet frame decode success.
- Added/verified regression coverage for the negative case where the contract is true but `frame_decode_succeeded` is false.
- Normalized MLX prefilter device dialect separately from Torch scorer-teacher device dialect:
  - Torch scorer teacher: `gpu -> cuda` or `mps`.
  - MLX prefilter: `mps`/`metal`/`mlx-gpu`/`gpu -> gpu`, `cpu -> cpu`.
- Added HiNeRV checkpoint-export alias coverage mirroring SNeRV.
- Added direct SNeRV native export scorer-device alias coverage so direct callers do not pass raw `gpu` into PyTorch teacher loaders.
- Fixed SNeRV checkpoint exporter retention race: checkpoint meta/state SHA-256s are captured before long receiver proof / MLX prefilter work, and final reports record whether those files still exist at report write.

## Harvested Artifact

- Final recovered checkpoint export:
  `/Volumes/VertigoDataTier/pact/experiments/results/snerv_scalarmean_hardpair_successor_fix2_epoch003199_ema_archive_export_prefilter_20260604T172056Z_codex/snerv_checkpoint_archive_export.json`
- Source live lane:
  `snerv_scalarmean_hardpair_successor_20260604`
- Checkpoint epoch: `3199`
- Packet bytes: `81983`
- Archive bytes: `93620`
- Archive SHA-256: `04aba750b9fd22cb80b62cd30fae5f9c384cd668245ab14402d825b25ca107b7`
- Receiver proof: passed; receiver contract satisfied.
- Local MLX prefilter: written, full 600-pair singleton pass, scorer device `gpu`.
- Local advisory components:
  - canonical score: `90.86453145613247`
  - avg SegNet distance: `0.5048246002693971`
  - avg PoseNet distance: `162.5680926767985`
- Authority state: `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`.
- Blocking verdict: no Modal. The archive is under byte cap and receiver-closed, but local MLX component signal is far from frontier and has scorer-input quality blockers.

## Retention Race Evidence

- First export attempt completed packet/archive/proof/prefilter but failed final JSON because the live checkpoint retention deleted `epoch001299_20260604T165821Z.meta.json` before report hashing.
- Patched exporter survived the same race on epoch `3199`:
  - `checkpoint_meta_present_at_report_write=false`
  - `checkpoint_state_present_at_report_write=false`
  - both SHA-256s preserved in final report.

## Verification

- `uv run pytest src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py -q` -> 54 passed.
- `uv run pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_mlx_prefilter_scorer_device_alias_uses_mlx_device_dialect src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_snerv_native_export_attachment_threads_mlx_prefilter_controls src/tac/tests/test_compact_renderer_mlx_spine_runner.py::test_hinerv_execute_allows_runner_startup_marker_only_dir src/tac/tests/test_export_hinerv_checkpoint_archive.py::test_hinerv_checkpoint_prefilter_device_aliases_use_mlx_dialect src/tac/tests/test_export_snerv_checkpoint_archive.py::test_snerv_checkpoint_export_can_write_receiver_decoded_mlx_prefilter src/tac/tests/test_export_snerv_checkpoint_archive.py::test_snerv_checkpoint_export_prefers_receiver_raw_cache_prefilter -q` -> 6 passed.
- `uv run pytest src/tac/tests/test_snerv_lf_payload_archive_recode.py src/tac/tests/test_snerv_snar_header_grammar_profile.py src/tac/tests/test_nerv_decoder_weight_waterfill.py -q` -> 22 passed.
- `uv run ruff check ...` across touched MLX/export/runner/test surfaces -> passed.
- `uv run python tools/lane_maturity.py validate` -> 1641 lanes clean.

## Next Local Steps

- Keep the live SNeRV hard-pair successor running; harvest later EMA checkpoints only when local MLX signal materially improves or when a new byte/quality gate changes.
- Do not dispatch Modal for the current SNeRV checkpoint; it is byte-closed but component-bad.
- Highest-EV SNeRV work is now scorer-input/cache-quality repair or representation changes that reduce the massive PoseNet/SegNet mismatch, not more exact auth eval.
- Highest-EV HiNeRV work remains byte/codec tightening around the near-cap archive and decoder/archive overhead before any exact auth eval.
