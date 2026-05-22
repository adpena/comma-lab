# Codex Findings: Decoder-Q Selective Runtime Packet And Materialization

- UTC: 2026-05-22T14:35:30Z
- Lane: `lane_codex_decoderq_selective_runtime_packet_20260522`
- Status: `runtime_packet_l0_materialized_top32_locality_passed`
- Score authority: `false`
- Ready for exact eval: `false`

## What Landed

Implemented a reusable L0 planner, append-tail materializer, and raw-output
locality harness for decoder-q selective runtime probes:

- `src/tac/optimization/decoder_q_selective_runtime_packet.py`
- `src/tac/optimization/decoder_q_selective_runtime_materializer.py`
- `tools/plan_decoder_q_selective_runtime_packet.py`
- `tools/materialize_decoder_q_selective_runtime_candidate.py`
- `tools/run_decoder_q_selective_runtime_locality_controls.py`
- `src/tac/tests/test_decoder_q_selective_runtime_packet.py`
- `src/tac/tests/test_decoder_q_selective_runtime_materializer.py`
- `src/tac/tests/test_decoder_q_selective_runtime_controls.py`

The planner validates the selective-window bridge plan against the FEC6 parent
archive and the full-video materialized decoder-q mutation. It recomputes the
q-domain mutation from the parent decoder bytes and verifies the mutated decoder
SHA before emitting the compact archive-local `DQS1` payload contract.

The canonical materialized grammar is a tail extension inside archive member
`x`: legacy `FP11` + FEC6 selector bytes, followed by a charged `DQS1` trailer.
The materializer now fails closed unless a loaded packet plan's claimed base ZIP
SHA, ZIP size, member name, member SHA, member bytes, CRC, compression method,
and decoder SHA match the actual `--base-archive`.

## Empirical Packet Plans

Generated non-promotional packet-plan artifacts:

- Singleton:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_runtime_packet_plan_singleton_pair501.json`
- Singleton markdown:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_runtime_packet_plan_singleton_pair501.md`
- Top32:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_runtime_packet_plan_top32.json`
- Top32 markdown:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_runtime_packet_plan_top32.md`

Singleton pair `501` maps to affected frames `[1002, 1003]` under
`pair_all_frames`, with a charged `DQS1` payload of 13 bytes and estimated
rate-score delta `0.000008656166390588228`. Top32 maps to 64 affected frames,
with a charged `DQS1` payload of 75 bytes and estimated rate-score delta
`0.00004993942148416285`.

## Materialized Artifacts

- Singleton packet:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_pair501/submission_dir/archive.zip`
  SHA-256 `788a7459fbcc70e6002e556e35dfcfae1b172ac01bde62fe68396994ff595803`,
  ZIP bytes `178530`, member `x` bytes `178430`, DQS1 payload bytes `13`.
- Top32 packet:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_top32/submission_dir/archive.zip`
  SHA-256 `3c4e15bfe7ae1004ad23e89a52c2836e609c1f99e25b58f45c01747226705d59`,
  ZIP bytes `178592`, member `x` bytes `178492`, DQS1 payload bytes `75`.

Both artifacts keep all selective bytes inside archive member `x`; no external
sidecar is required at inflate time. Both remain `score_claim=false`.

## Locality Controls

The locality harness now cross-checks CLI expectations against both
`selective_runtime_manifest.json` and the embedded archive-member DQS1 tail
before running the inflate comparison.

- Singleton control:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_pair501/locality_controls_pair501.json`
  passed. Selected frames `1002,1003` match the global decoder-q mutation;
  the other 1198 frames match FEC6 parent; all mismatch counters are zero.
- Top32 control:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_top32/locality_controls_top32.json`
  passed. 64 selected frames match the global decoder-q mutation; the other
  1136 frames match FEC6 parent; all mismatch counters are zero.

Top32 selected-frame hash:
`cff4cc7400011f2d1cbe0e4a19c7aeb66af1b409d0cdad9868b86edcb23e2cea`.
Top32 unselected-frame hash:
`d99cf5d5f735249a66bea6a8ed02bd45d44abbf9e79e20cfa801756cb95d0654`.
Top32 selective raw SHA-256:
`dee3ee3cf6c308f8dc2f11b3e611cc27ef75b3d452163bb1274e94603a268a00`.

## Advisory Scorer Signal

The earlier stale-path `[macOS-CPU advisory]` run
`experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/dqs1_top32_cpu_advisory_venv.json`
has the same top32 archive SHA and raw-output SHA as the canonical top32
locality output. Its canonical score is `0.1920602563025898` with
`score_claim=false`, `promotion_eligible=false`, and `score_axis=cpu_advisory`.

Against the matching local FEC6 advisory baseline
`0.19206131688110561 [macOS-CPU advisory]`, this is `-0.0000010605785158157577`.
Against the Linux x86_64 `[contest-CPU]` frontier, this is not apples-to-apples;
the naive cross-axis arithmetic is `+0.000008939402589808187`. Do not dispatch
exact eval from this result as a promotion candidate; use it as sign-calibration
data for the decoder-q response surface.

## Critical Design Guard

The full-video materialized decoder-q candidate differs from the FEC6 parent in
112,482 decoder bytes because one q-symbol mutation is recompressed through
split Brotli. A selective runtime must therefore apply the q-domain patch before
decoder state construction; it must not store or replay a recompressed byte-diff
of the decoder blob.

The repaired inflate adapter decodes the base local batch, decodes the full same
local batch through the mutated decoder when any selected pair is present, then
splices only the selected rows before the existing FEC6 selector transform. The
previous mutated-subset assignment shape was the root cause of selected-frame
parity failure and is now covered by tests plus real locality controls.

## Verification

- `ruff check src/tac/optimization/decoder_q_selective_runtime_packet.py src/tac/optimization/decoder_q_selective_runtime_materializer.py tools/materialize_decoder_q_selective_runtime_candidate.py tools/plan_decoder_q_selective_runtime_packet.py tools/run_decoder_q_selective_runtime_locality_controls.py src/tac/tests/test_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_runtime_materializer.py src/tac/tests/test_decoder_q_selective_runtime_controls.py`
- `.venv/bin/python -m pytest src/tac/tests/test_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_runtime_materializer.py src/tac/tests/test_decoder_q_selective_runtime_controls.py src/tac/tests/test_decoder_q_selective_window_bridge.py -q`
- `.venv/bin/python -m py_compile experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_pair501/submission_dir/inflate.py experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_top32/submission_dir/inflate.py`
- `shasum -a 256` and `zipinfo -v` on singleton/top32 materialized archives.
- Official `inflate.sh` raw-output locality controls for singleton and top32.

## Remaining Blockers

- Exact contest auth eval not run.
- Current top32 advisory signal is not strong enough to justify promotion
  dispatch; route it back into decoder-q sign calibration and selective-window
  planning.
