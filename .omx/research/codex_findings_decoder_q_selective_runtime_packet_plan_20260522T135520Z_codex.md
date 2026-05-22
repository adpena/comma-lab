# Codex Findings: Decoder-Q Selective Runtime Packet And Materialization

- UTC: 2026-05-22T13:55:20Z
- Lane: `lane_codex_decoderq_selective_runtime_packet_20260522`
- Status: `runtime_packet_l0_materialized_control_pending`
- Score authority: `false`
- Ready for exact eval: `false`

## What Landed

Implemented a reusable L0 packet planner for the decoder-q selective runtime
blocker and a trailer-based materializer/locality harness:

- `src/tac/optimization/decoder_q_selective_runtime_packet.py`
- `src/tac/optimization/decoder_q_selective_runtime_materializer.py`
- `tools/plan_decoder_q_selective_runtime_packet.py`
- `tools/materialize_decoder_q_selective_runtime_candidate.py`
- `tools/run_decoder_q_selective_runtime_locality_controls.py`
- `src/tac/tests/test_decoder_q_selective_runtime_packet.py`
- `src/tac/tests/test_decoder_q_selective_runtime_materializer.py`
- `src/tac/tests/test_decoder_q_selective_runtime_controls.py`

The planner validates the existing selective-window bridge plan against the
FEC6 parent archive and the full-video materialized decoder-q mutation. It
recomputes the q mutation from the parent decoder bytes and verifies the
mutated decoder SHA matches the materialized candidate before emitting a compact
archive-local `DQS1` payload contract.

The canonical materialized grammar is a tail extension inside archive member
`x`: legacy `FP11` + FEC6 selector bytes, followed by a charged `DQS1` trailer.
Do not use the discarded wrapper/prepend grammar; it costs eight extra bytes
and creates incompatible parser behavior.

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

Singleton pair `[501, 502]` maps to affected frames `[1002, 1003]` under
`pair_all_frames`, with a charged `DQS1` payload of 13 bytes and estimated
rate-score delta `0.000008656166390588228`. Top32 maps to 64 affected frames,
with a charged `DQS1` payload of 75 bytes and estimated rate-score delta
`0.00004993942148416285`.

Generated a top32 trailer-form packet at:

- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_runtime_trailer_top32/archive.zip`

Top32 materialized archive SHA-256:
`3c4e15bfe7ae1004ad23e89a52c2836e609c1f99e25b58f45c01747226705d59`.
Member `x` is `178492` bytes: `+75` bytes versus FEC6.

## Critical Design Guard

The full-video materialized decoder-q candidate differs from the FEC6 parent in
112,482 decoder bytes because one q-symbol mutation is recompressed through
split Brotli. A selective runtime must therefore apply the q-domain patch before
decoder state construction; it must not store or replay a recompressed byte-diff
of the decoder blob.

The runtime extension is FP12-compatible in spirit: legacy FP11 source plus FEC6
selector, followed by charged DQS1 patch bytes inside archive member `x`.
The legacy FP11 parser rejects trailing bytes and is not reusable unmodified.

## Verification

- `ruff check experiments/contest_auth_eval.py src/tac/optimization/decoder_q_selective_window_bridge.py src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_decoder_q_selective_window_bridge.py src/tac/optimization/decoder_q_selective_runtime_materializer.py src/tac/optimization/decoder_q_selective_runtime_packet.py tools/materialize_decoder_q_selective_runtime_candidate.py tools/plan_decoder_q_selective_runtime_packet.py tools/run_decoder_q_selective_runtime_locality_controls.py src/tac/tests/test_decoder_q_selective_runtime_materializer.py src/tac/tests/test_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_runtime_controls.py`
- `.venv/bin/python -m pytest src/tac/tests/test_contest_auth_eval.py::test_run_inflate_defaults_python_to_current_interpreter src/tac/tests/test_decoder_q_selective_window_bridge.py src/tac/tests/test_decoder_q_selective_runtime_materializer.py src/tac/tests/test_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_runtime_controls.py -q`
- `.venv/bin/python -m py_compile experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/decoder_q_selective_runtime_trailer_top32/inflate.py`

## Locality Result

The first top32 locality run used the stale selected-subset decoder path and
failed selected-frame parity (`64` selected mismatches, `0` unselected
regressions). The root cause was batch-shape drift: selected latents were
decoded alone, while the global decoder-q runtime decodes the full active
16-pair batch.

The materializer now decodes the full active batch with the mutated decoder
whenever that batch contains a selected pair, then splices selected offsets.
The rerun passed:

- Artifact:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/dqs1_top32_locality_controls_fullbatch.json`
- `locality_controls_passed=true`
- `selected_frame_mismatch_count=0`
- `unselected_frame_mismatch_count=0`
- selected frames match the full global decoder-q runtime hash
  `cff4cc7400011f2d1cbe0e4a19c7aeb66af1b409d0cdad9868b86edcb23e2cea`
- unselected frames match the parent hash
  `d99cf5d5f735249a66bea6a8ed02bd45d44abbf9e79e20cfa801756cb95d0654`

## Advisory Scorer Result

Local scorer smoke passed on `[macOS-CPU advisory]`:

- Artifact:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/dqs1_top32_cpu_advisory_venv.json`
- `canonical_score=0.1920602563025898`
- `archive_size_bytes=178592`
- `avg_posenet_dist=0.00002943`
- `avg_segnet_dist=0.00055988`
- `rate_unscaled=0.004756683286239793`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `hardware_compliance_blocker=contest_cpu_requires_linux_x86_64`

Same-axis local comparison against the existing FEC6 macOS advisory baseline
`0.19206131688110561 [macOS-CPU advisory]` is
`-0.0000010605785158157577` lower for DQS1 top32, despite `+75` archive bytes.
This is not a contest score and does not promote the lane. The naive comparison
to the Linux x86_64 `[contest-CPU]` frontier is `+0.000008939421484194243`, but
that mixes axes and remains non-authoritative until exact CPU/CUDA replay.

The full-video global decoder-q candidate was much worse on exact CPU
(`0.19244523120613244 [contest-CPU]`). Selective DQS1 therefore fixed the
global-mutation blast-radius issue at locality level and produced a useful
local advisory signal, but it still needs exact CPU/CUDA replay before any
score, rank, kill, or promotion claim.

## Meta-Bugs Hardened

- `experiments/contest_auth_eval.py` now defaults `PACT_PYTHON_BIN` to
  `sys.executable` alongside `PYTHON` and `PYTHON_BIN`, because generated FEC6
  inflaters may prioritize `PACT_PYTHON_BIN`. This prevents a local/system
  Python without `brotli` from being selected during auth eval.
- `src/tac/optimization/decoder_q_selective_window_bridge.py` no longer emits
  the stale `blocked_missing_decoder_q_selective_runtime_grammar` status.
  Future bridge plans now route to DQS1 tail-trailer materialization while
  preserving false-authority blockers.

## Remaining Blockers

- Exact contest auth eval not run.
- Lane claim and remote exact CPU/CUDA dispatch not run.
