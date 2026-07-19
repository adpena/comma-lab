# C1 two-independent-plane receiver timing — Codex handoff

Date: 2026-07-19 UTC
Lane: `lane_c1_two_plane_receiver_timing_20260719`
Task: #561 / SPEC_v10 §8 C1
Landing authority: delegated worktree only; **MAIN landing review required**

## Verdict

**C1 STRUCTURAL APPARATUS BUILT; FULL-N600
`BLOCKED_ENVIRONMENT_SSD_WRITE_CUSTODY`.**

**TIME CLASS: `CLOSE -> MODAL_MEASUREMENT_OWED` (`DEFAULT_UNMEASURED`).**
No admissible paired local/full-official contest calibration exists, and no C1
full-n600 local timing row exists. This is not a pass/fail inference from
`local < 1800`: the storage gate refused before materialization or decode.

The canonical pointer remains **`0.1910828242 [contest-CPU] UNMOVED`**. This
unit performed no training, provider launch, paid dispatch, official
evaluation, score claim, promotion, or submission action.

## Landed apparatus

Commit `e2e8a5e55e74a993708f4a085ba0b2a3ebef0081` adds:

- a scorer-free strict receiver for `predictor-residual-u8.v1` plus
  `description-frame0.v1`;
- independent factor-2 exact integer solves for Y0/Pose frame 0 and Y1/Pose
  frame 1 plus Seg, with no copied, aliased, or repeat-frame plane;
- one-worker attribution and fixed-order four-worker process-pool modes,
  including worker-side exact-numerator verification and parallel
  double-decode identity gates;
- write-once plane-0, pair, and 12-pair chunk checkpoints with digest-bound
  resume validation, plus reopened assembled-output verification;
- separate parse, expansion, solve0, solve1, assembly/I/O, and verification
  timing, including per-intervention pair rows;
- exact official ABI handling for unzipped `archive/0.bin`, canonical ZIP
  reconstruction, executable-mode `inflate.sh`, scored sibling
  `archive.zip` binding, and non-sibling output roots;
- an integer-only MLX/Metal local twin whose rows are always
  `[macOS-MLX research-signal]`, `score_claim=false`, and excluded from the
  contest timing verdict;
- a fail-closed prepare → independent serial/parallel inflate → compose DAG
  with exact archive, packet, source, output, stage, chunk, adapter, runtime,
  Git, host, and environment custody.

The runtime-custody check now proves that the declared checkout reproduces all
four runtime sources. At the source commit it reports
`remote_checkout_reproduces_sources=true` for the measurement tool, timed
receiver, existing production receiver, and existing exact integer solver.

Implementation was frozen first in
`.omx/research/c1_two_plane_receiver_timing_implementation_spec_20260719.md`.
No PDW, trainer, scorer, source cache, evaluator, or pointer file was edited.

## Full-n600 storage blocker

The canonical storage waterfall was rerun for **25,769,803,776 bytes**. The
preferred SSD had **816,407,822,336 free bytes** and
**773,458,149,376 usable bytes**, but this delegated sandbox could not create
or write the exact required root:

`/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/`

The fail-closed preflight receipt is
`.omx/research/c1_two_plane_receiver_storage_blocker_20260719.json`, SHA-256
`39c22f7ca08900dd3cefae3035e6c326a3982c61905b1aca3c6b87e0650a6383`.
Its refusal set is `selected_workload_root_missing`,
`selected_workload_root_mismatch`, `mkdir_failed:PermissionError`,
`workload_root_missing`, and `write_probe_failed`. Local and ephemeral storage
were not substituted. The sacred donor
`experiments/results/levelset_n600_witness_20260717T113932Z/` was not written.

The content-addressed failure receipt is
`.omx/research/c1_two_plane_receiver_timing_blocked_receipt_20260719.json`,
SHA-256
`0f6c3f9257fef3b7e6d7112453a2ff439db2ee1b08ca0e5eaa3f4f04f3570297`.
It explicitly marks every missing n600 measurement and every authority field
false.

Consequently, these remain **NOT MEASURED**:

- full-n600 serial receiver timing;
- either fresh full-n600 four-worker timing;
- full-n600 parallel double-decode byte identity;
- native-fp32 Seg/Pose hard-oracle observations on the six real pairs;
- M5 Max MLX/Metal parity or timing;
- full official contest evaluation time.

## Frozen input and exact workload custody

- `gt_n600.npz`: 5,078,017,610 bytes, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
- Y0 raw-C SHA-256:
  `5e86e419cdd5bd41c9482cabc78cf27cec22281098b64c715d91f1f067d11566`.
- Y1 raw-C SHA-256:
  `6a731946e3d9de82089c90de9784c5a5bc72c607c963fb6f79dac16f00ac89bc`.
- Full exact integer proof: **707,788,800 numerator values** — DERIVED from
  `600 pairs × 2 planes × 384 × 512 × 3`.
- Raw output per decode: **3,662,409,600 bytes** — DERIVED from
  `600 pairs × 2 frames × 874 × 1164 × 3`.
- CUDA follow-on: **707,788,800 numerator values**, **2,831,155,200 uniform
  2×2 tap-products**, and **3,662,409,600 output bytes**. This is
  `DERIVED_UNMEASURED_CUDA_WORKLOAD`; no CUDA seconds or verdict are claimed.

The native-fp32 hard-oracle gate cites
`f32_receiver_arithmetic_exactness_admissibility_v1` from
`src/tac/canonical_equations/f32_receiver_arithmetic_law_20260719.py`. It
requires both complete camera frames through official PoseNet 2×2/YUV6 and
frame 1 through SegNet. Frozen pair IDs are `90,175,277,381,424,573`.
Production composition forbids an injected oracle runner.

## Timing-anchor audit and verdict boundary

No same-archive pair proves both local receiver time and a full official
contest evaluation:

| Anchor | Local | Contest | Scope |
|---|---:|---:|---|
| Task #543 | 4.53 / 4.50 s n12; 226.5 s n600 projection | none | inflate-only; n600 is DERIVED |
| PR128 receiver | 80.0797 / 82.6509 s | 60.2853 s | same-archive inflate-only; contest/local 0.729–0.753 |
| PR128 complete path | 348.531 / 358.285 s custom local total | 189.364 s official-equivalent contest total | local receipt says `upstream_evaluate_py_run=false`; not a paired official evaluation |

PR128 binds archive SHA-256
`196acd18e4ca10a3ab0d826436aa46014a44cba8a55eb4abf9931876cc7e98b5`
(176,564 bytes). The contest row used the complete functional path
`archive.zip -> inflate.sh -> upstream/evaluate.py --device cpu`; it was not a
literal `evaluate.sh` invocation. These rows authorize neither a transfer
margin nor a C1 pass/fail.

Review round 5 found that self-attested calibration JSON could otherwise
launder `CLEARLY_UNDER` or `CLEARLY_OVER`. The final fail-closed closure makes
production composition reject every caller-supplied calibration path and
therefore default `CLOSE` until a canonical validator parses inner receipts
and proves same-archive durations plus a complete official evaluation. No
sixth review round was performed; MAIN must independently review this closure.

The **1,800-second constant is for the complete official evaluation** on
contest hardware: `T_inflate + T_scoring`. It is not per inflate invocation.
Only the exploited configuration can support a timing verdict; serial is
attribution only.

## Modal measurement debt

After successful full-n600 composition, the tool can emit an unfired ticket
bound to the exact archive and executable adapter bytes. The ticket runs the
literal full `upstream/evaluate.sh` path, uses CPU `4 cores / 16 GiB` first,
and keeps T4 blocked until a deterministic CUDA receiver and parity custody
exist. It also binds the #381 `<= $20` envelope, the lane claim through
`tools/claim_lane_dispatch.py`, `.omx/state/active_lane_dispatch_claims.md`,
and `.omx/state/modal_call_id_ledger.jsonl`.

No ticket JSON exists in this unit because no exact full-n600 archive was
prepared. No ticket was dispatched. The structural emitter also refuses
readiness unless the declared Git checkout reproduces every runtime source
byte.

## Verification

- `76 passed` across the new receiver and measurement-tool suites: 32 receiver
  tests and 44 tool tests.
- Ruff check: PASS; Ruff format check: PASS; `py_compile`: PASS;
  `git diff --check`: PASS.
- Final source SHA-256 values:
  - receiver:
    `f7ce598e067874813fcabc5b8767d368c4fae8e38c3af29fdff3868004b84b85`;
  - receiver tests:
    `5463375e6c9cc58eca9d33cf39ce7f08110571ee0cb0905d739ce839ca202a4c`;
  - measurement tool:
    `2950ba48bbba0c6be5db57158a651dcdc2403bf8498605c39a8bdd2d0b595d01`;
  - tool tests:
    `8b39559e99d8bd46ba62df3d96285e7afd9d74cf842eac78f523bac309d4652c`.
- An actual subprocess smoke used the generated executable adapter from an
  arbitrary working directory and a non-sibling output root. Serial and
  official-ABI outputs were both 576 bytes with SHA-256
  `eb3e20200ddd44695db95885e1bc338e327516e03db047e69af1db2dde90a7e9`.
  This is a six-pair tiny fixture only, never timing or contest authority. Its
  durable receipt is `.omx/research/c1_two_plane_receiver_cli_smoke_20260719.json`,
  SHA-256
  `764f6f01e0fc4a0ce5b83c9b993ecfe37299346b16ab8dbef880fd400a735823`.
- A real four-worker process-pool capability probe refused with
  `PermissionError: [Errno 1] Operation not permitted`; inline executor tests
  prove orchestration only and are not a timing row.
- MLX imports, but Metal execution refuses with `No Metal device available`.
  This is `BLOCKED_HOST_METAL_DEVICE_UNAVAILABLE`, not parity and not a
  contest-CPU blocker.

The environment-blocker receipt is
`.omx/research/c1_two_plane_receiver_environment_blockers_20260719.json`,
SHA-256
`859febfaf73b9519717be8884a0ac2fd282b5b464c13505cc208392893e635fc`.

Five bounded review rounds were used. Rounds 1–4 closed receiver persistence,
streaming custody, resume, MLX authority, official ABI, wall-boundary,
adapter-mode, runtime-source, oracle-injection, archive re-open, and ticket
TOCTOU findings. Round 5 produced the calibration finding and its final
fail-closed closure described above. The code is test-green, but that closure
has intentionally not been represented as a sixth independent review.

## Reactivation contract

1. MAIN independently reviews and lands both delegated commits, especially
   the post-round-5 production calibration refusal.
2. Grant write custody for the exact SSD root above and rerun `prepare`; do not
   opt into local storage.
3. Run one fresh serial attribution process and two fresh four-worker
   processes. Require identical archive/raw/stage/chunk custody and parallel
   double-decode identity.
4. Run the six-real-pair native-fp32 hard oracle and, on an M5 Max with usable
   Metal, the false-authority MLX parity gate.
5. Compose the content-addressed receipt. It must remain `CLOSE` until a
   canonical paired full-evaluation validator exists, then emit the exact
   one-shot Modal ticket.
6. Only after separate dispatch authority: claim the lane, run full official
   `evaluate.sh`, and append both dispatch ledgers. That contest measurement
   alone can settle the 1,800-second budget.

## Triality and stores consulted

- DSL: strict two-plane packet/receiver ABI and official adapter contract.
- DAG: storage gate → prepare → serial → parallel twice → hard oracle →
  compose → governed full-evaluation ticket.
- Equations: exact factor-2 integer numerator equality and the declared
  native-fp32 admissibility law.

Stores consulted:

- `.omx/research/SPEC_v10_integer_plane_vehicle_20260719.md` (§§3, 6, 8)
- `.omx/research/yhat_native_generator_20260719_codex.md`
- `.omx/research/c1_two_plane_receiver_timing_implementation_spec_20260719.md`
- `.omx/research/production_receiver_543_byteclose_receipt_20260719.json`
- `.omx/research/integer_plane_vehicle_spec_20260719_codex.md`
- `experiments/results/jrd_pr128_completion_20260713T022712Z/runtime_inflate_proof.json`
- `experiments/results/jrd_pr128_completion_20260713T022712Z/measurement_receipt.json`
- `experiments/results/modal_import_candidate_exact_cpu_20260712/contest_auth_eval.json`
- `experiments/results/modal_import_candidate_exact_cpu_20260712/modal_cpu_auth_eval_result.json`
- `src/tac/canonical_equations/f32_receiver_arithmetic_law_20260719.py`
- `docs/operating_manual_craft_handoff.md`
- `upstream/README.md` (official complete-evaluation budget)
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/modal_call_id_ledger.jsonl`

This artifact is a delegated-worktree result. **MAIN must independently review
the branch diff, exact hashes, five-round cap, final fail-closed calibration
closure, blocker scope, and pointer non-movement before landing.**
