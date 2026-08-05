# TK1 Receipt - TR1 PE3 Conditioning + cheapdct4 Accounting Consumers

## Answer First

TK1 implemented both owed TR1 trainer consumers, default OFF, with no scorer run and no launch.

- PE3 landed as `--pe3-conditioning-cache` + `--pe3-conditioning-mode {off,conditioning_only}`. ON parses the receiver-closed LC1/PK1 `PE3EDGE1` section, verifies section SHA `5cc024ad32df7fedb18afb75dbed6be9c1af948dac826a1736cb1084949855c2`, builds per-mode token-lattice prior channels, and feeds them into the renderer trunk through learned trust gates. It is conditioning only; no label-replacement loss exists.
- cheapdct4 landed as `--cheapdct4-pose-cache` + `--cheapdct4-pose-mode {off,accounting}`. I chose accounting, not full in-loop consumption, because full pose-renderer consumption would require a structural renderer/design decision; the small honest landing decodes OD9 `stage2_qcoeffs`, rechecks packet SHA, and reports OD9's measured n32 pose term in the trainer receipt and composed-S verdict when present.
- No `upstream/evaluate.py`, SegNet/PoseNet scorer job, MLX training launch, or pointer move occurred.

## Recall Evidence

| scope searched | query/source | finding beyond charter seeds | plan impact |
|---|---|---|---|
| Governing contract | `.omx/tmp/codex_runs/_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | PE3 is conditioning-only; scorer slot is not owned; protected files/index must be avoided; current own line is `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`. | Built only scorer-free code/tests/docs; no launch; no protected files; final line unchanged. |
| TP1/LC1 | `.omx/research/ddm_tp1_20260805/TP1_PACKET.md`, `.omx/research/ddm_lc1_20260805/LC1_RECEIPT.md` | LC1's direct PE3 target labels worsen all n32 pairs; `generator_pair_bisector` carries most introduced damage. | PE3 is input conditioning with separate per-mode gates; no target table or replacement loss. |
| PK1/PE3/PE4/RZ1 | `rg PE3EDGE1`, PK1 receipt, PE3 receipt, RZ1 parser, `experiments/inflate_runner_v4d.py` | Receiver-closed PE3 parse-back already exists; the LC1/PK1 section SHA and receiver private decoders are the reusable parse machinery. | Reused `inflate_runner_v4d` PE3 parser/record decoders and fail-closed section SHA instead of inventing a new section format. |
| OD8/OD9 | `rg stage2_qcoeffs cheapdct4`, OD9 receipt, OD8 persistence code | OD9's real carriage evidence lives in `OD9_RECEIPT.json` plus an OD5 packet; the measured d_pose is n32/subset-scoped and explicitly not projected to n600. | cheapdct4 mode requires the OD9 receipt JSON, decodes the packet, and labels the pose contribution n32-only. |
| Canonical equations | `.venv/bin/python tools/list_canonical_equations.py --json` filtered for `pe3`, `tr1`, `cheap`, `dct`, `pose_null`, `trajectory`, `conditioning` | No direct PE3 target-agreement or cheapdct4 TR1-consumer equation; adjacent pose-null/trajectory laws do not override LC1/OD9 scope. | Did not import unrelated laws or promote a score claim. |
| Memory registry | `rg -n "TR1|PE3|cheapdct4|OD9|TP1|LC1|conditioning|witness_dsl|Lever" /Users/adpena/.codex/memories/MEMORY.md` | No direct LC1/TK1 memory hit; relevant memory reminded that `build_real_trainer_parser()`/lever registry are the real parser truth surfaces. | Added DSL factories and tests that parse through the actual trainer argparse. |

## Flag Surface And Guards

| flag | default | ON value | guard/refusal |
|---|---:|---:|---|
| `--pe3-conditioning-mode` | `off` | `conditioning_only` | `conditioning_only` with no cache refuses. |
| `--pe3-conditioning-cache` | `None` | PE3 raw section / IX2 payload / archive path | Cache while mode is `off` refuses; missing path refuses; section SHA mismatch refuses. |
| `--cheapdct4-pose-mode` | `off` | `accounting` | `accounting` with no cache refuses. |
| `--cheapdct4-pose-cache` | `None` | OD9 receipt JSON | Cache while mode is `off` refuses; missing path refuses; non-JSON cache refuses; packet SHA mismatch refuses; missing OD9 pose/byte rows refuse. |

## DSL Proof

Added `src/tac/witness_dsl/tk1_pe3_conditioning_levers_20260805.py`:

- `lever_tk1_pe3_conditioning(cache_path)` emits only real TR1 flags.
- `lever_tk1_cheapdct4_pose_accounting(cache_path)` emits only real TR1 flags.
- Both carry `score_claim=False`, `default_off_byte_identity=True`, runtime receipt schema names, and constant provenance.
- Focused registry test confirms both factories appear under `experiments/train_tr1_partition_renderer_mlx.py` with `missing_flags == ()`.

## OFF Proof

MEASURED/static-unit:

- Parser defaults are OFF and cache paths are `None`.
- `TR1Config` has no PE3 or cheapdct4 fields, so default config hashing stays structurally unchanged.
- PE3 ON attaches `_pe3_conditioning` and a trainable `pe3_conditioning_gate`; OFF attaches neither and leaves `raw_tokens()` unchanged.
- cheapdct4 ON builds a receipt/accounting object only; OFF attaches no object and does not touch model/loss/optimizer state.

OWED empirical proof:

- Actual checkpoint-byte equality remains owed to MAIN's next MLX smoke because this sandbox cannot import `mlx.nn` with Metal unavailable. This matches the BI1 caveat; no empirical checkpoint-byte claim is made here.

## Verification

Passed:

- `.venv/bin/python -m py_compile experiments/train_tr1_partition_renderer_mlx.py src/tac/witness_dsl/tk1_pe3_conditioning_levers_20260805.py src/tac/tests/test_tk1_tr1_conditioning_consumers.py src/tac/witness_dsl/tests/test_tk1_pe3_conditioning_levers.py`
- `.venv/bin/python -m pytest src/tac/tests/test_tk1_tr1_conditioning_consumers.py src/tac/witness_dsl/tests/test_tk1_pe3_conditioning_levers.py -q` -> `42 passed`
- `.venv/bin/python -m pytest src/tac/witness_dsl/tests/test_bi1_birth_seed_levers.py -q` -> `4 passed`
- `.venv/bin/python -m ruff check src/tac/witness_dsl/tk1_pe3_conditioning_levers_20260805.py src/tac/tests/test_tk1_tr1_conditioning_consumers.py src/tac/witness_dsl/tests/test_tk1_pe3_conditioning_levers.py` -> clean
- `git diff --check -- experiments/train_tr1_partition_renderer_mlx.py src/tac/witness_dsl/tk1_pe3_conditioning_levers_20260805.py src/tac/tests/test_tk1_tr1_conditioning_consumers.py src/tac/witness_dsl/tests/test_tk1_pe3_conditioning_levers.py` -> clean
- Review tracker: two `mark-file --status reviewed` passes recorded for each touched Python file.

Not clean, scoped:

- `src/tac/tests/test_ddm_tb1_tr1_renderer.py` still fails in this sandbox at `mlx.nn` import with `No Metal device available`; no TK1 assertion was reached.
- Whole-file `ruff check experiments/train_tr1_partition_renderer_mlx.py` still reports 10 historical style findings outside the TK1 block.
- Full legacy `src/tac/tests/test_lever_registry.py` still reports pre-existing stale `--integer-plane-emitter-*` flags; TK1's package-registry assertions pass.

## NEXT_IF_RESUMED

```json
{
  "schema": "ddm_tk1_next_if_resumed.v1",
  "status": "IMPLEMENTED_NO_LAUNCH_NO_SCORE",
  "scorer_runs_by_tk1": 0,
  "launches_by_tk1": 0,
  "pe3_consumer": "conditioning_only_input_prior_with_per_mode_trust_gates",
  "cheapdct4_consumer": "accounting_hook_decodes_od9_stage2_qcoeffs_and_reports_n32_pose_term",
  "full_in_loop_cheapdct4_consumption": "OWED_DESIGN_DECISION_IF_REQUIRED",
  "empirical_off_checkpoint_byte_proof": "OWED_AT_MAIN_MLX_SMOKE",
  "recompiled_ticket_note": "Recompile TP1 tickets after this landing before any full crossed A/B; old ticket hashes do not include TK1 flags.",
  "score_claim": false,
  "contest_pointer": "borrowed/unmoved"
}
```

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
