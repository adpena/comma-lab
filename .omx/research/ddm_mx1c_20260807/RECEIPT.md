# ddm_mx1c Receipt - Governed-Fire Adoption for Lifted PR130 Entrypoints

Date: 2026-08-07
Axis: structural launch/governor adoption; CPU-only sandbox verification
Score claim: false
Promotion eligible: false
Tokens: [no-triality] [p0-ledger-ok]

borrowed_substrate_accounting: PR130 lifted renderer and pose-carrier training mechanisms remain
borrowed/intake substrate. This landing is OUR structural governor adoption, launch-ticket fire
wrapping, admission-coverage class fix, and tests.

## Fire-Ready Governed Args

The first executable fire is `argv_n32_arm_cap` from
`.omx/research/ddm_mx1c_20260807/launch_ticket_v2_two_arm_governed.json`:

```bash
.venv/bin/python tools/safe_run.py --rss-mb 90000 --timeout 28800 --projected-gib 66.268951 --label ddm_mx1_row1_launch_arm_cap_n32_metal -- .venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode mlx-train --device gpu --pairs 32 --steps 6000 --lr 2e-07 --ce-fraction 0.0 --softplus-fraction -999.0 --bits 4 --seed 20260806 --checkpoint-every 250 --eval-every 250 --input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --init /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt --run-dir .omx/research/ddm_mx1c_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal --out .omx/research/ddm_mx1c_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/result.json
```

The second n32 fire is `argv_n32_arm_veh`:

```bash
.venv/bin/python tools/safe_run.py --rss-mb 90000 --timeout 28800 --projected-gib 66.268951 --label ddm_mx1_row1_launch_arm_veh_n32_metal -- .venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode mlx-train --device gpu --pairs 32 --steps 6000 --lr 2e-07 --ce-fraction 0.0 --softplus-fraction -999.0 --bits 4 --seed 20260806 --checkpoint-every 250 --eval-every 250 --input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt --target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --init /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt --run-dir .omx/research/ddm_mx1c_20260807/row1_v2_two_arm/launch_arm_veh/n32_metal --out .omx/research/ddm_mx1c_20260807/row1_v2_two_arm/launch_arm_veh/n32_metal/result.json
```

n120 argvs are present in the same JSON, but they are queued behind the written selection rule:
no n120 dispatch until the two n32 CPU-torch verdicts select the scaled arm.

## Safe-Run Projection

Source: `.omx/research/ddm_mx1b_20260806/mem_probe_cpu_result.json`.

Arithmetic recorded in the ticket:

`1.268951 GiB measured CPU-side peak + 65.000000 GiB Metal/model/scorer/load-step unknown margin = 66.268951 GiB`

Wrapper policy encoded for every `mlx-train` fire argv:

- `tools/safe_run.py --rss-mb 90000 --timeout 28800 --projected-gib 66.268951`
- Scheduling: `SEQUENTIAL one-Metal-fire-at-a-time`
- RR8-F1 cure: a successful enumerator is required before every fire; if `pgrep` returns `rc>=2` and `ps axo command` also fails or is denied with `rc!=0`, REFUSE. A denied enumerator is never quiescence.

## What Changed

- `experiments/ddm_mx1_pr130_semantic_renderer.py`: adopted `assert_governed_admission()` for
  `mlx-train` and `torch-smoke`; `probe` and `mlx-parity` remain ungated, and `mem-probe`
  remains the light clearance mode required before fire.
- `src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py`: adopted
  `assert_governed_admission()` in `main()` before the lifted training code runs.
- Row-1 v2 two-arm ticket regenerated at
  `.omx/research/ddm_mx1c_20260807/launch_ticket_v2_two_arm_governed.json`; all four
  `mlx-train` fire argvs are real `safe_run.py` argvs, not prose-only wrappers.
- `src/tac/preflight.py`: extended the existing #254 static scan rather than adding a twin
  scanner. It now also covers top-level `experiments/*.py` and `src/tac/**/train*.py`
  argparse surfaces that expose a train mode plus GPU/Metal/CUDA/MPS device choice.
- Tests cover enforced-env refusal with exit 7, governed-env pass, light mx1 probe mode
  unaffected, ticket wrapper shape, RR8-F1 fire protocol, class-fix positive controls, waiver
  handling, and the live lifted target entrypoints not being flagged.

## Gate Coverage

Command:

```bash
.venv/bin/python - <<'PY'
from tac.preflight import check_heavy_witness_trainers_call_admission_guard
v = check_heavy_witness_trainers_call_admission_guard(strict=False, verbose=True)
print(any('experiments/ddm_mx1_pr130_semantic_renderer.py' in x for x in v))
print(any('src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py' in x for x in v))
PY
```

Observed:

- `145 violation(s) (150 heavy trainer(s) scanned; witness + substrate + argparse gpu/train entrypoints)`
- `experiments/ddm_mx1_pr130_semantic_renderer.py` flagged: `False`
- `src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py` flagged: `False`

The 145 violations are the existing warn-only historical backlog, not a strict blocker for this
arm.

## RECALL EVIDENCE

| source searched | query / command | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing files | Read the mx1c charter, common contract, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md` | Live board owns `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; scorer slot was not assigned to this arm. | Kept this landing structural and made no score or promotion claim. |
| RR8/MX1B receipts | Read `.omx/research/ddm_rr8_20260806/ROUND8_FINDINGS.md`, `.omx/research/ddm_rr8_20260806/CHARTER_ADDENDUM.md`, and `.omx/research/ddm_mx1b_20260806/RECEIPT.md` | RR8-F1 defined the denied-enumerator fail-open; RR8-F3 identified lifted entrypoint/gate bypass; MX1B provided CPU peak `1.268951 GiB` and required a passed Metal mem-probe before fire. | Encoded fail-closed liveness text, safe-run projection arithmetic, and kept Metal mem-probe as a pre-fire clearance blocker. |
| Source surfaces | Searched `src/tac/admission_guard.py`, `tools/safe_run.py`, `src/tac/preflight.py`, mx1, and mx2b pose trainer for admission/governor use | #254 is opt-in per entrypoint; safe_run supports `--projected-gib`; the previous static scan vocabulary was filename-family based and missed lifted argparser-shaped trainers. | Added direct asserts and extended the existing scanner vocabulary. |
| Canonical equation/search corpus | `.venv/bin/python tools/list_canonical_equations.py --json` filtered for `admission`, `govern`, `oom`, `memory`, `pr130`, `mx1`, `mx2`, `safe_run`, `metal`; `rg` over research index, DAG/state, docs, experiments, src, tools for `PR130`, `mx1`, `admission`, `governor`, `safe_run`, `RR8`, `Metal`, `OOM` | Found admission/memory-governor authority and PR130 Row-1 queue context; did not find a superseding PR130-lift equation or a current scored row from this surface. | Preserved Row-1 plan and made the receipt a structural fire-readiness artifact, not a score artifact. |

## Verification

```bash
.venv/bin/python -m pytest experiments/tests/test_ddm_mx1_memory_probe.py src/tac/pr130_lift/tests/test_mx2_pose_resumable_state.py src/tac/tests/test_admission_coverage_gate.py -q
.venv/bin/python -m ruff check experiments/ddm_mx1_pr130_semantic_renderer.py src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py src/tac/preflight.py experiments/tests/test_ddm_mx1_memory_probe.py src/tac/pr130_lift/tests/test_mx2_pose_resumable_state.py src/tac/tests/test_admission_coverage_gate.py
.venv/bin/python -m py_compile experiments/ddm_mx1_pr130_semantic_renderer.py src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py src/tac/preflight.py experiments/tests/test_ddm_mx1_memory_probe.py src/tac/pr130_lift/tests/test_mx2_pose_resumable_state.py src/tac/tests/test_admission_coverage_gate.py
git diff --check -- experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py src/tac/pr130_lift/tests/test_mx2_pose_resumable_state.py src/tac/preflight.py src/tac/tests/test_admission_coverage_gate.py .omx/research/ddm_mx1c_20260807/launch_ticket_v2_two_arm_governed.json .omx/research/ddm_mx1c_20260807/row1_v2_two_arm_ticket_result.json
```

Results: focused pytest `20 passed`; ruff passed; py_compile passed; diff check passed.

Local ticket generation exited 0 and wrote JSON, but the local MLX probe remains `status=blocked`
because this sandbox has no accessible Metal device. No Metal training, scorer run, archive build,
or `upstream/evaluate.py` run was performed.

Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
