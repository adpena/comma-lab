# ddm_mx1d Receipt - Hard MLX Cap + Fire Guard

Date: 2026-08-07
Axis: structural launch safety / scorer-free
Score claim: false
Promotion eligible: false
Tokens: [no-triality] [p0-ledger-ok]

## What Now Binds

The Row-1 mx1 Metal fire now has two fail-closed gates that did not exist in mx1c:
the MLX process refuses GPU mode unless `set_memory_limit(..., relaxed=False)` is
actually applied, and `mlx-train --device gpu` refuses unless it can read a passed
`ddm_mx1_fire_guard_verdict.v1`. The guard verdict is produced by
`tools/mx1_fire_guard.py` from the launch ticket plus the matching mem-probe receipt,
so a safe_run projection or admission banner alone can no longer clear fire.

## Code Changes

- `experiments/ddm_mx1_pr130_semantic_renderer.py`
  - Default memory budget is now 35% of available memory, not 50%.
  - Mem-probe default budget is capped at `min(24GB, 35% default)`.
  - MLX memory limit setup uses `inspect.signature` and calls the current
    `set_memory_limit` API with `relaxed=False` when the installed API supports it.
  - Deprecated `metal.set_memory_limit` / `metal.set_cache_limit` calls were removed
    from the limit-setting path.
  - GPU `mlx-train` requires `--fire-guard-verdict`; absent or failed verdict exits 9.
  - `--allow-soft-mem-limit` exists as the explicit operator escape hatch, otherwise a
    soft-only MLX limit refuses GPU mode.
  - Mem-probe receipts are written atomically and include host/config/memory-limit
    evidence. Non-environment probe execution errors write `status=failed` receipts.
- `tools/mx1_fire_guard.py`
  - New guard tool. Given `--ticket` and `--argv-key`, it refuses unless the keyed
    mem-probe receipt exists, parses, has `status=passed`, has MLX/load-stage samples,
    has a hard or explicitly-waived memory-limit record, and matches host/config.
  - Persists a typed `ddm_mx1_fire_guard_verdict.v1` beside the fire path.
- `experiments/tests/test_ddm_mx1_memory_probe.py` and
  `tools/tests/test_mx1_fire_guard.py`
  - Cover guard refuse/pass paths, relaxed detection, 35%/24GB budget derivation,
    failed receipt schema, and entrypoint refusal for absent/failed guard verdicts.

## Ticket Artifacts

Generated:

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode probe --run-dir .omx/research/ddm_mx1d_20260807/row1_v2_two_arm --out .omx/research/ddm_mx1d_20260807/row1_v2_two_arm_ticket_result.json --launch-ticket-path .omx/research/ddm_mx1d_20260807/launch_ticket_v3_fire_guarded.json
```

Ticket summary:

- Ticket: `.omx/research/ddm_mx1d_20260807/launch_ticket_v3_fire_guarded.json`
- Driver result: `.omx/research/ddm_mx1d_20260807/row1_v2_two_arm_ticket_result.json`
- Guard precheck verdict:
  `.omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json`
- `main_fire_sequence`: `guard_precheck -> probe -> gate -> fire`
- `mem_probe_receipt_paths` are keyed for all four argv keys, so CAP and VEH cannot share a
  mismatched input-cache receipt.
- First CAP probe input cache:
  `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt`
- First VEH probe input cache:
  `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt`

Local guard precheck:

```bash
.venv/bin/python tools/mx1_fire_guard.py --ticket .omx/research/ddm_mx1d_20260807/launch_ticket_v3_fire_guarded.json --argv-key argv_n32_arm_cap --out .omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json
```

Observed rc: `4` as expected in the CPU sandbox.

Verdict:

```json
{"schema":"ddm_mx1_fire_guard_verdict.v1","status":"failed","reason_code":"mem_probe_receipt_missing","argv_key":"argv_n32_arm_cap"}
```

This is the correct pre-probe state: the ticket cannot fire until MAIN writes the keyed
Metal mem-probe receipt and reruns the guard to a passed verdict.

## RECALL EVIDENCE

| source searched | query / command | found beyond charter seeds | changed plan |
|---|---|---|---|
| Governing files | Read `mx1d_prompt.md`, `_common_contract.md`, `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | Live board is scorer/full-n600 constrained and owns no score work for this arm; own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`. | Kept the landing scorer-free, structural, and pointer-honest. |
| Incident memory + RR9 | Read `concurrent_metal_fires_without_composed_preflight_oomed_the_machine_20260806.md` and `.omx/research/ddm_rr9_20260807/ROUND9_FINDINGS.md` | The second machine-down happened after safe_run and #254 passed because MLX limits were soft and the ticket-required receipt was absent. | Made hard MLX limit and receipt-consuming fire guard both entrypoint-enforced. |
| Prior mx1 receipts | Read/searched `.omx/research/ddm_mx1b_20260806` and `.omx/research/ddm_mx1c_20260807` for `mem_probe_receipt_required`, `argv_n32_arm_cap`, and launch-ticket fields | mx1c wrapped fires with safe_run but still exposed fire argv without a machine-checkable receipt gate. | Preserved the two-arm ticket shape but inserted guard/probe/gate/fire and keyed receipts. |
| Source surfaces | Searched/read `experiments/ddm_mx1_pr130_semantic_renderer.py`, `tools/safe_run.py`, `src/tac/admission_guard.py`, and focused tests | Existing mem-probe receipt shape existed but did not atomically testify failed paths and did not bind fire argv. | Added receipt host/config/memory-limit fields, atomic write, and train-time verdict refusal. |
| Canonical equations registry | `.venv/bin/python tools/list_canonical_equations.py --json | jq ... admission/govern/memory/safe_run/metal/oom/pr130/mx1 ...` | Found `adaptive_ceiling_admission_control_v1` and `oom_verdict_batch_spike_peak_rss_v1`; no equation superseded RR9-F1. | Kept system admission as necessary but not sufficient; added MLX-local and ticket-receipt gates. |
| Memory registry | `rg -n "mx1d|MX1D|common_contract|lane|charter" /Users/adpena/.codex/memories/MEMORY.md` | Reinforced Pact lane/artifact/score-custody boundaries; no mx1d-specific prior memory row found. | No score or dispatch claim was added. |

## Verification

```bash
.venv/bin/python -m pytest experiments/tests/test_ddm_mx1_memory_probe.py tools/tests/test_mx1_fire_guard.py -q
.venv/bin/python -m ruff check experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py
.venv/bin/python -m py_compile experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py
git diff --check -- experiments/ddm_mx1_pr130_semantic_renderer.py experiments/tests/test_ddm_mx1_memory_probe.py tools/mx1_fire_guard.py tools/tests/test_mx1_fire_guard.py .omx/research/ddm_mx1d_20260807/launch_ticket_v3_fire_guarded.json .omx/research/ddm_mx1d_20260807/row1_v2_two_arm_ticket_result.json .omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json
```

Results:

- Focused pytest: `14 passed`
- Ruff: passed
- Py compile: passed
- Diff check: passed
- Review tracker: two `mark-file` passes recorded for all four touched Python files
  (`mx1d_review_pass1`, `mx1d_review_pass2`)

## Hashes

```text
765805f82c33daeb2d9bc0b4537426b39894bb50ca4f97be06bc1e42367a0b18  experiments/ddm_mx1_pr130_semantic_renderer.py
037609a9c1c21fdd8f5ec6d6f024679ff69dbbccc95039abf6b5ecca3f9e8817  experiments/tests/test_ddm_mx1_memory_probe.py
0b8ca969eba2b9aecda555e92116ca640213dca299493b6be9fbcde4bf099514  tools/mx1_fire_guard.py
8b82429d7026960cebe29d0a4bfa57a3c5d78ade01ede95051ab35c9f1f5e4a4  tools/tests/test_mx1_fire_guard.py
e60856440dbe485956e7b4c207caac1d5836b06dbe5d5c078cbf51e8c1f4c853  .omx/research/ddm_mx1d_20260807/launch_ticket_v3_fire_guarded.json
378a1163368a1fd20c5da71354603d80421ecfac81be7fa45b79660f3d6d9dcf  .omx/research/ddm_mx1d_20260807/row1_v2_two_arm_ticket_result.json
e9a1844a366d77285a43792b55b08d04999393eedad3c2ef5716fac4346bd1ad  .omx/research/ddm_mx1d_20260807/row1_v2_two_arm/launch_arm_cap/n32_metal/fire_guard_verdict.json
```

## Boundaries

- No Metal training was run here; the sandbox has no accessible Metal device.
- No scorer job, archive build, full-n600 run, remote dispatch, or `upstream/evaluate.py` run was performed.
- The local guard verdict is intentionally failed because the keyed Metal mem-probe receipt is absent.
- Own-vehicle frontier unchanged: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
