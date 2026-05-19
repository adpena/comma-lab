# Codex session summary - PR95 auth-eval bridge

**UTC:** 2026-05-19T16:57:32Z  
**Primary artifact:** `tools/run_pr95_local_training_probe.py`

## Completed

- Added opt-in `--run-auth-eval` to the PR95 local training probe.
- Preserved authority boundaries: local macOS CPU/MPS bridge outputs are
  `score_claim=false`, `promotion_eligible=false`, and `rank_or_kill_eligible=false`.
- Added tests for bridge command construction, durable `--work-dir` usage,
  venv PATH preservation, advisory evidence flags, and missing-archive
  fail-closed behavior.
- Ran a real integrated local MPS -> archive -> macOS CPU auth-eval smoke.

## Evidence

```text
artifact: experiments/results/pr95_local_mps_integrated_auth_bridge_smoke_20260519T175800Z/manifest.json
training_best_score: 83.40266892376289
auth_eval_canonical_score: 83.40271170242858
absolute_score_delta: 0.00004277866568713762
archive_sha256: 17523537254adf179825294451b8e4a4ac75d0ad6e1c40078b6ba98f4bc160aa
auth_eval_json_sha256: 66019cb18f855eecbbf574d487c1df97dde496ecba6ca9058de97e2e41314173
```

## Parallel audit inputs consumed

- Z7-Mamba 600-pair handoff readiness is green, but current Z7 exact-eval
  scores remain around 90 on both axes and are not promotable. Highest-EV next
  Z7 action is a positive-control export/runtime-geometry check through the same
  exact handoff path.
- Current Codex queue audit ranks BUILD_1 HF Jobs SegNet surrogate Phase 1 as
  the best ready/high-value pending task, provided the active HF Jobs dispatch
  protocol WIP is stable before touching dispatch surfaces.

## Next Codex action

After this bridge lands, resume the canonical queue with BUILD_1 readiness:
verify HF Jobs partner WIP stability, run dispatcher dry-run against the current
CLI surface, then claim/launch only if the current recipe and dispatch protocol
are still authority-clean.
