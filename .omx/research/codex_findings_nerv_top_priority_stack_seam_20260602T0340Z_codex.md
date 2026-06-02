# Codex Findings: NeRV Top-Priority Stack Seam

UTC: 2026-06-02T03:40Z
Lane: lane_nerv_top_priority_stack_seam_20260602
Artifact: .omx/research/nerv_top_priority_stack_seam_20260602T0340Z.json
Axis: [planning/control]

## Verdict

GO for local SNeRV + HiNeRV stack optimization.

NO-GO for exact eval, promotion, score claim, rank/kill, or PR95-beat claim.

The contract preserves the current operating policy as machine-readable state:
SNeRV and HiNeRV are the two top-priority carrier stacks; PR95/HNeRV
`hnerv_muon` is the upstream baseline/control to beat; SR-NeRV, RNeRV,
FFNeRV-flow, and BoostNeRV are enhancers of the winner, not separate priority
carrier stacks.

## PR95 Baseline Custody

Live upstream PR metadata was checked through GitHub CLI:

- PR: https://github.com/commaai/comma_video_compression_challenge/pull/95
- Title: `hnerv_muon submission (0.20)`
- State: `MERGED`
- Head ref: `add-hnerv-muon`
- Head SHA: `9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9`

The local PR95 public intake cache was hashed in the artifact:

- Archive: `/Users/adpena/Projects/pact/experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/archive.zip`
- Archive bytes: `178417`
- Archive SHA-256: `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- Runtime files hashed: `inflate.sh`, `inflate.py`, `src/model.py`,
  `src/codec.py`, `src/optim.py`, `src/stages/stage8_muon_finetune.py`

This is baseline custody, not a fresh score claim. The contract requires a
same-axis PR95 control replay before any SNeRV/HiNeRV winner can claim to beat
PR95.

## Encoded Stack Direction

SNeRV:

- Continue pair-robust scorer-loop or NES decoder-QAT with PoseNet as a hard
  guard.
- Preserve receiver-decoded mixed precision bytes and explicit decoder mode
  assignment evidence.
- Do not use closed-form scalar/component HF sweeps as promotion evidence.

HiNeRV:

- Continue longer real-teacher SegNet/PoseNet training with coder-aware QAT and
  joint P18/P19 weighting.
- Keep full-video MLX prefilter before local CPU replay or exact spend.
- Treat the latest full-600 MLX prefilter as a demotion of that configuration,
  not a kill of HiNeRV as a family.

Enhancers:

- SR-NeRV remains the highest-priority enhancer only as trained/scorer-aware SR.
- RNeRV is the winner's per-video config optimizer.
- FFNeRV-flow is a pose-channel enhancer.
- BoostNeRV is a conditional-decoder/temporal-affine bolt-on.

## Dispatch State

The artifact blocks new exact/full-video launches because the active claim table
still contains current nonterminal blockers:

- `pr101_cpu_recovery_pending_blocks_new_exact_or_full_video`
- `z5_rao_ballard_modal_claims_still_need_terminal_adjudication`
- current-window generic exact/full-video blockers for active PACT/VQ full600
  lanes

Allowed now:

- Poll PR101 CPU through the canonical recovery tool and terminalize only if it
  yields a terminal artifact.
- Run local advisory SNeRV/HiNeRV stack optimization and receiver-proof work.

Not allowed now:

- New full-video/exact/CUDA launches.
- Promotion or submission.
- PR95-beat claims without same-axis PR95 control.

## Verification

```bash
/Users/adpena/Projects/pact/.venv/bin/ruff check \
  src/tac/analysis/nerv_top_priority_stack_seam.py \
  tools/build_nerv_top_priority_stack_seam.py \
  src/tac/tests/test_nerv_top_priority_stack_seam.py

/Users/adpena/Projects/pact/.venv/bin/python -m pytest \
  --import-mode=importlib -q \
  src/tac/tests/test_nerv_top_priority_stack_seam.py
```

Both passed.

Fresh-eyes xhigh sidecar review then found three issues before commit:

- the first blocker parser was too bespoke for PR101/Z5 and could miss a
  different active exact/full-video claim;
- a non-git upstream directory did not fail closed as PR95 custody;
- CLI defaults crossed from the SSD worktree back into the shared checkout.

The landed version fixes all three: generic blockers are parsed from the claim
table with a current active window, PR95 upstream git head/remote are mandatory
custody fields, and CLI defaults are repo-root relative unless the operator
passes external custody paths explicitly.

## Next Step

Use `.omx/research/nerv_top_priority_stack_seam_20260602T0340Z.json` as the
queue-owned control input for the next SNeRV/HiNeRV tranche: local stack
optimization first, PR101/Z5 terminalization before any new exact/full-video
dispatch, and PR95 same-axis replay as the mandatory baseline gate.
