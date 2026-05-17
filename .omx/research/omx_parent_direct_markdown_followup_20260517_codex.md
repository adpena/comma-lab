# OMX Parent Direct Markdown Follow-Up - 2026-05-17

## Trigger

The operator clarified that relevant signal may sit outside `.omx/research`
and asked to check all Markdown documents in the `.omx` parent directory.

## Recheck

Commands rerun from current worktree:

```bash
find .omx -name '*.md' -print | wc -l
find .omx -path .omx/research -prune -o -name '*.md' -print | wc -l
find .omx -maxdepth 1 -type f -name '*.md' -print
sed -n '1,220p' .omx/notepad.md
sed -n '1,220p' .omx/release_manifest_v0.2.0-rc1.md
```

Observed counts:

- Total `.omx/**/*.md`: `2413` in the current dirty worktree.
- Non-research `.omx` Markdown: `636`.
- Direct root `.omx/*.md`: exactly two files:
  `.omx/notepad.md` and `.omx/release_manifest_v0.2.0-rc1.md`.

## Finding

Neither direct root Markdown file supersedes the active May 17 L5-v2 / Rule #6
control plane:

- `.omx/notepad.md` explicitly marks itself as stale April Track-B/AV1 working
  memory and says it is not current L5, TT5L, FEC6, Rule #6, PR101, or
  submission authority.
- `.omx/release_manifest_v0.2.0-rc1.md` is release hygiene and historical
  release-candidate context. It preserves axis-label and public-disclosure
  discipline, but it is not a current dispatch, score, or architecture-lock
  authority.

Active non-research authority remains:

1. `.omx/state/current_focus.md`
2. `.omx/state/next_experiments.md`
3. `.omx/state/active_lane_dispatch_claims.md`

## Carried Signal

The direct parent files still preserve useful non-authoritative lessons:

- April AV1 film-grain, color/range, smoke-gate, and scorer-backed rejection
  history remain valid as old scorer-sensitivity examples.
- Release hygiene keeps axis labeling, public-disclosure hygiene, and
  non-promotion of macOS/MPS/proxy results as release-grade discipline.

## Action

No queue pivot and no provider dispatch were justified by the direct parent
Markdown recheck. The concrete hardening action remains the current local
research-signal custody patch: local MPS and macOS-CPU advisory dispatches must
fail closed if their canonical manifest row cannot be written.

Authority flags:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
