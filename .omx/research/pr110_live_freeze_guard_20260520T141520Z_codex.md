# PR #110 Live Freeze Guard

**UTC:** 2026-05-20T14:15:20Z
**PR:** https://github.com/commaai/comma_video_compression_challenge/pull/110
**Frozen head:** `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`
**Head repo/ref:** `adpena/comma_video_compression_challenge:hnerv_fec6_fixed_huffman_k16`
**Base repo/ref:** `commaai/comma_video_compression_challenge:master`

## Verdict

Local development may continue in `/Users/adpena/Projects/pact` / `adpena/comma-lab`
without touching PR #110, provided the boundary below is honored.

## Verified Safety Boundary

- Current working tree remote is `git@github.com:adpena/comma-lab.git`.
- Current working tree branch is `main`, not the PR #110 fork branch.
- No `/Users/adpena/Projects/*` checkout of `comma_video_compression_challenge`
  was found during this pass.
- Live PR #110 is open and mergeable-clean.
- Live PR #110 head is pinned at
  `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`.
- Live PR #110 runtime files are under
  `submissions/hnerv_fec6_fixed_huffman_k16/`, not repo root.

## Freeze Rules

1. Do not run `git push` to `adpena/comma_video_compression_challenge` or
   `commaai/comma_video_compression_challenge` from this session.
2. Do not run `gh pr edit 110`, `gh pr comment 110`, or modify the PR body
   unless a factual blocker is found and explicitly routed as a freeze-break.
3. Do not check out or mutate the PR #110 branch as a development workspace.
   Use detached read-only clones under `/tmp` only for verification.
4. Keep frontier development outputs in new `experiments/results/...` directories
   and dated `.omx/research/...` ledgers.
5. If a PR #110 factual fix becomes necessary, first write a new
   `.omx/research/pr110_freeze_break_<utc>_codex.md` memo with: reason,
   exact intended public change, verification command, and rollback plan.

## Continue-Local-Development Contract

Allowed now:

- Local candidate materialization under new experiment result directories.
- Non-public research/audit memos under `.omx/research/`.
- Local `README.md` / manifest mirror corrections that point back to the frozen
  live PR head, without pushing to the PR branch.
- Evidence-pack creation and transitive documentation audits.

Not allowed without freeze-break:

- Public PR branch commits.
- Public PR body edits.
- Release asset replacement.
- Any force-push or destructive git operation.

## Verification Commands Run

```bash
git remote -v
git branch --show-current
git rev-parse HEAD
gh api repos/commaai/comma_video_compression_challenge/pulls/110 \
  --jq '{state,mergeable,mergeable_state,head:{repo:.head.repo.full_name,ref:.head.ref,sha:.head.sha},base:{repo:.base.repo.full_name,ref:.base.ref},updated_at,html_url}'
gh api 'repos/adpena/comma_video_compression_challenge/git/trees/ec6cc7f98c16b6ad2db8bc7cde65757bb7993004?recursive=1' \
  --jq '.tree[].path' | rg '^(README.md|archive.zip|inflate\.py|inflate\.sh|src/|submissions/hnerv_fec6_fixed_huffman_k16)'
find /Users/adpena/Projects -maxdepth 3 -type d -name .git -print \
  | sed 's#/.git##' \
  | rg 'comma_video_compression_challenge|comma-lab|pact$'
```

## Observed Live PR State

```json
{
  "state": "open",
  "mergeable": true,
  "mergeable_state": "clean",
  "head_repo": "adpena/comma_video_compression_challenge",
  "head_ref": "hnerv_fec6_fixed_huffman_k16",
  "head_sha": "ec6cc7f98c16b6ad2db8bc7cde65757bb7993004",
  "base_repo": "commaai/comma_video_compression_challenge",
  "base_ref": "master",
  "updated_at": "2026-05-20T14:00:48Z"
}
```

## Observed Runtime Layout

```text
README.md
submissions/hnerv_fec6_fixed_huffman_k16
submissions/hnerv_fec6_fixed_huffman_k16/inflate.py
submissions/hnerv_fec6_fixed_huffman_k16/inflate.sh
submissions/hnerv_fec6_fixed_huffman_k16/src
submissions/hnerv_fec6_fixed_huffman_k16/src/codec.py
submissions/hnerv_fec6_fixed_huffman_k16/src/codec_sidecar.py
submissions/hnerv_fec6_fixed_huffman_k16/src/frame_selector.py
submissions/hnerv_fec6_fixed_huffman_k16/src/model.py
```
