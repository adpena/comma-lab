# PR #110 Freeze-Break Candidate

**UTC:** 2026-05-20T14:21:55Z
**PR:** https://github.com/commaai/comma_video_compression_challenge/pull/110
**Frozen head:** `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`
**Status:** candidate only; no live PR edit performed

## Why This Exists

The freeze guard says no live PR edits unless a factual blocker is found and
routed first. The transitive documentation audit plus xhigh public-surface
review surfaced several public-facing issues that are real enough to pre-route:

1. The PR body says the runtime tree is staged under
   `submissions/hnerv_fec6_fixed_huffman_k16` "alongside `archive.zip`", but
   the PR head tree has runtime files only. `archive.zip` is release-hosted.
2. The PR body reports CPU/CUDA score facts, but the public PR/release surfaces
   do not include `report.txt` or inline component values.
3. The PR body says PR #95 / #98 / #100 / #101 / #102 / #103 / PR #110
   "cluster within ~0.0008 on the CPU axis." That is false as written: the
   ~0.0008 delta applies to PR #101 -> PR #110, while the broader public HNeRV
   lineage spans roughly `0.1928`-`0.1987` on reported CPU values.
4. The PR-linked comma-lab docs transitively expose Claude/Codex/persona
   methodology and stale live-count claims. This is tone/source-control risk,
   not a packet-runtime defect.
5. The release body still has stale smoke instructions and "companion PR will
   be opened" wording. This is a release-text issue, not an archive asset issue.

## Freeze-Break Recommendation

Do **not** automatically break freeze from this memo. Continue local development
against new experiment artifacts. Break freeze only if the operator chooses to
polish public posture before maintainer review, or if maintainer/bot feedback
points at one of these issues.

If freeze is broken, use a PR-body-only edit first. Do not push branch commits,
replace release assets, or mutate the archived packet.

## Minimal Intended Public Change

Recommended PR-body-only patch:

1. Replace:

   ```text
   Runtime tree (...) is staged directly in this PR under [...] at head
   ec6cc7f..., alongside archive.zip per the upstream contract.
   ```

   with:

   ```text
   Runtime tree (...) is staged directly in this PR under [...] at head
   ec6cc7f.... The rate-charged archive.zip is hosted at the GitHub Release URL
   above; it is not embedded in the PR tree.
   ```

2. Replace the report section with the upstream report block and component
   recomputation:

   ```text
   CPU component recomputation: PoseNet distortion 0.00002943, SegNet
   distortion 0.00056029, rate 0.004754685709380427; score components
   0.017155174146594957 + 0.056028999999999995 +
   0.11886714273451067 = 0.1920513168811056 [contest-CPU].

   Paired CUDA/T4 component recomputation: PoseNet distortion 0.00016846,
   SegNet distortion 0.00066299, same rate 0.004754685709380427; score
   components 0.0410438789589873 + 0.066299 + 0.11886714273451067 =
   0.22621002169349796 [contest-CUDA T4].
   ```

3. Replace the broad HNeRV-cluster sentence with:

   ```text
   This packet improves the top HNeRV-family CPU anchor I used as the immediate
   byte substrate: PR #101 `0.192845 [contest-CPU]` -> this packet
   `0.192051 [contest-CPU]` (`-0.000794`, including the +259-byte rate cost).
   The broader public HNeRV lineage (#95/#98/#100/#102/#103) spans roughly
   `0.1928`-`0.1987` on reported CPU values, so I treat the tight local floor as
   the PR #101/PR #110 neighborhood rather than the full lineage.
   ```

4. Replace `num_threads: 2, matching the ubuntu-latest GHA runner family` with
   the factual-only wording `report.txt records upstream evaluator
   num_threads: 2`.

5. Either remove the two deep comma-lab doc links from the PR body or replace
   them with one sanitized public overview link after a new comma-lab commit is
   created. Do not point PR #110 at docs containing Claude/Codex/persona
   methodology from a first-level public path.

6. If release text is edited, update only the release body text. Do not replace
   `archive.zip`. The archive asset SHA/size are correct.

## Required Verification Before Any Live Edit

Run:

```bash
gh api repos/commaai/comma_video_compression_challenge/pulls/110 \
  --jq '{head:{repo:.head.repo.full_name,ref:.head.ref,sha:.head.sha},updated_at,body:.body}'

gh api 'repos/adpena/comma_video_compression_challenge/git/trees/ec6cc7f98c16b6ad2db8bc7cde65757bb7993004?recursive=1' \
  --jq '.tree[].path' | rg '^(README.md|archive.zip|submissions/hnerv_fec6_fixed_huffman_k16)'

shasum -a 256 \
  experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip
wc -c \
  experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip
```

Also re-read:

```text
.omx/research/pr110_final_evidence_pack_20260520T141144Z_codex/README.md
.omx/research/pr110_transitive_doc_audit_20260520T141423Z_codex.md
```

## Rollback Plan

- Save the pre-edit PR body from `gh api ... --jq .body` to a dated local memo.
- Apply only `gh pr edit 110 --body-file <body.md>` if operator approves.
- If the edit is wrong, restore the saved body with the same command.
- No branch push, no force-push, no release replacement.

## Current Action

No live PR mutation was performed. Local development may continue under the
freeze guard.
