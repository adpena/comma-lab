# Score-Lowering Continuation - 2026-05-08

## Scope

Operator directive: continue score lowering, stacking, CPU/CUDA exploit work,
mathematical optimal solving, and proof surfaces while preserving exact
evidence grades.

No remote/GPU/eval dispatch was launched from this note.

## Public Frontier Refresh

Command:

```bash
gh pr list --repo commaai/comma_video_compression_challenge \
  --state all --limit 20 \
  --json number,title,state,author,headRefOid,createdAt,updatedAt,mergedAt,url
```

Current top-of-list public state:

- PR108 `andimin01` remains open, updated `2026-05-05T16:33:07Z`.
  The only PR comment is the bot acknowledgement; no maintainer auth-eval
  score comment is present.
- PR108 body reports CPU self-eval `3.59`, archive `442,979` bytes, and the
  local intake artifact already records archive SHA-256
  `127b0b318ba2355cdac0d513f4027f0ca3297be4cba0f44e1ddb25cc70586804`.
- No newer public PR in the latest 20 changes the frontier queue.

Evidence grade: `external / byte_intake_only` for PR108. No rank promotion.

## Public Auth-Eval Comment Refresh

Command:

```bash
for pr in 101 102 103 105 106 107 108; do
  gh pr view "$pr" --repo commaai/comma_video_compression_challenge \
    --json comments \
    --jq '.comments[] | {author:.author.login, createdAt, body} |
      select(.body | test("Final score|Average PoseNet|Average SegNet|Submission file size|CPU|cuda|CUDA|Evaluation results"; "i"))'
done
```

Recomputed exact scores from public comment components:

| PR | Device | Bytes | seg | pose | recomputed score |
|---|---|---:|---:|---:|---:|
| PR101 | cuda | 178,258 | 0.00066304 | 0.00017103 | 0.226354458744 |
| PR101 | cpu | 178,258 | 0.00056023 | 0.00003286 | 0.192845012702 |
| PR102 | cuda | 178,981 | 0.00067565 | 0.00017347 | 0.228390831180 |
| PR102 | cpu | 178,981 | 0.00057599 | 0.00003460 | 0.195376176526 |
| PR103 | cuda | 178,223 | 0.00067623 | 0.00017198 | 0.227764851625 |
| PR103 | cpu | 178,223 | 0.00057654 | 0.00003443 | 0.194880702889 |
| PR105 | cuda | 177,857 | 0.00070456 | 0.00017267 | 0.230437255695 |
| PR105 | cpu | 177,857 | 0.00060913 | 0.00003472 | 0.197973979344 |
| PR106 | cuda | 186,239 | 0.00067142 | 0.00003351 | 0.209456642376 |
| PR107 | cuda | 178,392 | 0.00068841 | 0.00017394 | 0.229331025025 |

PR108 still has no maintainer auth-eval result comment.

Evidence caveat: these are public comment component recomputations. Exact
archive custody and local adjudicated JSON remain required before using any row
as a local exact-eval anchor.

## Arch Shrink Harvest

Existing paid lane `arch_shrink_x0.4_lightning` was checked, not duplicated.

- Job: `arch-shrink-x0-4-lightning-20260508T024304Z`
- SDK status: `stopped`
- Harvest attempt: `failed_artifact_rsync_rc_255`
- Root cause: Lightning SSH public-key rejection for `lightning-pact` against
  both live Studio and persisted job artifact roots.
- No archive, auth-eval JSON, score, or method evidence was harvested.

This is an artifact-harvest blocker, not a scientific result.

## A5 Frame-Conditional Bit-Budget Sweep

Command:

```bash
.venv/bin/python tools/pr101_frame_conditional_bit_anchor.py \
  --pr101-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --video-path upstream/videos/0.mkv \
  --etas 0.0 0.25 0.5 1.0 1.5 2.0 3.0 4.0 \
  --floor 0.25 \
  --cap 3.0 \
  --output-dir experiments/results/pr101_frame_conditional_bit_codex_20260508T211308Z
```

Artifact:

- Manifest:
  `experiments/results/pr101_frame_conditional_bit_codex_20260508T211308Z/build_manifest.json`
- Best row: `eta=4.0`, `floor=0.25`, `cap=3.0`
- Latent bytes: `15,387 -> 11,064` (`-4,323 B`)
- Side-channel overhead: `225 B`
- Archive-byte proxy: `178,158 -> 174,060` (`-4,098 B`)
- Per-pair bits range: `56.0 .. 672.0`
- Per-pair q-bits range: `2.0 .. 8.0`

Evidence semantics:

- `score_claim=false`
- `byte_proxy_only=true`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Blocking work before dispatch:

1. Per-pair score marginal evidence.
2. Runtime/inflate schema for per-pair bit-width side info.
3. Exact archive substitution that proves changed bytes are consumed.
4. Paired `[contest-CUDA]` and `[contest-CPU]` eval on identical archive bytes.

Delta versus prior A5 note:

- Prior Phase A table carried best A5 around `-1,264` to `-1,278 B`.
- The widened local sweep improves the byte proxy to `-4,098 B`, still
  non-promotable until the blockers above are closed.
