# OMX parent Markdown current verification - 2026-05-17

## Scope

Operator concern: relevant cargo-cult, L5/L5-v2, TT5L, Rule #6, PR101/FEC6,
Quantizr, entropy, PoseNet/SegNet, or no-signal-loss instructions may sit
outside `.omx/research`.

This pass rechecked the current no-ignore parent Markdown surface after the
prior parent-scope ledgers. It is a verification/update ledger, not a score
claim, dispatch claim, promotion, or architecture-lock authority.

## Commands

```bash
rg --files --hidden --no-ignore .omx -g '*.md' -g '!.omx/research/**'
rg --files --hidden --no-ignore .omx/research -g '*.md'
rg --hidden --no-ignore -l \
  'cargo|cult|assumption|local minima|local minimum|L5|TT5L|time[-_ ]?trav|staircase|Rule #6|rule6|FEC6|PR101|PR95|Quantizr|arithmetic|entropy|PoseNet|SegNet|frontier' \
  .omx -g '*.md' -g '!.omx/research/**'
```

## Current inventory

| Surface | Markdown files |
|---|---:|
| `.omx/research` | 1770 |
| `.omx/auto_memory_snapshot_*` | 562 |
| `.omx/context` | 28 |
| `.omx/state` | 22 |
| `.omx/tmp` | 16 |
| `.omx/plans` | 4 |
| `.omx/specs` | 1 |
| `.omx/interviews` | 1 |
| `.omx` root files | 2 |

Current total observed Markdown under `.omx` is `2406`; current observed
non-research Markdown is `636`.

The older full-scope ledger reported `2399` total and `1763` research files.
The delta is expected: new `.omx/research/*_20260517*.md` ledgers have landed
since that scan. The non-research count remains `636`.

## Authority result

No non-research `.omx` Markdown supersedes the active May 17 authority chain:

1. `.omx/state/current_focus.md`
2. `.omx/state/next_experiments.md`
3. `.omx/state/active_lane_dispatch_claims.md`
4. dated `.omx/research/*_20260517*.md` ledgers

`.omx/notepad.md` remains historical April Track-B/AV1 memory. It is useful
for no-signal-loss provenance around film grain, color/range, one-axis local
knees, and scorer sensitivity, but it explicitly is not current L5, TT5L,
FEC6, Rule #6, PR101, or submission authority.

`.omx/release_manifest_v0.2.0-rc1.md` remains release hygiene context, not a
current score or dispatch authority.

The ignored `.omx/auto_memory_snapshot_20260504T230223Z` and `.omx/tmp`
Markdown remain forensic inputs. They preserve useful failure-mode memory, but
they do not become implicit source of truth.

## Current routing consequences

The parent Markdown scan reinforces, rather than changes, the current route:

1. Keep L5-v2/TT5L P0, but do not promote rate-only side-info claims beyond
   the closed byte-bound unless a packet proves PoseNet/SegNet component
   movement.
2. Keep Rule #6 A1/FEC6 work focused on byte-closed, inflate-consumed packets
   with component rows; do not retread selector byte polish below the charged
   byte threshold.
3. Treat film grain, color/range, ROI allocation, half-frame contracts, and
   broad preprocessing as scorer contracts, not cosmetic transforms. Require
   train/inflate parity or direct component-response proof.
4. Treat generic arithmetic-coder substitutions as suspect until the stream is
   section-conditioned and the coded bytes are consumed by inflate.
5. Treat Quantizr 5-stage staircase as a training scaffold with
   `score_claim=false`; score movement requires real trainer adoption,
   transition records, consumed archive bytes, and CPU/CUDA component evidence.

## Related ledgers

- `.omx/research/l5_v2_omx_parent_markdown_scope_refresh_20260517_codex.md`
- `.omx/research/l5_v2_omx_parent_markdown_no_ignore_refresh_20260517_codex.md`
- `.omx/research/omx_parent_markdown_full_scope_followup_20260517_codex.md`
- `.omx/research/omx_parent_markdown_cargo_cult_and_quantizr_staircase_review_20260517_codex.md`
- `.omx/research/omx_parent_markdown_modal_cpu_dispatch_bugfix_20260517_codex.md`

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch was launched and no lane claim was opened by this scan.
