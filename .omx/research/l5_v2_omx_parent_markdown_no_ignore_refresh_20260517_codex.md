# L5 v2 OMX Parent Markdown No-Ignore Refresh - 2026-05-17

## Why This Follow-Up Exists

The operator noted that relevant OMX/Claude Markdown may sit outside
`.omx/research` and may also be hidden or ignored. The earlier parent-scope
refresh was correct about the active L5-v2 queue, but a reproducible
no-signal-loss scan must include ignored `.omx/auto_memory_snapshot_*` and
`.omx/tmp` Markdown as forensic inputs.

## Reproducible Scan

```bash
rg --files --hidden --no-ignore .omx --glob '*.md' --glob '!.omx/research/**'
rg -n -i --hidden --no-ignore \
  'l5|tt5l|time[-_ ]?trav|cargo[-_ ]?cult|rule #?6|rule6|local minima|local minimum|score[-_ ]?lower|no signal loss|stack|arithmetic|entropy|ball|nerv|hnerv|frontier' \
  .omx --glob '*.md' --glob '!.omx/research/**'
```

Observed non-research Markdown count with ignored files included:

| Bucket | Markdown files |
|---|---:|
| `.omx/auto_memory_snapshot_20260504T230223Z` | 562 |
| `.omx/context` | 28 |
| `.omx/interviews` | 1 |
| `.omx/plans` | 4 |
| `.omx/root` | 2 |
| `.omx/specs` | 1 |
| `.omx/state` | 22 |
| `.omx/tmp` | 16 |

## Active Authority Result

No ignored or historical parent-scope Markdown supersedes the May 17 active
queue in `.omx/state/current_focus.md` and `.omx/state/next_experiments.md`.

Current authority remains:

1. Rule #6 bolt-ons on verified A1 first.
2. L5-v2 / TT5L side-info effect curve and gate evidence in parallel.
3. High-risk per-pair-conditioning substrates stay behind SCORER-AWARENESS
   probes.
4. Existing Z3HV2 direct-residual exports remain historical controls, not the
   Balle implementation.
5. No provider dispatch without lane claim, source manifest, doctor/preflight
   evidence, and terminal claim plan.

## Carry-Forward Signal From Ignored Parent Markdown

These ignored auto-memory notes are not current score authority, but they are
useful control-plane signal:

- `.omx/auto_memory_snapshot_20260504T230223Z/feedback_no_signal_loss.md`
  says every experiment, score, council decision, and strategic correction must
  preserve provenance, axis labels, raw numbers, platform, cost, timestamp, and
  state updates.
- `.omx/auto_memory_snapshot_20260504T230223Z/project_codec_stacking_composition_canonical_orders_20260429.md`
  preserves the durable stack order:
  scorer-aware analysis -> representation -> prediction/transform ->
  water-fill/quantize/VQ -> hyperprior -> arithmetic -> archive packing.
  This supports Rule #6 A1 PR101-style entropy work but warns that arithmetic
  coding is terminal, not a substitute for symbols or context.
- `.omx/auto_memory_snapshot_20260504T230223Z/feedback_arithmetic_qint_codec_pr106_latents_unviable_brotli_already_below_entropy_20260504.md`
  is a concrete no-retread warning: plain zero-order arithmetic coding of
  HNeRV-like latent bytes can lose to Brotli because Brotli exploits LZ77 and
  context. Rule #6 entropy work should be context-aware or section-aware, not a
  generic arithmetic-coder rerun.
- `.omx/auto_memory_snapshot_20260504T230223Z/project_neural_compression_research_20260425.md`
  keeps Cool-Chic, C3, self-compression, LRConv-NeRV, PNVC, and improved
  overfitted-codec encoding in the longer non-HNeRV / non-local-basin frontier
  queue.
- `.omx/auto_memory_snapshot_20260504T230223Z/project_phases_2_3_4_design_implementation_math_provenance_20260429.md`
  keeps Joint ADMM, wavelet residuals, NeRV/Cool-Chic masks, bit-level archive
  optimization, and multi-pass compress optimization as long-burn candidates,
  but several assumptions there are pre-PR95 and must be reinterpreted through
  the later unique-and-complete-per-method rule.
- `.omx/auto_memory_snapshot_20260504T230223Z/feedback_git_reset_nukes_anchors_20260429.md`
  remains a live dispatch-hardening lesson: remote launchers must treat the
  shipped tarball/manifest as the runtime source of truth, not run destructive
  remote `git reset --hard` patterns that erase local-only anchors.
- `.omx/tmp/observability_section_append_20260516.md` is a reusable appendix
  template for design memos requiring the six-facet observability surface. It
  is not an L5-v2 score authority by itself.

## L5-v2 Consequence

The no-ignore scan does not justify a queue pivot. It does sharpen the first
Rule #6 implementation target:

- Do not run a generic zero-order arithmetic coder against A1 or PR101-style
  latent streams and call that frontier work.
- Profile section-conditioned entropy first, then choose Brotli/LZMA/Huffman/
  arithmetic/range/ANS per section with byte-consumption proof.
- Keep terminal arithmetic as a lowering pass after representation and symbol
  formation, not before.
- Preserve exact negative results and stale historical assumptions as control
  inputs, not as current dispatch authority.

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch was launched and no lane claim was opened by this scan.
