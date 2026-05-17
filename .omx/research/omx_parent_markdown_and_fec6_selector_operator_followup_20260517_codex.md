# OMX Parent Markdown And FEC6 Selector Operator Follow-Up - 2026-05-17

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch was launched and no lane claim was opened by this follow-up.

## Parent OMX Markdown Confirmation

The operator noted that useful Claude/OMX signal may sit outside
`.omx/research`. I refreshed the parent-scope inventory with:

```bash
find .omx -path .omx/research -prune -o -name '*.md' -print
```

Observed non-research Markdown count: `636`.

The existing no-ignore refresh remains the active broad scan ledger:

- `.omx/research/l5_v2_omx_parent_markdown_no_ignore_refresh_20260517_codex.md`

Current authority still resolves to:

1. Rule #6 bolt-ons on verified A1/FEC6 CPU-frontier anchors.
2. L5-v2 / TT5L side-info effect curve in parallel.
3. High-risk per-pair-conditioning substrates behind SCORER-AWARENESS probes.
4. PR106/HNeRV local-basin work as forensic donor/control only, not P0 retread.

## Extra Parent-Scope Signal Carried Forward

These parent-scope auto-memory files are not score authority, but they are now
explicitly carried into the active work order so future sessions do not retread
old traps.

| Source | Carried-forward rule |
|---|---|
| `.omx/auto_memory_snapshot_20260504T230223Z/feedback_no_signal_loss.md` | Preserve platform, hardware, cost, timestamp, config, raw numbers, score axis labels, and state updates for every material result. |
| `.omx/auto_memory_snapshot_20260504T230223Z/project_codec_stacking_composition_canonical_orders_20260429.md` | Keep codec order as scorer-aware analysis -> representation -> prediction/transform -> quantize/VQ -> hyperprior -> arithmetic -> archive. Arithmetic is terminal. |
| `.omx/auto_memory_snapshot_20260504T230223Z/feedback_arithmetic_qint_codec_pr106_latents_unviable_brotli_already_below_entropy_20260504.md` | Do not rerun generic zero-order arithmetic on HNeRV-like latents; Brotli already beat that grain. Use section-aware/context-aware symbols with byte-consumption proof. |
| `.omx/auto_memory_snapshot_20260504T230223Z/feedback_overfit_is_the_goal.md` | This contest is one 600-pair video; per-pair, per-frame, scorer-aware optimization is valid when contest compliant. |
| `.omx/auto_memory_snapshot_20260504T230223Z/feedback_posenet_sensitivity.md` | PoseNet uses whole-frame signal; blur/downsample/cliff changes must be judged against pose sensitivity, not just SegNet. |
| `.omx/auto_memory_snapshot_20260504T230223Z/feedback_film_eval_no_poses_critical.md` | FiLM/ego-motion variants without real pose plumbing are not meaningful evaluations. |
| `.omx/auto_memory_snapshot_20260504T230223Z/project_arbitrary_vs_learnable_taxonomy.md` | Prefer derived, swept, or learned parameters over arbitrary constants; if a choice is defensible in multiple ways, probe both. |

## FEC6 Selector Operator-Space Result

New reusable code:

- `src/tac/fec6_selector_operator_space.py`
- `tools/audit_fec6_selector_operator_space.py`
- `src/tac/tests/test_fec6_selector_operator_space.py`

Generated local artifact:

- `experiments/results/fec6_selector_operator_space_20260517_codex/operator_space_manifest.json`
- `experiments/results/fec6_selector_operator_space_20260517_codex/operator_space.md`

Source archive:

- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- bytes: `178517`
- sha256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- current score: `0.1920513168811056` `[contest-CPU; GHA Linux x86_64 1:1]`
- target: `<0.192`

Key manifest facts:

- required charged bytes to strictly cross `0.192` with unchanged components:
  `78`
- selector payload bytes: `249`
- selector index bytes: `243`
- selector fixed-Huffman code bits: `1944`
- selector zero-header entropy floor: `241` bytes
- payload gap to zero-header entropy floor: `8` bytes
- grammar-aware operator rows emitted: `40`
- raw archive byte rows emitted: `0`
- proxy-improving and nonpositive-bit rows: `0`

## Decision

The same-runtime FEC6 selector byte-only retread is blocked for the current
evidence set. The selector has only an 8-byte zero-header entropy gap, while
crossing `<0.192` with unchanged components requires at least 78 charged bytes.
The available pair-component proxy table produced no selector substitution
that is both proxy-improving and bit-nonpositive.

This does not falsify FEC6 or Rule #6. It narrows the next valid object:

1. A new CPU/CUDA component table with stronger per-pair selector evidence, or
2. A byte-different, grammar-valid packet operator that moves components and
   proves runtime consumption, or
3. A larger L5/Rule #6 side-info stream where the byte budget can actually
   cross the score threshold.

Do not spend another turn on raw FEC6 archive-byte gradients or same-runtime
selector polishing from the current rows. The valid next score-lowering path is
component-moving operator design, paired component measurement, or TT5L/L5-v2
side-info harvest.

## Verification

Completed before this ledger:

```bash
.venv/bin/python -m pytest src/tac/tests/test_fec6_selector_operator_space.py
.venv/bin/python -m ruff check src/tac/fec6_selector_operator_space.py tools/audit_fec6_selector_operator_space.py src/tac/tests/test_fec6_selector_operator_space.py
.venv/bin/python tools/audit_fec6_selector_operator_space.py
```

The broader verification suite is run separately before commit.
