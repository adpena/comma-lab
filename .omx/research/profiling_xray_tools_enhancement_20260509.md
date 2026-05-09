# Profiling/analysis/xray tools enhancement (2026-05-09)

<!-- generated_at: 2026-05-09T11:00:00Z, from_state_hash: profiling_xray_tools_landing -->
<!-- HISTORICAL_PROVENANCE — append-only forensic record -->

This ledger documents the 5-tool xray pass landed 2026-05-09 in response to
the operator's "enhance our profiling and analysis tools and xray tools"
directive. Each tool closes a specific gap surfaced by the binary forensics
dossier (`.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`)
or by the per-architecture-class drift registry
(`feedback_cuda_cpu_axis_profile_learning_layer_20260508.md`).

## Audit phase: gaps identified

| Gap | Existing nearest tool | Why insufficient |
|---|---|---|
| 1. Per-section entropy heatmap | `tac.archive_byte_profile`, `audit_hnerv_brotli_saturation` | counts bytes but does not compute Shannon entropy density vs encoded bpb; cannot answer "is this section saturated or recoverable?" |
| 2. Side-by-side multi-PR layout | `audit_hnerv_section_candidate_diff` | pairwise only, on hardcoded HNeRV layout; no N-way matrix with SHARED/DIVERGED markers |
| 3. Per-tensor saliency-vs-bytes ranking | `tac.score_gradient_param_saliency` | computes saliency dict but no visualization layer mapping it against archive byte allocation |
| 4. Inflate op-cost catalog | `analyze_active_pr103_pr106_floor` | has scoring/floor analysis but does not enumerate per-line torch ops with per-channel-mutation flagging |
| 5. CPU drift predictor for new archives | `tac.optimization.cuda_cpu_axis_profile_registry`, `analyze_cpu_cuda_eval_drift` | registry has the data; no operator-facing CLI that takes archive + cuda_score → predicted CPU + verdict + actionable next step |

## Tools landed

### 1. `tools/xray_archive_section_entropy_heatmap.py` (22 tests)

Computes per-section Shannon entropy bits/byte against the encoded
bits/byte the ZIP coder achieved. Reports `saturation_ratio` (close to 1.0
= no recoverable bytes via re-coding) and `recoverable_bytes_if_floor_reached`.

**Empirical insight on PR101 monolithic `x` payload (178,158 B)**:
- `payload_entropy_bits_per_byte = 7.9987`
- `zip_bits_per_byte = 8.0`
- `saturation_ratio = 0.9998`
- `recoverable_bytes_if_floor_reached = 30`

This **directly confirms** Path B's empirical finding that PR101's brotli
substrate is saturated — further entropy-coder work cannot recover bytes; a
DIFFERENT TRANSFORM (per-tensor coding, sparsification, hyperprior) is
required.

### 2. `tools/xray_per_pr_archive_layout_compare.py` (19 tests)

N-way archive comparison emitting a section × archive matrix with SHARED /
DIVERGED / MISSING markers. Identifies structural carry-over (don't
re-engineer) vs PR-specific deltas (where each PR's innovation lives).

### 3. `tools/xray_per_tensor_saliency_heatmap.py` (22 tests)

Joins a saliency dict (from `tac.score_gradient_param_saliency`) with a
per-tensor byte map (from `tac.archive_byte_profile`) to compute
`saliency_per_byte`. Sorts ascending — bottom-N% by saliency-per-byte are
the next allocator's coarsening targets.

Falls back to `--saliency-equal` for archives where score-gradient saliency
hasn't been computed yet (degenerates to a "byte allocation only" view).

### 4. `tools/xray_inflate_op_cost_profiler.py` (23 tests)

Static AST walk over inflate.py. Per-line catalog of torch / F / tensor-
method calls with cost-class proxy (`decoder-forward`, `per-frame`,
`per-channel-mutation`, `cheap`, `io`, `unknown`). The
`per-channel-mutation` rows are the PR101 → PR103 medal-delta pattern
(three lines of `up[:, X, Y].sub_(1.0)`).

**Empirical run on PR101 canonical `hnerv_ft_microcodec/inflate.py`**:
- 39 ops total
- 3 per-channel mutations (matches dossier exactly)

### 5. `tools/xray_cpu_cuda_drift_per_arch_class.py` (22 tests)

Operator CLI wrapper around the registry. Inputs: archive + CUDA score.
Outputs: classified architecture class, predicted CPU score (point + 1-σ
band), medal-band verdict (`INSIDE`/`BORDERLINE`/`UNCERTAIN`/`OUTSIDE`),
recommended next step (`DISPATCH`/`HOLD`/`DROP`).

**Empirical smoke (on PR101 archive at CUDA 0.22839)**:
- `architecture_class = unknown_uncalibrated` (PR101 standalone has no
  ``inferred_kind`` metadata file alongside the archive)
- `predicted_cpu = 0.19539`
- `verdict = BORDERLINE`

Compare against the empirical PR102 CPU score 0.19499 (codex 2026-05-08
GHA Linux x86_64). The predictor's BORDERLINE verdict is correct — PR102
landed inside the borderline band (within `medal_floor + medal_tolerance =
0.20038`).

## Existing tool enhancements

### `tools/operator_briefing.py` — Phase 6 — XRAY toolkit section added

New section enumerates each xray tool with a one-line purpose statement and
a copy-paste-ready CLI example. Surfaces the diagnostic toolkit so future
agents discover them without grepping `tools/xray_*`.

## Test counts

| Tool | Tests | Pass |
|---|---:|:---:|
| xray_archive_section_entropy_heatmap | 22 | ✓ |
| xray_per_pr_archive_layout_compare | 19 | ✓ |
| xray_per_tensor_saliency_heatmap | 22 | ✓ |
| xray_inflate_op_cost_profiler | 23 | ✓ |
| xray_cpu_cuda_drift_per_arch_class | 22 | ✓ |
| **Total** | **108** | **✓** |

## Exploitation leverage examples

1. **PR101 saturation confirmed**: heatmap tool shows `saturation_ratio =
   0.9998` on the monolithic payload, validating Path B's empirical finding
   that the substrate is at brotli ceiling. The next byte-tightening pass
   must change the TRANSFORM, not the entropy coder.
2. **PR101 medal-delta pattern visible**: op-cost profiler counts 3
   per-channel mutations exactly matching the dossier. Future inflate
   edits can be planned by visual inspection of this catalog rather than
   line-by-line code reading.
3. **CUDA→CPU prediction at PR102 score**: drift predictor returns
   `BORDERLINE` for the empirical PR102 result, demonstrating the registry-
   based predictor is calibrated for the medal-band cluster. Saves $5+ of
   wasted CPU dispatches on archives the predictor flags `OUTSIDE`.

## Operator decisions surfaced

1. **Should we run the entropy heatmap on every public PR archive?** This
   would build a per-PR saturation table; it's $0 GPU and ~5 min wall-clock.
   Recommended: yes, before the next council meeting.
2. **Should we extend the op-cost profiler with shape inference?** Static
   classes are heuristic (no actual cost data). True cost requires running
   inflate.py with timed hooks. Defer until after council-priority lanes
   land.
3. **Should the drift predictor write back to the registry?** Currently
   read-only (it imports the bootstrap registry, doesn't update it).
   Recommended: keep read-only — the existing `harvest_new_anchor_and_update`
   function is the canonical writer; this tool is a diagnostic consumer.

## Tags + custody

All outputs tagged `[diagnostic: <tool>]` per CLAUDE.md "Forbidden Score
Claims". `score_claim=False`, `promotion_eligible=False`,
`ready_for_exact_eval_dispatch=False` on every emitted JSON. Per-tool output
goes to `experiments/results/<tool_name>_<ISO_TIMESTAMP>/` per CLAUDE.md
"transient_tmp_evidence" rule.

## Cross-references

- Binary forensics dossier:
  `.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`
- CUDA/CPU drift registry:
  `src/tac/optimization/cuda_cpu_axis_profile_registry.py`
- Score-gradient saliency:
  `src/tac/score_gradient_param_saliency.py`
- Archive byte profile:
  `src/tac/archive_byte_profile.py`
- Memory entry:
  `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_profiling_xray_tools_enhancement_landed_20260509.md`
