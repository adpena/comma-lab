<!-- SPDX-License-Identifier: MIT -->
# master_gradient_xray — operator usage

`tools/master_gradient_xray.py` is the canonical operator-facing
visualization for the master-gradient anchor ledger
(`.omx/state/master_gradient_anchors.jsonl`). It is the observability
surface declared by lane `lane_master_gradient_xray_viz_tool_20260519`
under CLAUDE.md "Max observability — non-negotiable" + Catalog #305
(observability surface declaration) + Catalog #323 (canonical Provenance).

## Position in the stack

Producers (writers of the anchor ledger):
- `tools/extract_master_gradient.py` — canonical per-archive autograd
  extraction CLI.
- Substrate trainers that emit per-archive gradient anchors as a byproduct.

Consumers of the ledger (cathedral autopilot, Pareto solver, bit allocator):
- `tac.master_gradient_consumers` — canonical Python reader API.
- `tac.cathedral_consumers.*` — 21 sister cathedral consumers per Catalog
  #335 auto-discovery (see `feedback_master_gradient_consumer_cathedral_wire_in_landed_20260519.md`).

This tool is the **operator-facing observability surface** between them —
it turns the producer's gradient tensors into 5 canonical plots so the
operator can decide which downstream consumer route to invoke for which
archive.

## Quick start (5 patterns)

### 1. Full xray of one archive into a directory

```bash
.venv/bin/python tools/master_gradient_xray.py \
    --archive-sha 87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5 \
    --output-dir reports/master_gradient_xray/a1_baseline/
```

Emits 5 canonical plots + sister JSON sidecars + `index.html` landing page.
Open `index.html` in a browser for the operator-friendly view.

### 2. Single plot to a single file

```bash
.venv/bin/python tools/master_gradient_xray.py \
    --plot per_byte_heatmap \
    --archive-sha 87ec7ca5f2f328a8... \
    --output reports/x/heatmap.png
```

### 3. Cross-substrate correlation across multiple anchors

```bash
.venv/bin/python tools/master_gradient_xray.py \
    --plot cross_substrate_correlation \
    --archive-sha 87ec7ca5... \
    --archive-sha 6bae0201... \
    --archive-sha 9cb989ce... \
    --output reports/x/cross_substrate.png
```

The correlation matrix uses cosine similarity on per-byte L1 magnitude
profiles. Off-diagonal cells in the range `[-1, 1]`: positive = substrates
share leverage structure; negative = substrates have opposite leverage
profiles (Wyner-Ziv complementary candidates per the canonical Catalog
#319 deliverability classification).

### 4. Drift vs sensitivity cross-reference (Phase B follow-up)

```bash
.venv/bin/python tools/master_gradient_xray.py \
    --archive-sha 87ec7ca5... \
    --mps-drift-json .omx/state/mps_drift_granular_20260519T122700Z.json \
    --output-dir reports/master_gradient_xray/a1_with_drift/
```

When `--mps-drift-json` is provided, the 6th plot
`drift_vs_sensitivity_scatter` is emitted alongside the canonical 5. Per
the operator's MPS Phase B work
(`feedback_mps_phase_b_options_b_plus_c_completion_landed_20260519.md`)
the scatter quadrant analysis answers: "is MPS drift in the score-relevant
subspace or the nullspace?"

Quadrant interpretation:
- **top-right** (high drift + high sensitivity) — HIGHEST EV for
  engineering correction (canonical correction-targeting candidates).
- **top-left** (low drift + high sensitivity) — MPS-VIABLE confirmed for
  this byte region.
- **bottom-right** (high drift + low sensitivity) — locally-free zone
  (drift exists but doesn't affect score).
- **bottom-left** (low drift + low sensitivity) — neutral, no action.

### 5. List canonical plots and schema versions

```bash
.venv/bin/python tools/master_gradient_xray.py --list-plots
```

Useful for downstream tools that need to enumerate the plot taxonomy or
pin the sister JSON schema version.

## 5 canonical plot types

| plot_id | requires | output |
|---|---|---|
| `per_pair_distribution` | per-pair gradient anchor | histogram of per-pair \|g\| L1 per axis (3 sub-plots) |
| `per_byte_heatmap` | aggregate gradient anchor | top-K bytes × 3 axes heat (default K=128) |
| `cumulative_by_rank` | aggregate gradient anchor | Pareto leverage curve per axis with top-1% / top-10% annotations |
| `cross_substrate_correlation` | 2+ aggregate anchors | cosine-similarity matrix (Wyner-Ziv complementary signal) |
| `wyner_ziv_flow` | aggregate gradient + section manifest | per-section stacked bar (seg/pose/rate fraction per section) |
| `drift_vs_sensitivity_scatter` | aggregate gradient + `--mps-drift-json` | quadrant scatter with linear-fit overlay |

## Sister JSON sidecar contract (per Catalog #323 + #305)

Every plot in `--output-dir` mode gets a sister `<plot_id>.json`. Schema
version pinned: `master_gradient_xray_plot_sidecar_v1_20260519`. Fields:

```json
{
  "schema_version": "master_gradient_xray_plot_sidecar_v1_20260519",
  "plot_id": "per_byte_heatmap",
  "anchor": {
    "archive_sha256": "87ec7ca5...",
    "measurement_axis": "[macOS-CPU advisory]",
    "measurement_hardware": "darwin_arm64_local_cpu_advisory",
    "measurement_utc": "2026-05-19T01:29:45.818416+00:00",
    "gradient_tensor_kind": "aggregate_per_byte_v1",
    "n_bytes": 178162,
    "operating_point": {"d_seg": ..., "d_pose": ..., "rate": ..., "score": ...}
  },
  "summary_statistics": { ... },
  "extra": { ... },
  "provenance": {
    "artifact_kind": "predicted_from_model",
    "evidence_grade": "predicted",
    "promotion_eligible": false,
    "score_claim_valid": false,
    "measurement_axis": "[predicted]",
    "canonical_helper_invocation": "tac.provenance.builders.build_provenance_for_predicted"
  },
  "score_claim": false,
  "promotion_eligible": false,
  "ready_for_exact_eval_dispatch": false
}
```

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
**the visualization is a derived view, NOT a primary measurement.** Sister
JSONs are ALWAYS tagged `evidence_grade=predicted` with `score_claim=false`
and `promotion_eligible=false`. The underlying anchor's `measurement_axis`
determines any score-axis claim — sister JSONs preserve it under `anchor.measurement_axis`
for downstream consumers, but the **plot itself does not promote the
anchor**.

## index.html landing page (per Catalog #305 facet 4)

`--output-dir` mode emits `index.html` with:
- Operator question driving the analysis
- Archive-anchor table (sha[:12] / axis / hardware / tensor_kind / n_bytes / evidence_grade)
- All 5 plots with embedded PNG + sister JSON link
- Cross-references (canonical helpers, sister catalog gates, MPS drift JSON if provided)
- Schema version pinned for downstream queryable consumers

## Sample empirical output

The lane's sample run is at
`.omx/research/master_gradient_xray_sample_20260519/`. It processed 6
archives from the live ledger
(`.omx/state/master_gradient_anchors.jsonl`), emitting all 5 canonical
plots + `drift_vs_sensitivity_scatter` (cross-referenced against
`.omx/state/mps_drift_granular_20260519T122700Z.json`).

Top-3 actionable findings from the sample run:

1. **Leverage is broadly distributed across all 6 archives.** Top-1% byte
   leverage = 6.4% of total per-axis sensitivity (highly diffuse vs the
   ~50%+ that a Pareto-concentrated archive would show). This suggests
   per-byte targeted engineering corrections (Catalog #318 forbids raw
   byte authority, but typed `CandidateModificationSpec` rows can target
   ranges) have lower-than-naive expected value; structural changes
   (substrate-class shifts per Catalog #310) likely dominate.
2. **Cross-substrate correlation matrix** (cosine on aggregate L1 byte
   profiles) lets the operator identify Wyner-Ziv complementary
   candidates — archives whose leverage profiles disagree are stronger
   composition candidates than archives that overlap.
3. **MPS drift signal at landing was `NO_MASTER_GRADIENT_ANCHOR`** per the
   granular drift JSON (no per-pair drift rows). The drift scatter
   correctly falls back to per-frame drift overlay. Sister Phase B follow-up
   work landed in MPS Phase B which extends master-gradient anchors with
   per-pair tensors will enable the full quadrant analysis.

## Discipline

- Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" + Catalog
  #208: `--output` and `--output-dir` reject `/tmp/...` paths.
- Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192: plots from
  `darwin_arm64_*` advisory anchors render with an `[advisory: ...]`
  watermark on every figure title.
- Per Catalog #318: this tool consumes the master-gradient anchor as a
  diagnostic score-response tensor; it does NOT expose raw archive-byte
  authority APIs. Plot summaries are observability not score authority.
- Per Catalog #327: anchors carrying `[contest-CPU]` or `[contest-CUDA]`
  axis without authoritative custody are not silently consumed; the sister
  JSON sidecar preserves the original axis label so downstream consumers
  see the same custody as the canonical ledger reader.

## Integration

- Wired into `tools/operator_briefing.py` Phase 6 XRAY toolkit so the tool
  surfaces in the operator's daily briefing.
- Sister of Cable D consumer wave (`feedback_master_gradient_consumer_cathedral_wire_in_landed_20260519.md`)
  — the consumers compute the routing decisions; this tool lets the
  operator visualize the underlying signal those consumers are reading.

## See also

- `src/tac/master_gradient.py` — canonical anchor schema + ledger writer
- `src/tac/master_gradient_consumers.py` — canonical reader API
- `tools/extract_master_gradient.py` — producer CLI
- `tools/analyze_mps_drift_granular.py` — MPS drift JSON producer
- Catalog #305 / #323 / #318 / #327 — observability + provenance + authority
  invariants
