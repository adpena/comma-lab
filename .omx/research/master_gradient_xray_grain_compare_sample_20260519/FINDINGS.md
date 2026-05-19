<!-- SPDX-License-Identifier: MIT -->
# master_gradient_xray --grain compare_both — empirical sample findings (2026-05-19)

## Run

```bash
.venv/bin/python tools/master_gradient_xray.py \
    --archive-sha 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf \  # FEC6 frontier (CPU 0.19205)
    --archive-sha 7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb \
    --archive-sha 87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5 \  # A1 baseline
    --archive-sha 9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4 \  # PR106 format0d (CUDA 0.20533)
    --archive-sha b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e \  # PR101 op7
    --archive-sha f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd \  # A1 cascade backfill anchor
    --grain compare_both \
    --output-dir .omx/research/master_gradient_xray_grain_compare_sample_20260519
```

## Grain inventory at sample time

Live `.omx/state/master_gradient_anchors.jsonl` carries 10 anchor rows
across 6 distinct archives, ALL at raw-byte grains:

| archive[:12]   | grain                        | utc                  |
|----------------|------------------------------|----------------------|
| 6bae0201fb08   | zip_inner_member_payload     | 2026-05-19T01:33:43Z + 2026-05-19T12:28:07Z |
| 7ecb0df1c462   | zip_inner_member_payload     | 2026-05-19T12:28:51Z |
| 87ec7ca5f2f3   | zip_inner_member_payload     | 2026-05-19T01:29:45Z + 2026-05-19T12:27:43Z |
| 9cb989cef519   | zip_inner_member_payload     | 2026-05-19T12:28:29Z |
| b83bf3488625   | zip_inner_member_payload     | 2026-05-19T01:32:04Z (x2) |
| f174192aeadf   | scored_archive_bytes         | 2026-05-17T19:02:09Z + 2026-05-18T14:45:02Z |

ZERO post-decompress anchors. The cascade_smearing_comparison plot is
therefore NOT emitted (per-archive eligibility requires BOTH grains).

## Findings

### Finding 1: cascade comparison cannot fire today

Pre-condition for the 7th plot is BOTH raw-byte AND post-decompress
anchors per archive. The live ledger contains only raw-byte anchors. The
plot is structurally available + tested (47 tests pass for slot 6 + slot
10) and will fire automatically when sister slot 15 + slot 17 land the
post-decompress anchors for PR101 + the 5 sister families (PR106 / PR107
apogee_v2 / A1 / DP1 / HDM8).

### Finding 2: sister slots 15 + 17 are in-flight at sample time

* Slot 15 (`lane_codex_op7_iteration_items_3_4_post_brotli_grain_plus_mps_20260519`):
  helper module `tac.master_gradient_post_brotli_decompress` (388 LOC) is
  landed AND `MUTATION_GRAIN_POST_BROTLI_DECOMPRESS` constant matches the
  canonical slot 6 `GRAIN_POST_BROTLI_DECOMPRESS` value. The anchor-emit
  side has NOT yet appended to `.omx/state/master_gradient_anchors.jsonl`.
* Slot 17 (`multi_archive_post_decomp` subagent in-flight): WIP file
  `src/tac/master_gradient_post_decompress_multi_archive.py` (~1039 LOC)
  carries the 5-family extension scaffold. Not yet committed.

Once both land, re-running this sample will emit the cascade comparison
plot for every archive that has both grains.

### Finding 3: structural protection per Catalog #318

The infrastructure is forward-compatible:

* `PerByteSensitivityPayload.gradient_byte_domain` defaults to
  `GRAIN_RAW_ARCHIVE_BYTE` for backward compat; new post-decompress
  anchors will flip `cascade_smearing_risk` to False automatically per
  the dataclass invariant.
* `load_per_byte_sensitivity_for_archive(prefer_grain="post_decompress")`
  cascade routes to post-decompress first; falls back to raw-byte with
  explicit warning.
* Cathedral consumer v1.1 emits `grain_routing_reason` documenting which
  grain was chosen + why; downstream operators see the disambiguation
  reasoning at audit time.
* XRay `--grain compare_both` adds Hook #6 PROBE_DISAMBIGUATOR visual
  surface (currently skipped per Finding 1; will fire automatically per
  Finding 2).

## Operator-routable top-3 actionable items (pending slot 15/17 anchor landing)

1. **WAIT-FOR-SLOT-15** — re-run this sample once slot 15 emits the PR101
   post-decompress anchor for archive `b83bf3488625` (PR101 op7 baseline)
   or any archive sister-extracted via the
   `tac.master_gradient_post_brotli_decompress.build_post_brotli_decompress_anchor_payload`
   path. The first cascade comparison plot will reveal the empirical
   verdict for PR101: HIGH (raw-byte gradient is meaningfully misleading)
   vs MEDIUM (partial agreement) vs LOW (grains close to identical).

2. **WAIT-FOR-SLOT-17** — re-run this sample once slot 17 extends
   post-decompress extraction to PR106 / PR107 / A1 / DP1 / HDM8. The
   per-archive verdict distribution will inform whether the post-decompress
   grain should be the DEFAULT routing for cathedral autopilot ranking
   (recommended IF cascade smearing is uniformly HIGH across families;
   per-family-dispatch otherwise).

3. **DEPLOY-CANONICAL-CASCADE-PROBE** — once a sister archive has both
   grains, operators can use the cascade comparison plot as a one-shot
   diagnostic: HIGH verdict immediately invalidates any raw-byte-based
   mutation targeting (per Catalog #318 + codex op7 finding); MEDIUM /
   LOW verdict allows either grain (post-decompress still preferred per
   the canonical contract).

## Schema versions pinned at sample time

- xray sidecar v2: `master_gradient_xray_plot_sidecar_v2_20260519`
- xray index html v2: `master_gradient_xray_index_v2_20260519`
- per_byte_sensitivity_consumer: v1.1
- consumer hooks: [1, 3, 4, 6] (PROBE_DISAMBIGUATOR added in v1.1)

## Sister artifacts

- `tools/master_gradient_xray.py` — operator-facing xray tool (commit `558230385`)
- `src/tac/master_gradient_per_byte_consumer.py` — canonical reader (commit `d154a1014`)
- `src/tac/cathedral_consumers/per_byte_sensitivity_consumer/__init__.py` — cathedral consumer v1.1 (commit `40c8c8abf`)
- `src/tac/master_gradient_post_brotli_decompress.py` — slot 15 PR101 helper (sister-landed)
- `src/tac/master_gradient_post_decompress_multi_archive.py` — slot 17 5-family WIP (sister in-flight)

## Provenance

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287 / #323:
this findings memo is a derived view tagged `[predicted]`. Score-axis
claims must reference the underlying contest-CUDA / contest-CPU anchors;
the xray visualization is observability-only.
