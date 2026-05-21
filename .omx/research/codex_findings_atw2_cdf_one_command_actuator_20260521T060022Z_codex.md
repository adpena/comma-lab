# Codex Findings - ATW2 CDF One-Command Compaction Actuator

**Timestamp (UTC)**: 2026-05-21T06:00:22Z  
**Scope**: Add a repeatable operator command for the ATW2 CDF removal lane: materialize smoke `archive.zip`, compact the current-runtime-dead CDF section, prove raw parity, and write a custody report.  
**Verdict**: ACTUATOR_LANDED_AND_VERIFIED

## Summary

Codex added `tools/materialize_atw2_cdf_compaction_smoke.py`, a one-command runner that composes the existing ATW2 smoke trainer with the compact-CDF archive.zip proof helper.

The tool performs the full local smoke proof path:

1. runs `experiments/train_substrate_atw_codec_v2.py --smoke`,
2. requires the trainer to emit `source/archive.zip`,
3. writes `compact/archive.zip` with the compact CDF sentinel,
4. proves raw-output parity through the current ATW2 inflate path,
5. writes `materialized_compaction_report.json` and `.md`.

All outputs are marked:

- `score_claim`: false
- `promotion_eligible`: false
- `ready_for_exact_eval_dispatch`: false

## Code Change

- `tools/materialize_atw2_cdf_compaction_smoke.py`
  - New operator-facing actuator for the ATW2 CDF removal lane.
  - Default output path: `experiments/results/atw2_cdf_materialized_smoke_<utc>`.
  - Preserves report-level custody fields for source payload, source ZIP, compact ZIP, trainer stdout/stderr, smoke stats, and proof object.
- `src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py`
  - Adds CLI coverage for the new tool.

## Verification

Focused test command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/substrates/atw_codec_v2/tests/test_atw_codec_v2.py \
  src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py
```

Result:

- `40 passed in 15.51s`

One-command actuator run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/materialize_atw2_cdf_compaction_smoke.py \
  --output-dir experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z \
  --epochs 1 --device cpu --variant B \
  > experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z.stdout.json
```

## Empirical Results

Source smoke payload:

- Path: `experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z/source/0.bin`
- Bytes: `90,850`
- SHA-256: `83410a0a9962327bdc401112a94a897b461fe46b7a82e0644fa63220042d787b`

Source ZIP:

- Path: `experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z/source/archive.zip`
- Bytes: `90,664`
- SHA-256: `dbde1713566a06538a6e718fca8d855b190cb3581d3ddf05d4bfa205c94a4f0f`

Compact ZIP:

- Path: `experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z/compact/archive.zip`
- Bytes: `88,266`
- SHA-256: `dff8e3f2ecfb06fcb787bfcc2954ceb314971469639a4d188b6cb5c8c65fb57f`

Compaction proof:

- ZIP bytes saved: `2,398`
- ZIP rate-only delta estimate: `-0.001596729769586967`
- Inner ATW2 bytes saved: `2,552`
- Compact CDF bytes: `8`
- Raw output bytes compared: `48,832,128`
- Raw output equality: `true`
- Max absolute raw byte delta: `0`
- Source raw SHA-256: `c2c6e18c3ca3706d437126a5fd632be9d8e36eb88b8259353db17ddafc4c88fd`
- Compact raw SHA-256: `c2c6e18c3ca3706d437126a5fd632be9d8e36eb88b8259353db17ddafc4c88fd`

Ignored proof artifact hashes:

- `experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z/materialized_compaction_report.json`: `e7dd625e6a21facb64ab104e3dab9cb9656b118a58c99eb423c6f6346683c56e`
- `experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z/materialized_compaction_report.md`: `237192b9e356e55e70d104e460333435986ce7a22071289f08494cfae02df800`
- `experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z.stdout.json`: `e7dd625e6a21facb64ab104e3dab9cb9656b118a58c99eb423c6f6346683c56e`

## Interpretation

The ATW2 CDF removal lane now has a single reproducible local command. This removes a procedural gap between the synthetic helper proof and any future full ATW2 candidate archive.

This still does not prove contest score movement. The artifact is a tiny CPU smoke archive, not a full 600-pair candidate. The next frontier-moving use of this actuator is to replace the smoke trainer input with a full ATW2 candidate archive, or generalize the same one-command shape to other substrates with parser-visible decode-opaque sections.
