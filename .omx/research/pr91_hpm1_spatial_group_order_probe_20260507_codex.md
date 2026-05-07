# PR91 HPM1 Spatial Group Order Probe - 2026-05-07

Scope: PR91/HPM1 semantic entropy grammar only. Local CPU probe; no remote,
no GPU, no lane claim, and no score claim.

Artifact:

- `experiments/results/pr91_hpm1_spatial_group_order_probe_20260507_codex/spatial_group_order_probe.json`
- Tool: `tools/audit_pr91_hpm1_spatial_group_order_probe.py`
- Canonical payload without tool manifest SHA-256:
  `6532b11e2e25c7e39c74ca9146c21e7fa70ea865be6aba1797349e1795929815`

Hypothesis tested:

- The submitted source decodes HPAC patch groups using boolean-mask full-grid
  row-major traversal. This probe tested whether source-adjacent spatial
  traversal order explains the first PR91 HPM1 entropy mismatch.

Rows:

| candidate | status | first failure row | passes source failure row |
| --- | --- | --- | --- |
| `source_mask_row_major` | `failed_at_first_entropy_mismatch` | frame `0`, group `10`, symbol `191`, decoded-before `5951` | no |
| `full_col_major` | `failed_at_first_entropy_mismatch` | frame `0`, group `5`, symbol `473`, decoded-before `2201` | no |
| `tile_major_row_major` | `failed_at_first_entropy_mismatch` | frame `0`, group `12`, symbol `210`, decoded-before `8274` | yes |
| `phase_major_row_major` | `failed_at_first_entropy_mismatch` | frame `0`, group `11`, symbol `14`, decoded-before `6926` | yes |

Conclusion:

- `source_mask_row_major` reproduces the known source failure row exactly.
- `full_col_major` is a false lead because it fails earlier.
- `tile_major_row_major` and `phase_major_row_major` remain live local grammar
  clues: both decode past the source row but still fail inside frame 0.
- PR91/HPM1 remains non-dispatchable and not ready for exact eval. Full
  600-frame decode, byte-exact token re-encode, and sidecar-free runtime parity
  are still missing.

Verification:

```text
.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py -k 'spatial_group_order or probe_contracts_fail_closed_on_bad_inputs'
3 passed, 15 deselected
```
