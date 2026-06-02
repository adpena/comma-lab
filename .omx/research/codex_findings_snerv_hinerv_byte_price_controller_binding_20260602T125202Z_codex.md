# Codex findings: SNeRV and HiNeRV byte-price controller binding

## Verdict

SNeRV is not method-negative from the current evidence. The current explicit-LF
storage route is rate-blocked, but that is a representation/export bottleneck,
not a proven fundamental limitation of the SNeRV family.

The shared reusable surface for SNeRV and HiNeRV is
`tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller`: it prices
candidate bytes against the fixed contest byte price, preserves custody and
axis blockers, and emits final `demote` decisions whenever scorer replay,
archive SHA, full-video coverage, or receiver proof is missing.

## Landing

- `tac.analysis.snerv_lf_payload_codec_sweep` now emits controller-ready
  `section_value_rows` and an embedded `byte_price_plan`.
- `tac.analysis.hinerv_archive_size_ladder` now emits controller-ready
  modelsize-increment rows and an embedded `byte_price_plan`.
- `tools/build_snerv_lf_payload_codec_sweep.py` can decode an actual SNAR1
  receiver packet, sweep LF payload codecs, and write JSON/Markdown reports.

## Important bug caught

The first packet-derived SNeRV smoke selected a failed int2 row because failed
modes had `payload_bytes = 0` and the sweep sorted by byte count only. That is
now fixed: failed/zero-byte rows sort after valid packets, and a regression test
ensures failed zero-byte modes cannot win selection.

Superseded smoke:
`snerv_lf_payload_codec_sweep_20260602T125127Z`.

Corrected smoke:
`snerv_lf_payload_codec_sweep_20260602T125202Z`.

## Evidence

- Focused tests: `12 passed`.
- Ruff: clean for the changed SNeRV/HiNeRV/controller surfaces.
- Py compile: clean for the changed SNeRV/HiNeRV/tool surfaces.
- Corrected tiny SNAR packet sweep selected valid `int64_lzma` at 419 B; int2,
  int4, and int8 refused because the decoded LF values exceeded their exact
  signed ranges.

## Authority

All rows remain false-authority. The new surfaces are planning/rate-control
tools only. They do not claim contest score, promotion readiness, rank/kill
authority, or exact-eval readiness.

## Next work

Run the SNeRV LF codec sweep on a real full-600 SNAR packet after the next
receiver package is emitted, then feed paired scorer non-rate deltas into the
same controller. For HiNeRV, attach decoder-weight group saliency/scorer replay
to the modelsize-increment rows so the controller can decide which model-size
bytes, quant groups, zeros, or residual sidecars economically deserve to exist.
