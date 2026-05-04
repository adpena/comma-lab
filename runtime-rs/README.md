# runtime-rs

Rust workspace for hot-path experiments.

This lane is intentionally separate from the Python harness.

Targets:
- raw frame writer
- QMA-family semantic mask parser/decode prototypes
- STBM1BR top-band semantic mask decode acceleration
- sparse residual decode
- ROI patch application
- future CPU inflator core

Current QMA status:
- `crates/qma-codec` parses QMA9 headers fail-closed and decodes QMA9 storage-order masks in Rust.
- Tests cover a Python-contract QMA9 fixture and the known PR85 QMA9 segment header (`600 x 512 x 384`, `159011` packed bytes).
- This is not wired into `inflate.sh` or the robust submission runtime yet; public replay custody can keep the C++ decoder until the Rust path has full archive parity evidence.

Current STBM1BR status:
- `crates/stbm1br-codec` decodes the PR90-derived `STBM1BR\0 + brotli(QTBM5)` mask segment to render-order `uint8` bytes.
- Unsupported QTBM/top-band/boundary subformats fail closed; the first integration gate is exact byte parity against the Python `tac.stbm1br_mask_codec` reference and the real PR85 STBM1BR mask segment.
- It is not an ambient contest sidecar. Python inflate may only use it when an explicit decoder path is supplied by an experiment or fixed runtime.
