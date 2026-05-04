# STBM1BR Rust Decode Acceleration - 2026-05-04

## Scope

Lowered the PR90-derived `STBM1BR\0 + brotli(QTBM5)` semantic mask decode hot
path into Rust under `runtime-rs/crates/stbm1br-codec`.

This is a CPU/runtime velocity patch only. No remote jobs were dispatched, no
GPU eval was run, and no score claim is made.

## Evidence

- [empirical:experiments/results/stbm1br_rust_decode_profile_20260504_codex/timing_report.json]
  Real STBM1BR segment:
  `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/mask_segment.stbm1br`.
- Segment bytes: `152439`.
- Segment SHA-256:
  `1b1ec60b64e284aae11e838dc3d9996bce00125df5712a8ba9c3e8f739c9d313`.
- Decoded render-order shape: `(600, 384, 512)`.
- Decoded bytes: `117964800`.
- Python reference SHA-256:
  `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`.
- Rust release CLI decoded-output SHA-256:
  `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`.
- Parity: passed, byte-for-byte.

## Timing

- Python reference full decode: `30.820150707964785s`.
- Rust release CLI subprocess wall time: `0.9265687079750933s`.
- Rust decoder self-reported decode time: `0.756s`.
- Debug Rust smoke before release profiling: `14.242s`.

## Integration Gate

The default Python reference path remains the fallback. The contest inflate path
may use Rust only when `PACT_STBM1BR_RUST_DECODER` points to an executable
decoder that is part of the fixed runtime or deliberately bundled by the
experiment. Missing, non-executable, bad-magic, unsupported-subformat, or
shape-mismatched Rust decode fails closed.

This avoids hidden sidecars: a candidate archive/runtime must carry or fix the
native binary before exact inflate can rely on it.

## Verification Commands

```bash
cargo test -p stbm1br-codec
.venv/bin/python -m pytest src/tac/tests/test_stbm1br_rust_bridge.py src/tac/tests/test_stbm1br_mask_codec.py
.venv/bin/python experiments/profile_stbm1br_rust_decode.py
PACT_STBM1BR_RUST_DECODER=runtime-rs/target/release/stbm1br-codec .venv/bin/python - <<'PY'
import hashlib, importlib.util
from pathlib import Path
p=Path('submissions/robust_current/inflate_renderer.py')
spec=importlib.util.spec_from_file_location('inflate_renderer_stbm_gate_smoke', p)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
seg=Path('experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/mask_segment.stbm1br')
masks=mod._load_masks_from_stbm1br(seg, expected_frames=600)
raw=masks.detach().cpu().numpy().astype('uint8', copy=False).tobytes()
print(tuple(masks.shape), getattr(masks, '_half_frame_only', False), hashlib.sha256(raw).hexdigest())
PY
```

Inflate gate smoke output: `(600, 384, 512) True
0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`, with
`via rust (0.8s, half-frame)` in the loader log.

## Result

The Rust path is ready for fixed-runtime experiment integration behind the
explicit decoder-path gate. It is not yet a promoted contest-runtime default.
