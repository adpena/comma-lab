# ddm_mx2 Parity And Smoke

## Results

- Python compile: passed for mx2 wrapper/test files.
- Unit slice: `PYTHONPATH=src .venv/bin/python -m pytest src/tac/pr130_lift/tests/test_mx2_pose_lift.py` -> 5 passed in 0.18 s.
- CPR1 codec smoke: deterministic PR130 legacy-shape symbols encoded and decoded exactly through vendored `carrier_codec.py`.
- Generic coder smoke: stored/deflate/brotli/lzma race decoded exactly on the test payload.
- MLX execution probe: **BLOCKED** locally.

MLX probe result:

```text
RuntimeError: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.
```

The package import can expose `Device(gpu, 0)`, but a manual tiny `mx.array([1.0]) + 1.0` evaluation fails with the Metal runtime error above. Therefore no local MLX/PoseNet parity pass is claimed. The unit test now mocks the missing-runtime branch instead of importing MLX, so the test suite can pass in this headless/no-Metal environment.

## Non-Claims

- No synthetic tensor parity is promoted to real-frame parity.
- No real decoded `upstream/videos/0.mkv` frame parity was run.
- No batch-shape parity claim was made.
- No gradient parity claim was made.
- No n600 scorer work was run; the full scorer slot belongs to et4.

## Shape And Render Binding

The mx2 MLX wrapper follows upstream PoseNet preprocessing shape: two RGB frames become `t * 6 = 12` YUV channels before `MLXPoseNetAdapter`. This was checked against `upstream/modules.py` before landing the wrapper.

The carrier render order mirrors PR130's `learned_pose_carrier_oracle.render_slave`: uint8 master at camera resolution, bilinear down to eval resolution, add 12-D carrier, uint8 at eval resolution, bicubic up to camera resolution, uint8 again, then PoseNet preprocessing downsamples the two camera-resolution frames.

## Residual Risk

The local tests verify custody, import safety, CPR1 symbol closure, noncarrier rejection, and exact generic round-trip. They do not verify training convergence or MLX numerical parity on real frames. Those remain MAIN-only gates.
