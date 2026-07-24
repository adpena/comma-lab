# DDM E3 inflate composition and dependency closure — build spec

## Objective

Promote the PA1 frame-0 scorer-stat affine from an encode-side measurement into
the governed E2 receiver and close the Brotli runtime dependency with one
receiver surface.

## Authority and boundaries

- Authority: task #661 delegated prompt, SHA-256
  `1b277419d78b48851e9f3f2726cb9e3f9affe2b77c270bc1be59ab4d6bb43dd1`.
- Allowed: local `$0` n600 replay through the frozen CPU Torch scorers and the
  actual upstream `evaluate.sh`.
- Forbidden: upstream mutation, paid or remote dispatch, Modal exact eval,
  frontier archive mutation, training, or a score/pointer claim.
- Evidence axis:
  `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`; pointer
  `0.1910828242 [contest-CPU]` remains unchanged.

## Design

1. Replace both counted DDM packet streams' Brotli framing with measured
   stdlib raw LZMA1 (`dict_size=1 MiB`, `lc=3`, `lp=0`, `pb=2`). On the sealed
   E2 raw streams this measured 17,768 B + 411,205 B, versus Brotli-Q11
   18,412 B + 315,033 B; it beats measured zlib9 and XZ6 among the tested
   stdlib forms. The receiver imports only stdlib LZMA plus Torch.
2. For E2, preserve a base-render stage for every 16-pair batch. Accumulate the
   exact official 12-channel YUV6 first and second moments from those preserved
   base bytes.
3. Derive the 12 gain/bias values at decode time from those moments and fixed
   PA1 scorer-only target moments. The target constants derive only from the
   frozen PoseNet first-stem convolution weights and BN running statistics;
   no scorer, scorer weight, GT statistic, or video-derived table is present
   in `inflate.py`.
4. Apply PA1's exact camera-side realizer to frame 0 only: official bilinear
   resize, RGB→YUV6, affine, block-constant inverse, low-resolution residual,
   bilinear lift, clamp/round/uint8. Preserve each corrected stage separately,
   then assemble and hash the final raw.
5. Recompute the exporter's expected output identity through the same receiver
   helpers. Mark D1 active and FREE; the LZMA packet-byte delta is COUNTED; no
   new amplitude payload member exists.

## Files in scope

- `src/tac/optimization/ddm_runtime_receiver.py`
- `src/tac/optimization/ddm_runtime_exporter.py`
- `src/tac/optimization/tests/test_ddm_runtime_exporter.py`
- `tools/rehearse_ddm_runtime_upstream.py`
- `tools/verify_ddm_runtime_export.py`
- DDM E3 typed configs, receipts, DAG/equations/findings/review artifacts

Do not edit `upstream/` or unrelated DDM/trainer surfaces.

## Acceptance

- Focused unit tests prove raw-LZMA framing, dependency closure, dynamic-affine
  equivalence to PA1 constants, frame-1 byte identity, resume custody, and
  tamper refusal.
- Exported archive is deterministic with exact byte homes and parse-back.
- Composed n600 receiver measurement matches PA1
  `d_pose=147.49104204339514` within float tolerance and preserves
  `d_seg=0.02861480712890625`.
- Full frozen upstream `evaluate.sh` passes in the locked environment, records
  component values, exact archive bytes/SHA, wall clock, and stays under
  1,800 seconds.
- Round-1 adversarial review plus three consecutive clean passes are durable.
- Commit is created on the isolated branch with no co-author trailer; MAIN
  reviews the full base-to-head diff before merge.
