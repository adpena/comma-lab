Name: mrs6 | Archive: [upload archive.zip](ARCHIVE_LINK) | SHA-256: `aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957` | 179,286 B | GPU: yes, 1,045.8 s inflate on a Tesla T4 (evaluation 44.3 s) | Compression script: no | Decoder: eight files — `inflate.sh`, `inflate.py`, three C files, `README.md`, `FORMAT.md`, `archive.zip`. `FORMAT.md` documents the archive byte by byte. `inflate.sh` requires `cc` and stops if it is absent; there is no second implementation.
```text
=== Evaluation config ===
  batch_size: 16
  device: cuda
  num_threads: 2
  prefetch_queue_depth: 4
  report: /root/modal_auth_eval_work/eval_work/report.txt
  seed: 1234
  submission_dir: /root/modal_auth_eval_work/eval_work
  uncompressed_dir: /workspace/pact/upstream/videos
  video_names_file: /workspace/pact/upstream/public_test_video_names.txt
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00000414
  Average SegNet Distortion: 0.00010288
  Submission file size: 179,286 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00477517
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.14
```
