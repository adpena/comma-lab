# submission name:
hnerv_lc_v2_scale095_rplus1

# upload zipped `archive.zip`
Included in this PR under `submissions/hnerv_lc_v2_scale095_rplus1/archive.zip`.

# report.txt
CPU-only run with `CUDA_VISIBLE_DEVICES=''` so inflation also used CPU:
```
=== Evaluation config ===
  batch_size: 16
  device: cpu
  num_threads: 2
  prefetch_queue_depth: 4
  report: submissions/hnerv_lc_v2_scale095_rplus1/report.txt
  seed: 1234
  submission_dir: submissions/hnerv_lc_v2_scale095_rplus1
  uncompressed_dir: /root/comma_video_compression_challenge/videos
  video_names_file: /root/comma_video_compression_challenge/public_test_video_names.txt
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00003460
  Average SegNet Distortion: 0.00057602
  Submission file size: 178,981 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00476704
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.20
```
Exact score from rounded CPU report components: `0.1953791765`.

Fast PyAV/CUDA scorer during tuning gave exact `0.194986956` with Pose `0.000033274` and Seg `0.000575697`.

# does your submission require gpu for evaluation (inflation)?
no

# did you include the compression script? and want it to be merged?
yes. `compress.sh` reproduces `archive.zip` by fetching the unchanged PR #100 release archive payload and verifying SHA256. Yes, please merge this submission if accepted.

# additional comments
Built on top of @BradyMeighan's `hnerv_lc_v2` PR #100, which is built on top of @EthanYangTW's `hnerv_muon_finetuned_from_pr95` PR #98 and @AaronLeslie138's `hnerv_muon` PR #95.

Changes from PR #100:
- retuned latent correction scale from `0.0100` to `0.0095`;
- added a zero-byte decode-side nudge: frame 0 red channel `+1`.

The archive payload is unchanged from PR #100; only inference-time code constants changed.
