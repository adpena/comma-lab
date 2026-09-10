# Submission name

semantic_joint_ctxmix

# Upload zipped `archive.zip`

Draft release notice: the move-43 URL below is planned and has not been hosted or verified. Remove this notice only after publication authorization and a successful download/hash check.

Download: [archive.zip](https://github.com/adpena/comma_video_compression_challenge/releases/download/semantic_joint_ctxmix-move43/archive.zip)

- SHA-256: `7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e`
- Size: 180,466 bytes

# report.txt

```
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00000459
  Average SegNet Distortion: 0.00010345
  Submission file size: 180,466 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00480660
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.14

Archive SHA-256: 7beb6a5fc7c2bf477d04a107ab0cf4113d3bd74b2c5a6b94ff77f7ff61971c1e
Archive size bytes: 180466
Recomputed score from report components and exact archive bytes: 0.1372848557085275
Measurement axis: [contest-CUDA T4 n600]; deployment target: linux-nvidia-t4
```

The evaluation result block is retained from the Tesla T4 evaluation over all 600 samples; the custody lines were added during packet preparation. The score recomputed from the evaluator's eight-decimal component values and exact archive size is 0.1372848557085275. This is report-component precision, not unrounded tensor precision.

# Does your submission require GPU for evaluation (inflation)?

Yes. Deployment target: `linux-nvidia-t4`. The retained run measured 1089.634 s inflation and 42.740 s evaluation. This update makes no contest-CPU score claim.

# Did you include the compression script? And want it to be merged?

This update packet contains the retained archive and its inflation runtime. The retained legacy compression script is included for custody; it does not rebuild the move-43 archive. The previous packet's five-stage compression script does not establish reproduction of these updated bytes. Training and solve reproduction remain separate from the scored decoder artifact. Supporting research history is at https://github.com/adpena/comma-lab; this draft does not assert a newly published source commit.

# Is this submission competitive or innovative? Explain why

Competitive on the measured contest-CUDA axis: the retained move-43 score is 0.1372848557085275. The update adds token pre-distortion pass 6 to the move-42 field and re-solves the stored pose carrier on that edited field. It retains the earlier semantic context-mixing receiver.

The learned semantic renderer and pose-carrier vehicle are inherited from PR #130 (Fesal Fayed, @fesalfayed) and PR #135 (Shreyan Mohanty, @codexblack), with PR #133 (@JasonMo123) in the ancestry. I claim the decision and representation improvements on top of that vehicle, not the inherited vehicle itself.

# Additional comments

I used automated research and engineering tools extensively for the work behind this submission. The linked repository contains implementation history, experiment receipts, and provenance records.

This is a retained T4 result, not a new evaluation performed while preparing this update.
