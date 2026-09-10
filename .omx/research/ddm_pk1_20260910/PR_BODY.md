# Submission name

semantic_joint_ctxmix

# Upload zipped `archive.zip`

Draft release notice: the move-40 URL below is planned and has not been hosted or verified. Remove this notice only after publication authorization and a successful download/hash check.

Download: [archive.zip](https://github.com/adpena/comma_video_compression_challenge/releases/download/semantic_joint_ctxmix-move40/archive.zip)

- SHA-256: `986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`
- Size: 180,233 bytes

# report.txt

```
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00000489
  Average SegNet Distortion: 0.00010636
  Submission file size: 180,233 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00480039
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.14

Archive SHA-256: 986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857
Archive size bytes: 180233
Recomputed score from report components and exact archive bytes: 0.13763861019288715
Measurement axis: [contest-CUDA T4 n600]; deployment target: linux-nvidia-t4
```

The evaluation result block is retained from the Tesla T4 evaluation over all 600 samples; the custody lines were added during packet preparation. The score recomputed from the evaluator's eight-decimal component values and exact archive size is 0.13763861019288715. This is report-component precision, not unrounded tensor precision.

# Does your submission require GPU for evaluation (inflation)?

Yes. Deployment target: `linux-nvidia-t4`. The retained run measured 990.054 s inflation and 45.231 s evaluation. This update makes no contest-CPU score claim.

# Did you include the compression script? And want it to be merged?

This update packet contains the retained archive and its inflation runtime. It does not contain a compression script that rebuilds the move-40 archive. The previous packet's five-stage compression script does not establish reproduction of these updated bytes. Training and solve reproduction remain separate from the scored decoder artifact. Supporting research history is at https://github.com/adpena/comma-lab; this draft does not assert a newly published source commit.

# Is this submission competitive or innovative? Explain why

Competitive on the measured contest-CUDA axis: the retained move-40 score is 0.13763861019288715. The update composes segmentation repairs and rate-directed token edits, re-verifies the edited field, and re-solves the stored pose carrier on their union. It retains the earlier semantic context-mixing receiver.

The learned semantic renderer and pose-carrier vehicle are inherited from PR #130 (Fesal Fayed, @fesalfayed) and PR #135 (Shreyan Mohanty, @codexblack), with PR #133 (@JasonMo123) in the ancestry. I claim the decision and representation improvements on top of that vehicle, not the inherited vehicle itself.

# Additional comments

I used automated research and engineering tools extensively for the work behind this submission. The linked repository contains implementation history, experiment receipts, and provenance records.

This is a retained T4 result, not a new evaluation performed while preparing this update.
