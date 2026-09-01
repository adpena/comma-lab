# submission name:

joint_waterfill_rider

# upload zipped `archive.zip`

Attached to this pull request. Verify after download:

- SHA-256: `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`
- Size: `180,002` bytes (one stored member `p`, 179,902 bytes)

# report.txt

```text
=== Evaluation results over 600 samples (device: cuda) ===
  Average PoseNet Distortion: 0.00000637
  Average SegNet Distortion:  0.00020139
  Submission file size:       180,002 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate:           0.00479424
```

Score recomputed from components: `0.14797617125559104` (the evaluator prints 0.15
at two decimals).

# does your submission require gpu for evaluation (inflation)?

Yes — `linux-nvidia-t4`, `--device cuda`. Measured on T4: 578.9 s inflation +
42.7 s evaluation, well inside the 30-minute budget.

# did you include the compression script? and want it to be merged?

Yes, with a limitation: it does not rebuild these exact bytes — five late
lossless stages are not wired behind the single entry point. It refuses this
archive's SHA up front and names the missing stages rather than producing
substitute bytes. Fine either way on merging given that.

# is this submission competitive or innovative? explain why

Competitive: `0.14797617125559104` on contest CUDA (T4, all 600 samples) vs
PR #135's `0.162` on the same axis.

The learned renderer and pose-carrier vehicle are inherited from PR #130/#135
(PR #133 in the lineage) — no originality claimed there; per-section accounting
is in `BORROWED_SUBSTRATE_ACCOUNTING.md`. What's ours is the layer on top:
joint admission of segmentation edits priced against their pose cost, an exact
receiver-assembly identity check, and a chain of lossless model, coder, and
container transforms — the last five moves cut 454 bytes with byte-identical
decoded output.

Some of this line predates the related public PRs in our tree — a stored
pose-target sidecar on 2026-04-11 and a direct-partition coding stack on
2026-06-10, both before PR #130 — and our decode-time corrector work ran
concurrent with PR #138, which published that class first. Dates stated for
honest accounting of concurrent work; no priority claimed.

# additional comments

PR #135 supplied the trained vehicle. Joint edit admission re-solves the pose
carrier against the edited renders, then lossless steps re-represent the same
decoded object — the same 3,662,409,600-byte CUDA raw output at 454 fewer
bytes.

One runtime note: the inflater pins Brotli (and installs it from the network if
absent) and compiles native code during inflation.

Source: <https://github.com/adpena/comma-lab>

<!--
INTERNAL (removed at publish): tightened v2 per operator 2026-09-01 "Should it be
streamlined or tightened"; cuts vs PR_BODY_FINAL_DRAFT.md recorded in session.
STORES CONSULTED: pq7 Yousfi-comment census (terse-body norm) · live PR template ·
canonical frontier pointer (afr1) · git-dated provenance receipts (fea4a953f9 ·
752a30cdb9) · task rows #1111/#1363/#1381.
-->
