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

Yes, `linux-nvidia-t4` `--device cuda`. Measured on T4: 578.9 s inflation +
42.7 s evaluation.

# did you include the compression script? and want it to be merged?

Yes to both.

# is this submission competitive or innovative? explain why

Competitive: `0.14797617125559104` on contest CUDA vs PR #135's `0.162` on the
same axis.

The learned renderer and pose-carrier vehicle are inherited from
PR #130/#133/#135. My work is the optimization layer on top of that vehicle —
twenty-three admitted improvement moves, each accepted only on the recomputed
exact 600-sample score of the rebuilt archive (or, for the lossless moves, a
bit-identical decode proof plus exact rate arithmetic), never a proxy:

- **Joint edit admission** (the submission's namesake): candidate segmentation
  edits of the semantic tokens are solved across 573 pairs, each priced against
  its exact pose cost through the frozen PoseNet, and admitted through a
  Lagrange-multiplier waterfill — 455 of 573 admitted. The pose carrier is then
  re-solved (damped Gauss–Newton) against the edited renders. This is the
  largest single move and produced the sub-0.15 crossing.
- **In-compile pose compensation**: a frame-0 compensation solved inside the
  compile so segmentation edits carry approximately zero pose tax — on its
  proof row, pose distortion landed below the unedited base.
- **A zero-byte pose re-solve** of the stored carrier coefficients against the
  frozen scorer on the shipped renders.
- **A lossless representation chain** on the coder and container: fixed-point
  integer log-odds context mixing, group-conditioned token contexts, an
  address-free tile-conditioned re-encode, and container transforms. The five
  moves after the prior packet freeze cut 454 bytes with byte-identical decoded
  output — 0 differing bytes across all 3,662,409,600 raw output bytes.

What didn't work better: re-mixing the stored 12-dimensional pose basis was a
measured null; sub-4-bit semantic quantization lost far more distortion than
its rate credit at every depth tried; a token-drop admission branch was not
built because the receiver has no path for it.

Some of this line predates the related public PRs, including a stored
pose-target sidecar on 2026-04-11 and a direct-partition coding stack on
2026-06-10, and our decode-time corrector work ran concurrent with PR #138,
which published that class first.

# additional comments

One runtime note: the inflater pins Brotli (and installs it from the network if
absent) and compiles native code during inflation.

Source: <https://github.com/adpena/comma-lab>

<!--
INTERNAL (removed at publish): v3 — operator's 2026-09-01 tightening edits folded in
(report.txt trimmed further · GPU one-liner · "Yes to both" · merged additional-comments
paragraph into the innovation section) + the enriched innovation mechanisms per operator
"add a bit of additional detail on our original work and innovations". GATE: "Yes to both"
on the compression script is contingent on ddm_ce1's byte-exact e2e proof (arm live);
restore one honest limitation clause if ce1 blocks. Numbers verified against: afr1 pointer
receipt (S/bytes/sha/identity counts) · ddm_pq2_compress_e2e.py NOT_EXPRESSIBLE registry
(573 pairs / 455 admitted, jg5 stage list) · qs5 verdict (compensation below-base proof) ·
up3 (zero-byte pose re-solve) · hot-state POINTER_LINE (native receiver identity
3,662,409,600 B / 0 differing; 454 B post-freeze tail). Per the contest coding-agents
policy the final public text is operator-authored; this remains source material.
STORES CONSULTED: pq7 census · live PR template · canonical frontier pointer · pq2
refusal registry · qs5/up3/jg5 receipts · task rows 1111/1363/1382.
-->
