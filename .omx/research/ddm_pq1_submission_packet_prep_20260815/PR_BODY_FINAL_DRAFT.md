# submission name:

joint_waterfill_rider

# upload zipped `archive.zip`

Attached to this pull request via the upload feature. Verify the exact bytes after
download:

- SHA-256: `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`
- Size: `180,002` bytes
- One stored member: `p`, `179,902` bytes
- Member SHA-256: `cf1afed8542e9dbd274b52ef14cc844a42cf2d659efceecf33edc3ab59c2edac`

# report.txt

```text
=== Evaluation results over 600 samples (device: cuda) ===
  Average PoseNet Distortion: 0.00000637
  Average SegNet Distortion:  0.00020139
  Submission file size:       180,002 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate:           0.00479424
  Seg contribution:            0.020139
  Pose contribution:           0.007981227975693965
  Rate contribution:           0.11985594327989708
  Score recomputed from components: 0.14797617125559104
  Evaluator display at 2 decimals: 0.15
```

This is a `[contest-CUDA]` result on a Linux x86_64 Tesla T4 over all 600 samples. The
exact score is recomputed from the components; the evaluator's two-decimal display is not
the cited value.

There is no `[contest-CPU]` score for this archive. A same-lineage predecessor measured
`0.20513189128858372` at 186,269 bytes on `[contest-CPU]` Linux x86_64 over 600 samples;
that result belongs to the predecessor bytes and is not transferred to this archive.

# does your submission require gpu for evaluation (inflation)?

Yes. Select `linux-nvidia-t4` and evaluate with `--device cuda`.

Measured T4 times: 578.935 seconds for inflation and 42.696 seconds for evaluation
(conservative charged total 621.632 seconds), inside the 30-minute budget with
comfortable margin. The component timings are measured; the residual cold-cache ceiling
is a projection.

# did you include the compression script? and want it to be merged?

Yes, with a stated limitation. The included compression entry point does not rebuild
these exact bytes, because five late lossless stages are not wired behind the single
entry point. It refuses this archive's SHA up front and names the missing stages rather
than producing substitute bytes. Merge is requested only if that limitation and the
section-level lineage accounting below are acceptable.

# is this submission competitive or innovative? explain why

**Competitive on the claimed axis:** `0.14797617125559104` `[contest-CUDA T4, n600]`
versus the public PR #135 result of `0.162` on the same axis.

**What is inherited, credited plainly.** The learned semantic renderer and pose-carrier
vehicle come from PR #130 and PR #135, with PR #133 transitively in the lineage. No
originality is claimed for that vehicle; the accounting is itemized section by section in
`BORROWED_SUBSTRATE_ACCOUNTING.md`.

**What is original here.** The decision and lossless-representation layer built on top of
that vehicle: joint admission of segmentation edits priced against their pose cost, an
exact receiver-assembly identity check, and a sequence of lossless model, coder, and
container transforms. The five moves after the prior packet freeze removed 454 bytes with
byte-identical decoded output, so their entire score improvement is rate.

**Independent development, with dates and no priority claims.** Parts of this line were
developed independently before the PRs that later shipped related ideas publicly: a
stored PoseNet-target sidecar that conditions the decoder on ground-truth pose targets
was in this project's tree on 2026-04-11, months before PR #130's semantic-pose carrier
was published; a direct-partition coding stack (region-adjacency, contour coding,
margin-based region merging — treating the segmentation partition itself as the coded
object) was in the tree on 2026-06-10, before PR #130's dense semantic-token vehicle
appeared. PR #130/#135 shipped working public vehicles for these ideas first, and this
submission builds on their shipped form; the dates are stated for honest accounting of
concurrent work, not as priority claims. Likewise PR #138 published the online
decode-time-corrector mechanism class first; the corresponding work here was concurrent,
no priority is claimed, and a later refinement used PR #138's state-synchronization
warning as a design check.

# additional comments

**Baseline.** PR #135 supplied the trained vehicle and the edit-then-recompensate
pattern. Most learned content in this archive is inherited and credited per section.

**Change.** Joint edit admission re-solves the pose carrier against the edited renders;
lossless model, coder, and container steps then re-represent the same decoded object.
The result is 180,002 bytes, 454 bytes below the prior packet freeze, with the same
3,662,409,600-byte CUDA raw output.

**Score.** `S = 0.14797617125559104` at 180,002 bytes on `[contest-CUDA T4, n600]`.
Archive SHA-256
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.

**What did not work better.** Re-mixing the stored 12-dimensional pose basis was a
measured null. A token-drop admission branch remains unbuilt because the receiver has no
path for it. The single-entry-point rebuild is verified only on an older packet
generation and is not claimed for this archive.

**Known runtime limitation.** The evaluated inflater pins Brotli and may install it from
the network if absent, and it compiles native code during inflation. These are stated
properties of the evaluated receiver, not hidden portability claims.

**Verification identity.** The archive is bound to 38 enumerated receiver rows: runtime
tree `6cdfa27dd1e9b46fc2bbbe88774c78d95ed3605fee7a15ba3861f96e24041e58`, portable content
tree `4856087f5f857c83f045736db1db18d41667eb98942b25242422ab636a797c84`, runtime-files
digest `b2638b491371fd0961382b99f1dfacb42b2b22ae37c28ee4306f7e0ae1b32ffc`.

Public source repository: <https://github.com/adpena/comma-lab>.

<!--
STORES CONSULTED: pq7 Yousfi-comment census · retained #137/#138 API receipts + 09-01 gh
census (#139) · live README + PR template (source-inspected) · pinned snapshot drift diff ·
gen-7 packet materials · canonical frontier pointer (afr1 components) · git-dated
provenance receipts (fea4a953f9 2026-04-11 · 752a30cdb9 2026-06-10) · task rows #1111/#1363.
This comment is invisible when rendered and is removed at publish time.
-->
