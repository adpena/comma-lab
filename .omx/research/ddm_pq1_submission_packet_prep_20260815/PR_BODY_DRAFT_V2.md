# INTERNAL PROPOSAL — NOT AUTHORIZED FOR PUBLICATION

The repository owner must verify every statement, rewrite the final description
in their own words, supply a fetched-back archive URL for the exact bytes below,
and explicitly authorize publication. This proposal does not authorize hosting,
opening or editing a pull request, or posting a comment.

# submission name:

joint_waterfill_rider

# upload zipped `archive.zip`

Not yet hosted. Do not publish a URL until it resolves with HTTP 200 to exactly:

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

This is a `[contest-CUDA]` result on a Linux x86_64 Tesla T4 over all 600
samples. The exact score is recomputed from the components; the evaluator's
two-decimal display is not the cited value.

There is no AFR1 `[contest-CPU]` score. A same-lineage predecessor measured
`0.20513189128858372` at 186,269 bytes on `[contest-CPU]` Linux x86_64 over 600
samples. That result belongs to the predecessor bytes and is not transferred to
this archive.

# does your submission require gpu for evaluation (inflation)?

Yes. Select `linux-nvidia-t4` and evaluate with `--device cuda`.

The measured T4 times were 578.935 seconds for inflation and 42.696 seconds for
evaluation. The conservative charged total was 621.632 seconds. This passed a
projected 822-second cold-cache residual ceiling by 200.368 seconds; the
component timings are measured, while the residual ceiling is a projection.

# did you include the compression script? and want it to be merged?

Yes, with a fail-closed limitation. The included compression entry point does
not rebuild these exact AFR1 bytes because five post-rc2 lossless stages are not
wired behind it. It refuses the AFR1 SHA before doing work and names the missing
stages instead of producing substitute bytes. Merge is requested only if that
limitation and the section-level lineage accounting are acceptable.

# is this submission competitive or innovative? explain why

Competitive on the claimed axis: `0.14797617125559104` `[contest-CUDA]` versus
the public PR #135 result of `0.162` on `[contest-CUDA]`.

The learned semantic renderer and pose-carrier vehicle are inherited from PR
#130 and PR #135, with PR #133 transitively in the lineage; no originality is
claimed for that vehicle. The contribution is the decision and lossless
representation layer over it: joint admission of segmentation edits against
their pose cost, receiver assembly, and lossless coder/container transforms.
The five moves after the prior packet freeze removed 454 bytes with
byte-identical decoded output, so their full score improvement is rate.

PR #138 published the online decode-time-corrector mechanism class first. The
base work here was developed concurrently, but no priority is claimed; a later
refinement used PR #138's state-synchronization warning as a design check.

# additional comments

**Baseline.** PR #135 supplied the trained vehicle and edit-then-recompensate
pattern. Most learned content in this archive is inherited and is credited in
`BORROWED_SUBSTRATE_ACCOUNTING.md` section by section.

**Change.** Joint edit admission re-solves the pose carrier against the edited
renders, then lossless model, coder and container steps re-represent the same
decoded object. AFR1 is 180,002 bytes, 454 bytes below the prior packet freeze,
with the same 3,662,409,600-byte CUDA raw output.

**Score.** `S = 0.14797617125559104` at 180,002 bytes on
`[contest-CUDA T4, n600]`. Archive SHA-256 is
`cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`.

**What did not work better.** Re-mixing the stored 12-dimensional pose basis
was a measured null. A token-drop admission branch remains unbuilt because the
receiver has no path for it. The single-entry-point rebuild is verified only on
an older packet generation and is not claimed for AFR1.

**Known runtime limitation.** The evaluated inflater pins Brotli and may install
it from the network if absent, and it compiles native code during inflation.
Those are properties of the evaluated receiver, not hidden portability claims.

**Verification identity.** The archive is bound to 38 enumerated receiver rows:
runtime tree
`6cdfa27dd1e9b46fc2bbbe88774c78d95ed3605fee7a15ba3861f96e24041e58`,
portable content tree
`4856087f5f857c83f045736db1db18d41667eb98942b25242422ab636a797c84`,
and runtime-files digest
`b2638b491371fd0961382b99f1dfacb42b2b22ae37c28ee4306f7e0ae1b32ffc`.

Public source repository: <https://github.com/adpena/comma-lab>.


<!-- INTERNAL PROPOSAL ANNOTATION (not public text; final body is operator-authored) -->
STORES CONSULTED: pq7 Yousfi-comment census (81 threads / 76 comments, drafting rules) ·
retained #137/#138 API receipts + MAIN's 09-01 gh census (#139) · live README + PR template
(VERIFIED-VIA-LIVE-SOURCE-INSPECTION) · pinned upstream snapshot (drift diff, read-only) ·
gen-7 packet materials (pq12 reswap memo + compliance table) · canonical frontier pointer
(afr1 components) · git-dated provenance receipts (fea4a953f9 04-11 · 752a30cdb9 06-10, in
the companion PROVENANCE_ADDENDUM) · task ledger rows #1111/#1363.
