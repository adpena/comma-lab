# PR #110 Live Surface Update Applied

**UTC:** 2026-05-20T14:33Z; second style pass applied 2026-05-20T14:46Z
**PR:** https://github.com/commaai/comma_video_compression_challenge/pull/110
**Release:** https://github.com/adpena/comma_video_compression_challenge/releases/tag/fec6-frontier-submission-20260520
**PR head after edit:** `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`
**Mutation scope:** PR body text + release body text/title only. No branch push, no release asset replacement.

## Validated Facts Used

- Local archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Local archive bytes: `178517`
- ZIP member: single member `x`, stored uncompressed, compressed/uncompressed size `178417`
- Release asset after edit: `archive.zip`, size `178517`, digest `sha256:6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- PR #110 head after edit: `adpena/comma_video_compression_challenge:hnerv_fec6_fixed_huffman_k16` at `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`
- Author handles verified by `gh api`: PR #95 `@AaronLeslie138`, #98 `@EthanYangTW`, #100 `@BradyMeighan`, #101 `@SajayR`, #102 `@EthanYangTW`, #103 `@rem2`, #110 `@adpena`
- PR #108 discussion attribution verified: `@YassineYousfi` comment on 2026-05-11.

## Corrections Applied

1. Removed the false implication that `archive.zip` is embedded in the PR tree; it is release-hosted.
2. Pasted the CPU `report.txt` block and full component recomputation.
3. Added paired CUDA/T4 component recomputation while keeping CPU and CUDA axes separate.
4. Replaced the `single-thread` / `ubuntu-latest` inference with factual `num_threads: 2` wording.
5. Replaced the broad false HNeRV `~0.0008` cluster claim with the narrower PR #101 -> PR #110 delta.
6. Removed deep pinned comma-lab doc links from the PR body; kept root-level research references only and stated they are not runtime dependencies.
7. Fixed release body stale smoke instructions and "companion PR will be opened" wording.
8. After downloading the `Zipper` email attachment, applied validated tone/style recommendations: replaced "packet" rhetoric with "submission" where appropriate, softened the recompression/saturation claim, clarified training scripts are not part of the submission runtime, and changed the release title from "frontier" to "selector submission."

## Inbox Package Intake

- GWS query: `subject:Zipper filename:zip`
- Gmail message id: `19e45d4d3a5a22d1`
- From: `Alejandro Pena <alejandrod.pena@icloud.com>`
- Subject: `Zipper`
- Date: `Wed, 20 May 2026 09:40:21 -0500`
- Attachment: `research_package.zip`
- Local path: `.omx/research/inbox_zipper_20260520T144021Z_codex/research_package.zip`
- Attachment SHA-256: `9ffce5d7802ebdb0669b21888b7ec76496561dee652c72f7993be803c38d2506`
- Attachment bytes: `46396`

## Verification After Edit

Stale-string scans of the live PR body and release body returned no hits for:

```text
cluster within
single-thread
matching the ubuntu-latest
active candidate inventory
full-stack source map
b7f16a
b392343d
Claude
Codex
persona
Anthropic
companion PR will be opened
```

The release asset digest and byte count were unchanged.

After the second pass, the live PR body was re-read from GitHub and the head
remained `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`. The release asset was
again re-read from GitHub and remained `archive.zip`, `178517` bytes,
`sha256:6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`.
