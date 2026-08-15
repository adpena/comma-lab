# submission name: e480b RX2

Prepared by the repository operator. This is a hold-state draft and must not
be opened as a pull request until the download URL, exact-byte CPU row, source
visibility check, strict compliance pass, and five consecutive clean review
passes are complete.

# upload zipped `archive.zip`

Download status: pending operator-authorized public hosting. No public URL is
claimed in this draft.

Exact file identity:

- 183,502 bytes
- SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`
- single stored member `p`, 183,402 bytes, SHA-256
  `30c0165ec56dd9327ca4dcda477c34c25f7664622ac37ec8ed171114267d1b58`

# report.txt

```text
Evidence axis: [contest-CUDA]
Hardware: Tesla T4, Linux x86_64
Samples: 600
Average PoseNet Distortion: 0.00000688
Average SegNet Distortion: 0.00029611
Seg contribution: 0.029611
Pose contribution: 0.008294576541331089
Rate contribution: 0.12218644961582469
Archive size: 183502 bytes
Archive SHA-256: e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3
Recomputed score: 0.1600920261571558
Inflation wall time: 364.761996965 seconds
Evaluation wall time: 41.22217605 seconds

Evidence axis: [contest-CPU]
Status on the exact e480b bytes: pending; no CPU score claimed.
```

# eval host info

The exact measured row used a Linux x86_64 Tesla T4 and all 600 samples. The
exact e480b contest-CPU row has not been run. The retained predecessor and
macOS receipts below establish only CPU runtime feasibility and output
identity, not a transferable CPU score.

# build cost info

No public total-training-cost claim is made in this draft. The retained source
records the seeded terminal checkpoint, stage boundaries, source hashes, and
candidate materialization receipts. A final PR must publish a sanitized build
manifest from those real receipts rather than estimate or reconstruct a cost
after the fact.

# does your submission require gpu for evaluation (inflation)?

The measured score above used a T4 GPU. The shipped receiver is also
CPU-runnable through the device-flexible F26 port, but its exact-byte
`[contest-CPU]` score is still pending and must not be inferred from CUDA.

# did you include the compression script? and want it to be merged?

The score-bearing archive and inflation receiver are included. The exact
compression-side source exists at the pinned commit, including the RX2
current-label materializer, resumable HPAC trainer, terminal identity race,
and RX1 representation/packet builder. Those scripts currently contain
storage-layout paths and repository-internal dependencies, so they are **not
yet proposed for merge** and are not described as a friendly standalone
compressor. Before this answer can become “yes,” the operator must publish a
sanitized, seeded, documented bundle and prove that it reproduces the retained
archive from the pinned inputs. The HOLD state prevents a fake reproducibility
claim.

# changes from upstream

The learned semantic and carrier state is byte-identical to the PR130/PR135
lineage. This packet retrains the PR130-lineage HPAC probability object on the
current MC36 labels, carries the resulting current-label RC64 stream, uses the
RX1M split-section lossless container, and binds the F26 device-flexible CPU
port to the exact receiver. The detailed byte accounting below is the claim
boundary.

# competitive or innovative?

**Competitive: yes.** On the exact submitted bytes, the measured `[contest-CUDA]`
score is 0.1600920261571558, below the prior custodied 0.1619344578804448 row.
The improvement comes from a current-label HPAC retrain and lossless
whole-container composition while preserving exact decoded labels. The learned
semantic and carrier substrate is borrowed from PR130/PR135 and is itemized
below rather than presented as new work.

# additional comments

## Score and runtime boundary

The CUDA score is a 600-sample exact evaluation of the archive hash printed
above through the unmodified upstream scorer. CPU and CUDA are separate axes.
The exact e480b `[contest-CPU]` score remains pending.

Two CPU receipts bound feasibility without substituting for that missing row:

- `[contest-CPU]` predecessor: the 186,269-byte MC36 archive inflated in
  831.5345 seconds on Linux x86_64 CPU, leaving 2.1647x, rounded to 2.17x,
  headroom against the 1,800-second wall. Its score was
  0.20513189128858372 on different bytes and does not transfer to e480b.
- `[macOS-CPU advisory]` exact e480b receiver-identity replay: the
  device-flexible F26 port decoded the 183,502-byte archive in 915.5 seconds
  with raw-output identity to the predecessor vehicle. This proves the packet
  is CPU-runnable, not what its contest-CPU score will be.

## Borrowed-substrate accounting

| Section or mechanism | Classification | SHA-256 receipt and boundary |
|---|---|---|
| Semantic renderer physical state | `PR130/135-byte-identical` | decoded 36,040-byte section `b0d41ec904aca82f93f3c8bc68d0e48896ba08efdaa7a4a2ee204f002fc28ec8`; shipped as 34,763 bytes |
| Carrier physical state | `PR130/135-byte-identical` | decoded 22,219-byte section `065fce08fc3d44e49d29ad624561cbef86d01282cc73dcd32533b5d63115bd9f`; shipped as 22,161 bytes |
| HPAC IHS1 probability object | `PR130-lineage-retrained-on-our-labels` | decoded 17,996-byte object `94526d667a9c8b98f1e3ef8d39fe8769d6cc6721cb9a102629ad47f26016460d`; shipped as 13,619 bytes |
| RC64 token stream | `PR130-lineage-retrained-on-our-labels` | 112,749-byte stream `b981b8399f184795da7cd99b8ee44416bd672c8c4ed1672f1252b32a64c10627`; decoded label field `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52` |
| Current-label correction table and residual | `ours-original` | table `3572a0db3d511f2c26b0ade0734e11112fec3f068bcba5900b54a0646eae61ec`; residual `64bbf9dfd88d6eb50d111f72d968ab7e8f8dc0ab00fb675d8ed2ee8a410b73ac` |
| RX1M split container and lossless selection | `ours-original` | 70,557-byte wrapper `7cf390160189e8708faf3a7b09a76fc18cee85e45fdc7f71d30f725014417411`; learned contents retain the borrowing labels above |
| Device-flexible CPU port and receiver binding | `ours-original` | portable executable content tree `26c7d4f6a8e111c071d74208fc625bf2358e077a06dc59b54ec9421a8d198e0b`; underlying F26 renderer remains borrowed lineage |
| Final member and deterministic archive assembly | `ours-original` | member `30c0165ec56dd9327ca4dcda477c34c25f7664622ac37ec8ed171114267d1b58`; archive `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3` |

This is a new exact-byte composition and current-label probability retrain on
a PR130/PR135 learned substrate, not a claim that the learned vehicle is wholly
original.

## Public source and reproducibility

- Source repository: https://github.com/adpena/comma-lab
- Runtime/evaluation source pin:
  https://github.com/adpena/comma-lab/commit/19dd7916eb9ab5058bbeafa885ac68d8218d0a1e
- Portable executable-runtime content tree:
  `26c7d4f6a8e111c071d74208fc625bf2358e077a06dc59b54ec9421a8d198e0b`
- Upstream snapshot:
  `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`
- Compression-side files at the pin:
  `experiments/ddm_rx2_mc36_label_hpac.py`,
  `tools/train_ddm_cl1_hpac_capacity_mps.py`,
  `experiments/ddm_rx2_mc36_identity_race.py`, and
  `experiments/ddm_rx1_rate_representation_attack.py`.

Before submission, the operator must verify anonymous visibility of the pinned
source URL, replace the download-status paragraph with the verified public
archive URL and its hosted manifest, and either land a friendly reproducible
compression bundle or keep the compression-script answer “no” without implying
otherwise.
