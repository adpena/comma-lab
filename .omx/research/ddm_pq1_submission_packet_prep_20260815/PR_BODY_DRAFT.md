# submission name: sz1 composed re-encode

Prepared by the repository operator. **This is a hold-state draft.** It must not
be opened as a pull request until the download URL, source-visibility check,
strict compliance pass, and five consecutive clean review passes are complete.
Generation 3 of this packet; it supersedes the rr4, hv1, and e480b drafts.

# upload zipped `archive.zip`

Download status: pending operator-authorized public hosting. No public URL is
claimed in this draft.

Exact file identity:

- 179,930 bytes
- SHA-256 `debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a`
- single stored member `p`, 179,830 bytes, SHA-256
  `be6db33bce471fe38b3d32cf6b421368721b1ea2ddd3f77b577f2bd27d06b7a8`, CRC32 3747474564

# report.txt

```text
Evidence axis: [contest-CUDA]
Hardware: Tesla T4, Linux x86_64
Samples: 600
Average PoseNet Distortion: 0.00000688
Average SegNet Distortion: 0.00029611
Seg contribution: 0.029611
Pose contribution: 0.008294576541331089
Rate contribution: 0.11980800143527229
Archive size: 179930 bytes
Archive SHA-256: debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a
Recomputed score: 0.15771357797660338
Inflation wall time: 1143.270127967 seconds
Evaluation wall time: 38.307284003 seconds

Evidence axis: [contest-CPU]
Status on these exact bytes: inflation measured 3,422.7 s vs the 1,800 s budget on 4-thread x86_64 CPU; harness failed closed. GPU required; no CPU score exists or is claimed.
```

# eval host info

Linux x86_64, Tesla T4, all 600 samples, unmodified upstream scorer
(`evaluate.py` SHA-256 `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`,
upstream snapshot `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`).
Inflation used 1,143.3 s of the 1,800 s budget — 1.57x headroom. The
token-mixer decode is the dominant cost; it is deterministic integer
arithmetic.

# build cost info

No public total-training-cost claim is made. This submission adds **no training
cost at all** over its base candidate: both changes are lossless re-encodes of
an already-trained archive (a decode-time probability model and a serialization
permutation), and both run in minutes from the retained checkpoint. The base
candidate's training cost is a separate figure and is not reconstructed after
the fact here.

# does your submission require gpu for evaluation (inflation)?

**Yes — this submission requires a GPU for evaluation, as a measured fact.**
The measured score above used a T4 (inflation 1,143.3 s of the 1,800 s
budget). We also measured the CPU side rather than assuming it: on a
contest-like 4-thread x86_64 CPU, inflation of these exact bytes took
3,422.7 s against the 1,800 s budget (token decode 3,108.7 s), and the
harness failed closed at 1,800 s. No CPU score exists and none is claimed.
The decode itself is correct on CPU — the decoded token field hashes to the
exact pinned value — so this is purely a wall-clock boundary, not a
correctness one.

# did you include the compression script? and want it to be merged?

**Yes, and it is offered for merge.** The chain, stated exactly:

- **Stage A — provenance (documented, not re-run).** Reproducing the underlying
  checkpoint from raw video is multi-day GPU compute. The chain emits the
  lineage, the stage scripts, their arguments, and the input manifest with
  every SHA-256. It does not pretend to re-run training.
- **Stage B — build (exact and verifiable).** From the retained checkpoint the
  chain replays the shipped decode order, re-encodes the token stream under
  the 13-context integer log-odds mixer, applies the semantic serialization
  split, and repacks the archive. Each stage hashes its output and fails
  closed on mismatch. At build time `archive.zip` and a fresh-process repeat
  were byte-identical.
- **Stage C — decode.** Runs the shipped receiver over the rebuilt archive and
  checks the decoded token field against its pinned SHA-256
  (`9ba2e52b3096…`) — verified in a clean environment on CPU.

Honesty note on packaging status: the prior generation's single entry point
(`experiments/ddm_pq2_compress_e2e.py`) asserts the prior archive's hashes and
is being re-bound to this generation's chain (the fx2 byte-close driver plus
the sz1 split builder, both in the public repository) under the same
fail-closed assertions. That re-bind completes before this draft is opened.

# changes from upstream

This submission changes **two sections** of an inherited archive, both
losslessly:

1. The RC64 token stream is re-encoded from 110,512 to 109,801 bytes by a
   decode-time probability model (a 13-context fixed-point integer log-odds
   mixer) that ships in the receiver, stores zero archive bytes, and is driven
   entirely by already-decoded symbols. The decoded token field is
   byte-identical (SHA-256 `9ba2e52b3096…`).
2. 8,284 bytes of raw interleaved fp16 metadata in the semantic section are
   byte-planed (all high bytes, then all low bytes) before the container's
   Brotli pass, saving 520 bytes. The receiver applies the exact inverse
   permutation before parsing; decoded values are unchanged. The transform is
   versioned in bit 0 of an existing reserved header byte — zero transmitted
   bytes, and an archive without the bit decodes exactly as before.

Decode identity was verified at the byte level: the first inflated output
hashes identically between this row and the token-model-only row
(`9a6b75e5…`), so `d_seg` and `d_pose` are unchanged by construction and the
entire improvement is rate.

# competitive or innovative?

**Competitive, on a measured row, stated against what is actually verified.**

On the exact submitted bytes the measured `[contest-CUDA]` 600-sample score is
`0.15771357797660338`, which we re-derived from the reported components
independently. That is below the best ranked score on the leaderboard at the
time of writing (PR #135, `semantic-pose-HPAC_CPR1_polished`, 0.162) and below
our own prior custodied rows.

Three honesty qualifications we would rather state than have found:

1. There is an open PR claiming `0.1591495384` (PR #138, `opal_v1`). That
   figure is **author-claimed and not yet evaluated by the maintainers**, as is
   ours until this PR is run. Our measured number is lower than that claim,
   but we are comparing a measured row against an unverified one and say so.
2. **PR #138 published the decode-time-corrector mechanism class first, and we
   did not know it when we built our first corrector.** Its online correction
   is learned from the already-decoded prefix, reproduced identically by
   encoder and decoder, adds no table or weight to the archive, and yields
   pure rate — the same class as our token probability model, by a different
   construction. PR #138 opened 2026-08-17 08:31Z; our first measured
   corrector result landed 14:41Z the same day, and we first read PR #138 at
   19:32Z, after our byte-close. We describe this as **concurrent independent
   development** and make **no priority claim**. PR #136 is adjacent and also
   earlier. The 13-context mixer in this generation is our own deepening of
   that shared mechanism class; the semantic serialization split is a separate
   transform class (a storage-layout permutation, not a probability model).
3. The **innovation here is narrow and we scope it narrowly**: two zero-byte
   receiver mechanisms that losslessly save 2,829 archive bytes relative to
   the inherited base (1,598 + 711 on the token stream across two
   generations, 520 on the semantic serialization). The learned vehicle
   underneath — the semantic renderer and the pose carrier — is PR130/PR135
   lineage, is not ours, and is itemized below rather than folded into the
   claim. The one learned object in the archive that *is* ours is the HPAC
   probability object: PR130's architecture, retrained here on our own label
   field. It is inherited unchanged from the base candidate in this
   generation.

# additional comments

## Score and runtime boundary

The CUDA score is a 600-sample exact evaluation of the archive hash printed
above through the unmodified upstream scorer. CPU and CUDA are separate axes.
On the CPU axis we report a measured boundary instead of a pending promise:
inflation of these exact bytes on a contest-like 4-thread x86_64 CPU took
3,422.7 s against the 1,800 s budget, so no contest-CPU score exists on these
bytes and none is claimed. The CPU decode is correct (decoded token field
hash-identical to the pinned value); the boundary is wall-clock, and the
dominant term is the token-mixer decode, which is the named optimization
surface if CPU feasibility is ever wanted.

## Borrowed-substrate accounting

Classes: `inherited-substrate` (theirs, used as-is) · `mechanism-adopt-with-
attribution` (their idea or source, our implementation or re-fit) ·
`ours-original` (built here, with a receipt).

"Byte-identical to base" means identical to the archive we inherited at the
previous step, **not** identical to PR130's or PR135's bytes — the base already
contains our retrained HPAC object and our compensation edits. For the
semantic section, "value-identical" means the decoded tensors are unchanged
while the on-disk serialization is ours.

| Section or mechanism | Classification | SHA-256 receipt and boundary |
|---|---|---|
| Semantic renderer state | `inherited-substrate` (PR135 lineage, decoded values unchanged); **on-disk serialization ours** (byte-planed, receiver un-splits) | decoded values verified identical through the patched receiver; split length 8,284 B |
| Pose carrier state | `inherited-substrate` (PR135, byte-identical) | decoded 22,242 B, `196f0e5136f4d6bf…` |
| Compressed model container | `inherited-substrate`; unchanged from base, PR-level equality not independently verified | 70,453 B, `e35d12371fa79747…` |
| **HPAC probability object** | **`mechanism-adopt-with-attribution`** — PR130's architecture, **retrained here on our own label field**; inherited unchanged in this generation | checkpoint ep0634 selected from 81 retained candidates |
| Compensation blob | `mechanism-adopt-with-attribution` — container inherited, contents include our admitted edits | 36 B, `38792b4953318117…` |
| Residual payload + table codes | `inherited-substrate`; **provenance unresolved, no originality claimed** | residual 100 B `74775aab04c7615c…` |
| **RC64 token stream (re-encoded)** | **`ours-original` probability model over inherited symbols** — 13-context fixed-point integer log-odds mixer | 109,801 B, `5b09fd784a7c80cf…` (prior 110,512 B); decoded field `9ba2e52b3096…` unchanged |
| **Semantic serialization split** | **`mechanism-adopt-with-attribution`** — byte-plane (shuffle-filter) layouts are standard compression practice (HDF5/Blosc lineage); the section-scoped application, zero-byte reserved-bit versioning, and receiver un-split are ours | −520 B measured through the container's own Brotli with a delta-zero control |
| RC64 backend, encoder side | `inherited-substrate` (PR135, verbatim) | compiles PR135's `rc64_backend.c` unmodified |
| RC64 backend, shipped receiver | `mechanism-adopt-with-attribution` (PR135-derived, modified) | shipped `05839d1416e68a49…`, which **differs** from the PR135 source |
| Receiver binding and archive assembly | `ours-original` | validated runtime tree `0d0fc008d6a37bd5…`; archive `debb025f45bb42e3…` |
| Compression chain | `ours-original` | fx2 byte-close driver + sz1 split builder; e2e re-bind in progress (stated above) |

This remains a lossless re-encode program on a PR130/PR135 learned substrate,
not a claim that the learned vehicle is original. The full accounting,
including the ancestry chain and the open provenance items, is in
`BORROWED_SUBSTRATE_ACCOUNTING.md`, shipped beside this archive.

## Credits and prior work

Every number below was read from the pull request itself, not from our notes.

- **PR #130 — `semantic-pose-HPAC_CPR1`, Fesal Fayed (`fesalfayed`)**, leaderboard
  0.172, archive 191,052 B. The base vehicle: the semantic-token / HPAC / CPR1
  architecture this submission descends from.
- **PR #135 — `semantic-pose-HPAC_CPR1_polished`, Shreyan Mohanty (`codexblack`)**,
  leaderboard 0.162, archive 186,724 B. The archive we actually built on. Its
  semantic renderer and pose carrier decode identically from our archive, and
  our encoder compiles its `rc64_backend.c` unmodified. It is also the ranked
  score this submission is measured against.
- **PR #133 — `cpr1_cbq_matched8`, `JasonMo123`**, leaderboard 0.166. Not taken
  directly, but in our ancestry transitively: PR #135 already incorporates its
  constrained basis and re-solved int12 carrier.
- **PR #138 — `opal_v1`, Cristian (`ccastillo1043`)**, author-claimed 0.1591495384.
  Published the decode-side probability-correction mechanism class first; see
  the competitive section above. Concurrent and independent; no priority claim.
- **PR #136 — `hnerv_rc`, Jacky Li (`JPL11`)**. Adaptive range coding with
  per-tensor context reset, on a different vehicle. Adjacent prior work.
- **Upstream** — `commaai/comma_video_compression_challenge`: the scorer,
  `evaluate.py` (`7da71a84ce24286b…`), the frozen SegNet and PoseNet weights, the
  600-sample test list, and the 37,545,489-byte denominator.
- **Third-party runtime** — PyTorch, Brotli 1.2.0, `constriction`, and a C
  compiler at inflate time.

## Public source and reproducibility

- Source repository: https://github.com/adpena/comma-lab
  (anonymous visibility to be re-verified at packet freeze)
- Evaluation source pin: commit `2e0af59966c4a1405bad342de5969d0de4d99f7a` —
  the commit the T4 evaluation actually ran from (matches
  `provenance.pact_commit` in the receipt)
- Encoder source pins: commit `31c64e4ce0…` (semantic serialization split) and
  commit `85880c77a6…` (token probability model, frozen shipped configuration)
- Validated runtime tree SHA-256:
  `0d0fc008d6a37bd5cfa804073e617a8ea30a7c6b6e6c4a1022e2c5d7ce6f9513`
- Upstream snapshot:
  `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`
- Chain scripts: `experiments/ddm_rr2_encoder_byteclose.py` (encode/build),
  `experiments/ddm_rr2_receiver_close.py` (receiver/parse-back),
  `experiments/ddm_fx2_model_axis_corrector.py` and
  `experiments/ddm_fx1_logistic_mixer_corrector.py` (token probability model),
  `experiments/ddm_sz1_semantic_metadata_split.py` (serialization split).

Before submission the operator must verify anonymous visibility of the pinned
source URLs, complete the e2e entry-point re-bind, and replace the
download-status paragraph with the verified public archive URL and its hosted
manifest.
