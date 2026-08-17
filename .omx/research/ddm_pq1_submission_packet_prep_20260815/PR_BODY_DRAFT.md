# submission name: rr4 free-corrector re-encode

Prepared by the repository operator. **This is a hold-state draft.** It must not
be opened as a pull request until the download URL, source-visibility check,
strict compliance pass, and five consecutive clean review passes are complete.
Generation 2 of this packet; it supersedes the e480b and hv1 drafts.

# upload zipped `archive.zip`

Download status: pending operator-authorized public hosting. No public URL is
claimed in this draft.

Exact file identity:

- 181,161 bytes
- SHA-256 `35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956`
- single stored member `p`, 181,061 bytes, SHA-256
  `1a6b40cc7bee289e5efd4ce81205888ef23829ed4a78c198344bb679ba9da47a`, CRC32 885609521

# report.txt

```text
Evidence axis: [contest-CUDA]
Hardware: Tesla T4, Linux x86_64
Samples: 600
Average PoseNet Distortion: 0.00000688
Average SegNet Distortion: 0.00029611
Seg contribution: 0.029611
Pose contribution: 0.008294576541331089
Rate contribution: 0.12062767380656568
Archive size: 181161 bytes
Archive SHA-256: 35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956
Recomputed score: 0.15853325034789678
Inflation wall time: 476.611040218 seconds
Evaluation wall time: 61.047152556 seconds

Evidence axis: [contest-CPU]
Status on these exact bytes: pending; no CPU score claimed.
```

# eval host info

Linux x86_64, Tesla T4, all 600 samples, unmodified upstream scorer
(`evaluate.py` SHA-256 `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`,
upstream snapshot `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`).
Inflation used 476.6 s of the 1,800 s budget — 3.78x headroom.

# build cost info

No public total-training-cost claim is made. This submission adds **no training
cost at all** over its base candidate: it is a lossless entropy re-encode of an
already-trained archive, and the re-encode itself runs in well under a minute
from the retained checkpoint. The base candidate's training cost is a separate
figure and is not reconstructed after the fact here.

# does your submission require gpu for evaluation (inflation)?

**Yes — treat this submission as requiring a GPU for evaluation.** The measured
score above used a T4.

Being precise rather than convenient: the receiver has been demonstrated to
decode these exact bytes on CPU (the parse-back produced an output byte-identical
to the base candidate's own CPU inflate), so CPU decoding is not impossible. But
we have **not** measured a `[contest-CPU]` score on these bytes, and we will not
infer one from the CUDA row. Until that row exists, "requires GPU" is the honest
answer.

# did you include the compression script? and want it to be merged?

**Yes, and it is offered for merge.** The entry point is
`experiments/ddm_pq2_compress_e2e.py`. What it does, stated exactly:

- **Stage A — provenance (documented, not re-run).** Reproducing the underlying
  checkpoint from raw video is multi-day GPU compute. The script emits the
  lineage, the stage scripts, their arguments, and the input manifest with every
  SHA-256. It does not pretend to re-run training.
- **Stage B — build (exact and verifiable).** From the retained checkpoint it
  replays the shipped decode order, re-encodes the token stream under the
  decode-time corrector, splices it into the member, and repacks the archive.
  It then **hashes the result and exits non-zero unless it equals
  `35ac2b9beb7e6fa8…`**. It also writes a second archive from the same member and
  records whether the two are byte-identical; only the sha and byte-count checks
  gate the exit code (the repeat result is reported, not exit-gating).
- **Stage C — decode.** Runs the shipped receiver over the rebuilt archive and
  checks the decoded token field against its pinned SHA-256.

Verified on 2026-08-17: rebuilt into a clean store, token stream hashed
`6c3757bd52a18d3c…` (match), archive hashed `35ac2b9beb7e6fa8…` (match),
determinism repeat byte-identical.

The script contains no local filesystem layout. Every input root is supplied
through `--inputs-json` or the matching environment variables and is verified by
SHA-256 before any stage runs; `--emit-inputs-template` prints the schema.

# changes from upstream

This submission changes **one section** of an inherited archive. Seven of the
eight parsed sections are byte-identical to the base candidate; the eighth, the
RC64 token stream, is re-encoded from 112,110 to 110,512 bytes by a decode-time
probability corrector that ships in the receiver, stores zero archive bytes, and
is driven entirely by already-decoded symbols. The decoded token field is
byte-identical to the base candidate's (SHA-256 `9ba2e52b3096…`), so `d_seg` and
`d_pose` are unchanged by construction and the entire improvement is rate.

# competitive or innovative?

**Competitive, on a measured row, stated against what is actually verified.**

On the exact submitted bytes the measured `[contest-CUDA]` 600-sample score is
`0.15853325034789678`, which we re-derived from the reported components
independently. That is below our own prior custodied row and below the best
ranked score on the leaderboard at the time of writing (PR #135,
`semantic-pose-HPAC_CPR1_polished`, 0.162).

Two honesty qualifications we would rather state than have found:

1. There is an open PR claiming `0.1591495384`. That figure is **author-claimed
   and not yet evaluated by the maintainers**, as is ours until this PR is run.
   We are not asserting a win over an unverified number; we are reporting our
   measured one.
2. The **innovation here is narrow and we scope it narrowly**: a zero-byte
   decode-time probability corrector that losslessly saves 1,598 archive bytes.
   The learned vehicle underneath — semantic renderer, carrier, HPAC probability
   object — is PR130/PR135 lineage, is not ours, and is itemized below rather
   than folded into the claim.

# additional comments

## Score and runtime boundary

The CUDA score is a 600-sample exact evaluation of the archive hash printed above
through the unmodified upstream scorer. CPU and CUDA are separate axes and the
exact `[contest-CPU]` score on these bytes remains pending. One CPU receipt bounds
feasibility without substituting for that missing row: the receiver parse-back
decoded these exact bytes on arm64 CPU and produced a 3,662,409,600-byte output
whose SHA-256 equals the base candidate's own CPU inflate. That proves the packet
is CPU-runnable and that the decoded content is identical; it is not a CPU score.

## Borrowed-substrate accounting

| Section or mechanism | Classification | SHA-256 receipt and boundary |
|---|---|---|
| Semantic renderer state | `PR130/135-byte-identical` | decoded 36,051 B, `b489c73567046e64…`; byte-identical to base |
| Carrier state | `PR130/135-byte-identical` | decoded 22,242 B, `196f0e5136f4d6bf…`; byte-identical to base |
| Compressed model container | `PR130/135-byte-identical` | 70,453 B, `e35d12371fa79747…`; byte-identical to base |
| HPAC probability object | `PR130-lineage`, inherited unchanged here | 17,952 B, `e8c0cfd73d3275ad…`; byte-identical to base |
| Compensation blob | `PR130/135-byte-identical` | 36 B, `38792b4953318117…`; byte-identical to base |
| Residual payload + table codes | `ours-original`, inherited unchanged here | residual `74775aab04c7615c…`; table codes `76afdc3ceda1212a…` |
| **RC64 token stream (only changed section)** | **`ours-original` estimator over borrowed probabilities** | 110,512 B, `6c3757bd52a18d3c…` (base 112,110 B, `73a878891a31c366…`); corrector `96fd35aaf82c737a…` |
| RC64 backend, encoder side | `PR135-byte-identical` | compiles PR135's `rc64_backend.c`, `5c75e2c70b89f148…`, unmodified |
| RC64 backend, shipped receiver | `PR135-lineage-modified` | shipped `05839d1416e68a49…`, which **differs** from the PR135 source above |
| Receiver binding and archive assembly | `ours-original` | runtime tree `7acedb07e670e76c…`; archive `35ac2b9beb7e6fa8…` |
| End-to-end compression entry point | `ours-original` | `experiments/ddm_pq2_compress_e2e.py`, rebuild verified |

This is a lossless entropy re-encode on a PR130/PR135 learned substrate, not a
claim that the learned vehicle is original.

## Public source and reproducibility

- Source repository: https://github.com/adpena/comma-lab
  (PUBLIC — operator-authorized re-publication 2026-08-17; anonymous visibility verified
  HTTP 200 on the repo and on the evaluation pin, same-day)
- Evaluation source pin: commit `e7ca85754bb9e6a4b319e5a8fa206366c90bd6f4` — the commit
  the T4 evaluation actually ran from (matches `provenance.pact_commit` in the receipt)
- Compression-script pin: commit `a411f612aaf095ec27ff05eaf38c5d8c17f28c30` — the commit
  that carries `experiments/ddm_pq2_compress_e2e.py` (the entry point landed AFTER the
  evaluation pin; two labelled pins are stated rather than one misstated pin)
- Runtime tree SHA-256: `7acedb07e670e76c798f153ac53a3045b053074d702e226411a2353745b98351`
- Portable executable-runtime content tree:
  `4358aaf34fcbfc1cdc4a8865b9aead709199465c9909321abf279ebcd0fe3721`
- Upstream snapshot: `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`
- Compression entry point: `experiments/ddm_pq2_compress_e2e.py`; stage scripts
  `experiments/ddm_rr2_encoder_byteclose.py` and
  `experiments/ddm_rr2_receiver_close.py`; corrector
  `experiments/ddm_rr4_free_corrector_v2.py`.

Before submission the operator must verify anonymous visibility of the pinned
source URL and replace the download-status paragraph with the verified public
archive URL and its hosted manifest.
