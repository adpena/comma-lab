# rr4 free-corrector re-encode — submission packet

This directory describes the exact 181,161-byte archive and receiver used for the
measured rr4 result. The score-bearing payload is `archive.zip`; the receiver is
`inflate.sh`, `inflate.py`, `cpr1/`, and `runtime/`.

## Evidence boundary

- `[contest-CUDA]`: a 600-sample exact evaluation on a Tesla T4 measured
  `S = 0.15853325034789678` on the archive identified below.
- `[contest-CPU]`: pending on these exact archive bytes. **No CPU score is
  claimed.**
- CPU runtime evidence, which is not a score: the receiver decoded these exact
  bytes on arm64 CPU and produced a 3,662,409,600-byte output whose SHA-256
  equals the base candidate's own CPU inflate. That proves the packet is
  CPU-runnable and that the decoded content is identical. It does not tell you
  what the contest-CPU score would be, and it is not used as if it did.

## Exact identity

- Archive SHA-256:
  `35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956`
- Archive bytes: `181161`
- Member: `p`, `181061` bytes, stored, SHA-256
  `1a6b40cc7bee289e5efd4ce81205888ef23829ed4a78c198344bb679ba9da47a`
- Runtime tree SHA-256:
  `7acedb07e670e76c798f153ac53a3045b053074d702e226411a2353745b98351`
- Portable executable-runtime content tree SHA-256:
  `4358aaf34fcbfc1cdc4a8865b9aead709199465c9909321abf279ebcd0fe3721`
- Upstream snapshot SHA-256:
  `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`
- Evaluation source commit: `e7ca85754bb9e6a4b319e5a8fa206366c90bd6f4`
- Public source repository: https://github.com/adpena/comma-lab
- Pinned source revision (what the T4 evaluation ran from):
  https://github.com/adpena/comma-lab/commit/e7ca85754bb9e6a4b319e5a8fa206366c90bd6f4
- Compression-script revision (the entry point landed after the evaluation pin):
  https://github.com/adpena/comma-lab/commit/a411f612aaf095ec27ff05eaf38c5d8c17f28c30

Two labelled revisions are given rather than one misstated one. Anonymous
visibility of both was verified the day this packet was frozen.

## Custody correction: two receipt files in this tree are stale

`GENERATION_RECEIPT.json` and `RECEIVER_PARSEBACK.json` were inherited from the
base candidate's source tree and were never regenerated. They declare a
182,759-byte archive with SHA-256 `80d9c8c6fdc72caa…`. **That is not the archive
in this directory.** The archive here is 181,161 bytes, SHA-256
`35ac2b9beb7e6fa8…`, as stated above and as measured by the evaluator.

They are stale **labels**, not stale proofs. The genuine binding receipts are
`RESULT_build.json`, `RESULT_receiver_build.json`, and `RESULT_parseback_v2.json`
in the retained store, all bound to `35ac2b9beb7e6fa8…`.

The two files are shipped unchanged, and this correction is stated here rather
than in the tree, for one specific reason: the evaluated runtime-tree hash
`7acedb07e670e76c…` is computed over those files. Editing them, or adding a
correction file beside them, would change that hash and break replay against the
value recorded in the exact-authority row. A reader must be able to reproduce the
evaluated tree byte-for-byte, so the tree is left alone and the correction lives
in this README, which is not part of the hashed set.

The same two files also contain absolute paths from the machine that generated
them. They are inert: nothing in `inflate.sh`, `inflate.py`, `cpr1/`, or
`runtime/` reads either file. They are shipped only because the evaluated
runtime-tree hash was computed over them, and for the same reason they are not
edited.

## Two names a reader will meet in the receiver

Both are cosmetic and neither changes behaviour.

- `CP135` and `F26` appear in `inflate.sh` as an error string, environment
  variables, and file names. They are internal codenames for the inherited
  PR130/PR135 lineage, kept because renaming them would change the evaluated
  runtime-tree hash.
- `inflate.sh` carries a `Darwin` branch that calls `brew --prefix libomp`. It is
  **unreachable on the contest runner**: it requires `F26_TOKEN_DECODER` to equal
  `native-hpac`, and the script defaults that variable to `python` with nothing
  setting it otherwise. It is a local acceleration path, not a macOS
  requirement. This submission assumes Linux.

## What this submission is

A lossless entropy re-encode of an inherited archive. Seven of its eight parsed
sections are byte-identical to the base candidate. Only the RC64 token stream
changed, 112,110 to 110,512 bytes, because a decode-time probability corrector
ships in the receiver, stores zero archive bytes, and is driven entirely by
already-decoded symbols. The decoded token field is unchanged, so `d_seg` and
`d_pose` are unchanged by construction and the improvement is entirely rate.

Read `BORROWED_SUBSTRATE_ACCOUNTING.md`, shipped in this directory, before
treating any of the learned content as ours. Most of it is not — the semantic
renderer and the pose carrier are PR #135's, byte-identical after decode. The
same table is reproduced inline in the pull-request body. It also records that
the decode-time-corrector mechanism class was published first by PR #138, opened
the same day our result was measured; we make no priority claim.

## Reproduction

The end-to-end compression entry point is `experiments/ddm_pq2_compress_e2e.py`.
It contains no local filesystem layout; supply input roots through
`--inputs-json` (run `--emit-inputs-template` for the schema) or the matching
environment variables. Every input is verified by SHA-256 before any stage runs.

```bash
python experiments/ddm_pq2_compress_e2e.py --emit-inputs-template
python experiments/ddm_pq2_compress_e2e.py \
    --stage all --resume \
    --store <working directory> \
    --inputs-json <your manifest>
```

The build stage exits non-zero unless the rebuilt archive hashes to the pinned
value. Verified 2026-08-17 into a clean store: token stream
`6c3757bd52a18d3c…` (match), archive `35ac2b9beb7e6fa8…` (match), determinism
repeat byte-identical.

To evaluate the shipped archive, first download `archive.zip` from the release
asset linked in the pull request into this directory — `inflate.py` resolves it
beside itself and refuses without it, and the upstream scorer reads it from the
submission directory as well.

```bash
sha256sum archive.zip
mkdir -p archive inflated
unzip -q archive.zip -d archive
./inflate.sh archive inflated public_test_video_names.txt
bash evaluate.sh --submission-dir . --device cuda
```

The CUDA score claim is valid only when the archive hash, runtime content tree,
upstream snapshot, 600-sample count, and reported components all match the
retained authority receipt. CPU and CUDA are distinct score axes and neither is
inferred from the other.

## Dependency closure

The receiver needs four things from the evaluation environment:

- **PyTorch**, from the image.
- **Brotli 1.2.0**, installed into an isolated build directory from a pinned
  wheel when the environment does not already provide it. This is the existing
  fail-closed dependency-closure mechanism: if it cannot be satisfied, the
  receiver exits 69 with a named reason rather than continuing.
- **`constriction`**, the entropy-coding library.
- **A working C compiler.** `inflate.sh` compiles `runtime/entropy/rc64_backend.c`
  into the build directory on every run, using `${CC:-cc}`. This is a hard
  requirement and it is stated here because, unlike the Brotli path, it is not
  guarded — on an image without a toolchain the script fails under `set -e` with
  a raw compiler error rather than a diagnosis. The evaluated T4 image satisfies
  it: inflation completed in 476.6 s.

The receiver downloads no model, table, or video-derived payload. All
score-bearing learned content is inside `archive.zip`.
