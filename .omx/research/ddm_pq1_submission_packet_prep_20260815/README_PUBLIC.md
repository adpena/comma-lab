# sz1 composed candidate — submission packet

This directory describes the exact 179,930-byte archive and receiver used for
the measured sz1 composed result. The score-bearing payload is `archive.zip`;
the receiver is `inflate.sh`, `inflate.py`, `cpr1/`, and `runtime/`.

## Evidence boundary

- `[contest-CUDA]`: a 600-sample exact evaluation on a Tesla T4 measured
  `S = 0.15771357797660338` on the archive identified below.
- `[contest-CPU]`: **measured infeasible within the contest budget** — on a
  contest-like 4-thread x86_64 CPU, inflation took 3,422.7 s against the
  1,800 s budget (token decode 3,108.7 s), and the harness failed closed at
  1,800 s. No CPU score exists and none is claimed. This submission is
  GPU-required for evaluation as a measured fact.
- CPU decode-correctness evidence, which is not a score: the decoded token
  field on x86_64 CPU hashes to the exact sealed value (`9ba2e52b…`) — the
  fixed-point integer token decode is device-exact. The float neural render
  is not bit-identical across CPU microarchitectures (x86_64 raw output
  differs from an arm64 smoke), which is expected.

## Exact identity

- Archive SHA-256:
  `debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a`
- Archive bytes: `179930`
- Member: `p`, `179830` bytes, stored, SHA-256
  `be6db33bce471fe38b3d32cf6b421368721b1ea2ddd3f77b577f2bd27d06b7a8`
- Runtime tree SHA-256 (validated by the exact-authority run):
  `0d0fc008d6a37bd5cfa804073e617a8ea30a7c6b6e6c4a1022e2c5d7ce6f9513`
- Upstream snapshot SHA-256:
  `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`
- Encoder source commits (public repository, to be re-pinned at packet freeze):
  `31c64e4ce0` (semantic serialization split, encoder + receiver + tests) and
  `85880c77a6` (token probability model, frozen shipped configuration).
- Public source repository: https://github.com/adpena/comma-lab

## What this submission is

A composition of two lossless rate moves over the same decoded state as the
prior candidate:

1. **Token probability model.** The RC64 token stream is re-encoded by a
   13-context fixed-point integer log-odds mixer that runs at decode time,
   stores zero archive bytes, and is driven entirely by already-decoded
   symbols. 110,512 → 109,801 bytes; the decoded token field is unchanged
   (SHA-256 `9ba2e52b3096…`).
2. **Semantic serialization split.** 8,284 bytes of raw interleaved fp16
   metadata inside the semantic section are byte-planed (all high bytes, then
   all low bytes) before the container's Brotli pass, −520 bytes. The receiver
   applies the exact inverse permutation before parsing, so decoded values are
   unchanged. The transform is versioned in bit 0 of an existing reserved
   header byte: zero transmitted bytes, and an archive without the bit decodes
   exactly as before. The split offset was selected by argmax over offsets
   0–400; roughly 22 of the 520 bytes are Brotli alignment noise fitted to
   this frozen payload rather than mechanism (adjacent offsets swing ±20 B).

Decode identity was verified at the byte level: the first inflated output file
hashes identically between this row and the token-model-only row
(`9a6b75e5…`). Identical decoded output cannot move `d_seg` or `d_pose`, so
the entire improvement over the prior candidates is rate.

Read `BORROWED_SUBSTRATE_ACCOUNTING.md`, shipped in this directory, before
treating any of the learned content as ours. Most of it is not — the semantic
renderer and the pose carrier are PR #135's, byte-identical after decode. The
same table is reproduced inline in the pull-request body. It also records that
the decode-time-corrector mechanism class was published first by PR #138; we
make no priority claim. The token mixer's context design and the serialization
split are our own work, documented in the accounting table at section level.

## Custody correction: two receipt files in the runtime tree are stale

`GENERATION_RECEIPT.json` and `RECEIVER_PARSEBACK.json` were inherited from a
base candidate's source tree and were never regenerated. They describe a
182,759-byte archive. **That is not the archive in this packet.** The archive
here is 179,930 bytes, SHA-256 `debb025f45bb42e3…`, as stated above and as
measured by the evaluator.

They are stale **labels**, not stale proofs. The binding receipts are the
sealed fire order, the fire manifest's seal block (pin == measured), and the
returned exact-authority JSONs, all bound to `debb025f45bb42e3…`. The two
files are shipped unchanged for one specific reason: the evaluated
runtime-tree hash `0d0fc008d6a3…` is computed over the sealed tree containing
those exact bytes. Editing them would break replay against the value recorded
in the exact-authority row. A `CUSTODY_SUPERSEDED.json` in the retained
custody copy (not the sealed tree) names the real candidate and the receipt
chain.

The same two files contain absolute paths from the machine that generated
them. They are inert: nothing in `inflate.sh`, `inflate.py`, `cpr1/`, or
`runtime/` reads either file.

## Two names a reader will meet in the receiver

Both are cosmetic and neither changes behaviour.

- `CP135` and `F26` appear in `inflate.sh` as an error string, environment
  variables, and file names. They are internal codenames for the inherited
  PR130/PR135 lineage, kept because renaming them would change the evaluated
  runtime-tree hash.
- `inflate.sh` carries a `Darwin` branch that calls `brew --prefix libomp`. It
  is **unreachable on the contest runner**: it requires `F26_TOKEN_DECODER` to
  equal `native-hpac`, and the script defaults that variable to `python` with
  nothing setting it otherwise. This submission assumes Linux.

## Reproduction

Status: VERIFIED. The end-to-end entry point
(`experiments/ddm_pq2_compress_e2e.py`) rebuilds these exact bytes from the
retained checkpoint state under a candidate recipe: the token stream is
re-encoded by the shipped probability model, the intermediate archive is
hash-asserted, the serialization-split builder repacks the container with its
own decode-bit-identity proof, and the final archive must hash to
`debb025f45bb42e3…` at 179,930 bytes — a mismatch at any stage exits non-zero.
The verification run reproduced the archive exactly, with a byte-identical
fresh-process determinism repeat.

To evaluate the shipped archive, first download `archive.zip` from the release
asset linked in the pull request into this directory — `inflate.py` resolves
it beside itself and refuses without it.

```bash
sha256sum archive.zip
mkdir -p archive inflated
unzip -q archive.zip -d archive
./inflate.sh archive inflated public_test_video_names.txt
bash evaluate.sh --submission-dir . --device cuda
```

Measured on the T4 authority run: inflation 1,143.3 s, evaluation 38.3 s —
1.57× headroom against the 1,800 s inflate budget. The token-mixer decode is
the dominant cost; it is pure integer arithmetic and deterministic.

The CUDA score claim is valid only when the archive hash, runtime tree,
upstream snapshot, 600-sample count, and reported components all match the
retained authority receipt. CPU and CUDA are distinct score axes and neither
is inferred from the other.

## Dependency closure

The receiver needs four things from the evaluation environment:

- **PyTorch**, from the image.
- **Brotli 1.2.0**, installed into an isolated build directory from a pinned
  wheel when the environment does not already provide it; if it cannot be
  satisfied, the receiver exits 69 with a named reason.
- **NumPy**, from the image.
- **A working C compiler.** `inflate.sh` compiles
  `runtime/entropy/rc64_backend.c` into the build directory on every run,
  using `${CC:-cc}`. On an image without a toolchain the script fails under
  `set -e` with a raw compiler error rather than a diagnosis. The evaluated T4
  image satisfies it.

The receiver downloads no model, table, or video-derived payload. All
score-bearing learned content is inside `archive.zip`.
