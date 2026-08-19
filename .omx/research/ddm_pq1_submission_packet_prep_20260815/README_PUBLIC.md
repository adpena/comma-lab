# ck1_composed_rowprune — submission packet

(Submission directory name: `submissions/ck1_composed_rowprune/` — the exact
string the PR body's "submission name" answer carries, per the upstream
template's match-exactly requirement.)

This directory describes the exact 177,182-byte archive and receiver used for
the measured ck1 composed result. The score-bearing payload is `archive.zip`;
the receiver is `inflate.sh`, `inflate.py`, `cpr1/`, and `runtime/`.

## Evidence boundary

- `[contest-CUDA]`: a 600-sample exact evaluation on a
  Tesla T4 measured `S = 0.15710198138050818` on the
  archive identified below. The harness publishes its own quantization custody:
  the score is built from 8-decimal-place report components with a worst-case
  absolute error bound of 3.336608391523776e-06.
- `[contest-CPU]`: **no row exists on these bytes, and none is claimed.** An
  earlier candidate in this lineage measured CPU inflate at 3422.711146813 s
  against the 1800 s budget on a contest-like 4-thread x86_64 CPU; this
  candidate ships that token decoder unchanged, so the axis is expected to stay
  infeasible. That expectation is INHERITED, not measured here. This submission
  is GPU-required for evaluation.
- Decode-correctness evidence, which is **not** a score: a full 600-sample local
  decode-and-score of these exact bytes ran to completion on macOS arm64
  (`score_axis=cpu_env_mismatch_advisory`, `score_claim=False`). Its rate
  contribution (0.11797822103209257) is identical to the T4 row's, and its
  pose residual (0.00014829) matches the encoder-side solve to 8 decimal
  places. That agreement is what proves the composed receiver decodes the composed
  container correctly; the float neural render is not bit-identical across
  microarchitectures, which is expected and is why the local seg and pose numbers
  are not a score on either axis.

## Exact identity

- Archive SHA-256:
  `35c318d541d703708ab06c55473c200bb893491e24bea312e37be42f010677e3`
- Archive bytes: `177182`
- Member: `p`, `177082` bytes, stored, SHA-256
  `ee904fbf6b10e4fadd69ca9c820bd7db0d334694bdf23c4a93147cd242d8c462`
- Runtime tree SHA-256 (validated by the exact-authority run):
  `da91e06744b94f77077303b2b760cb259aa84b078d998921fb99e018d52fff6f`
- Portable runtime content tree SHA-256:
  `944c8c574f377cbe625c007b44bfc8e88ec572bf3fc7a2e9ac7aca5750217078`
- Upstream snapshot SHA-256:
  `cdad563c2a3eee39c027d531a8c276ec7970ace47741e937d18d32938bfe7008`
- Encoder source commits (public repository, to be re-pinned at packet freeze):
  see the pull-request body; the compile receipt for these exact bytes is
  `SA3_REBASE.json`, retained beside this packet's receipts.
- Public source repository: https://github.com/adpena/comma-lab

## What this submission is

The same inherited vehicle as the prior candidate, with the semantic section
re-quantized and the resulting pose damage cancelled at compile time.

1. **Row-pruned, mixed-depth semantic quantization (SM3R mode 6).** Three FiLM
   weight tensors keep only their two highest-L2 rows, transmitted as a row
   bitmask plus a compact kept-rows block; a per-tensor 4-bit depth table then
   lets two more tensors (`frame_embed.weight` and `blocks.0.film.weight`) drop
   from 4-bit to 3-bit codes while the rest stay at 4. The resulting semantic
   stream is 31,469 bytes and carries a 36,130-byte body.
   **This is lossy:** the decoded semantic
   values are no longer PR #130/#135's. The receiver learns the new layout from
   one file, `cpr1/ddm_mp2_semantic_receiver.py`; the mode number is carried in
   the container, so an archive that does not use it decodes through the exact
   prior code path.
2. **In-compile frame-0 pose compensation.** Re-quantizing the semantic section
   damages PoseNet, because PoseNet reads the frame pair while the semantic
   renderer produces only frame 1. Rather than accept that damage, the frame-0
   carrier lattice is re-solved at compile time — a damped Gauss-Newton step on
   the receiver-realized Jacobian followed by a multi-scale integer descent — so
   the induced pose error is cancelled in the frame SegNet does not read.
   6,713 of the 7,200 signed-int12 carrier coordinates change. The
   compensation costs 41 archive
   bytes, and its effect is measured rather than assumed: the local solve cancels
   99.9785% of the leakage energy. **The
   decoded pose carrier is therefore no longer byte-identical to PR #135's
   either.**
3. **What this candidate does NOT ship.** The prior candidate's semantic
   byte-plane serialization split is **off** here (`semantic_split=False`,
   reserved bit 0). The row-prune changes the semantic body length, and two
   credits over the same redundancy do not add — re-measured on the edited body,
   the split is negative. Its receiver support remains in the tree and is inert
   on these bytes.

The tail section — the fx2 token stream plus the residual payload and table
codes — is carried over byte-identically from the prior candidate
(109,897 bytes), and so is the HPAC stream
(13,515 bytes). The 13-context fixed-point integer log-odds mixer
that produces the token stream is unchanged and still stores zero archive bytes.

**Unlike the prior candidate, this one does not hold decoded state constant.**
The previous generation's whole improvement was rate, with `d_seg` and `d_pose`
provably unchanged. This one buys 394 bytes and pays for them in
both distortion legs. Against the immediately prior row in this lineage
(177,576 bytes, S 0.1571619225142182) the measured legs are
rate -2.6235e-04, seg +1.7400e-04, pose +2.8407e-05, for a net of
-5.994113e-05; 22.8% of the rate credit is retained.
Against packet generation 3
(179,930 bytes, S 0.15771357797660338) the net is
-0.0006115966.

Read `BORROWED_SUBSTRATE_ACCOUNTING.md`, shipped in this directory, before
treating any of the learned content as ours. Most of it is still not ours: the
semantic renderer and the pose carrier are PR #130 / PR #135's trained state,
and the compressed model container, the HPAC probability object's architecture,
the residual payload, and the range-coder backend all come from that lineage.
What changed at this generation is that we no longer reproduce the renderer and
carrier states BYTE-IDENTICALLY after decode — we ship a lossy re-representation
of theirs — which raises the attribution question rather than settling it, and
the accounting says so in its own section. The same table is reproduced inline in
the pull-request body. It also records that the decode-time-corrector mechanism
class was published first by PR #138, and that the edit-then-recompensate pattern
is PR #135's; we make no priority claim on either.

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

The compile that produced these exact bytes is retained with its receipt
(`SA3_REBASE.json`): it asserts the decoded-state identity between the two
lineages BEFORE building — refusing rather than carrying a compensation across a
changed lattice — then reassembles tail, HPAC, edited semantic, and compensated
carrier sections, and the resulting archive is pinned at SHA-256
`35c318d541d703708ab06c55473c200bb893491e24bea312e37be42f010677e3` / 177,182 bytes. Receiver
parse-back over the shipped runtime returned `PASS` with
`max_abs_code_deviation = 0`.

An end-to-end rebuild of this candidate from pinned retained inputs through one
entry point — the property the previous generation could claim — has **not** been
re-run for these bytes. That is an open item, and the pull-request body says so
rather than inheriting the prior generation's VERIFIED label.

To evaluate the shipped archive, first download `archive.zip` from the release
asset linked in the pull request into this directory — `inflate.py` resolves it
beside itself and refuses without it.

```bash
sha256sum archive.zip
mkdir -p archive inflated
unzip -q archive.zip -d archive
./inflate.sh archive inflated public_test_video_names.txt
bash evaluate.sh --submission-dir . --device cuda
```

Measured on the T4 authority run: inflation 1197.696784587 s, evaluation
40.764544933000025 s — 1.503× headroom against the
1800 s inflate budget. The token-mixer decode is the dominant cost; it is pure
integer arithmetic and deterministic.

The CUDA score claim is valid only when the archive hash, runtime tree, upstream
snapshot, 600-sample count, and reported components all match the retained
authority receipt. CPU and CUDA are distinct score axes and neither is inferred
from the other.

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
