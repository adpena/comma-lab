# e480b RX2 submission packet

This directory contains the exact 183,502-byte archive and receiver used for
the measured e480b RX2 result. The score-bearing payload is `archive.zip`; the
receiver is `inflate.sh`, `inflate.py`, `cpr1/`, and `runtime/`.

## Evidence boundary

- `[contest-CUDA]`: 600-sample exact evaluation on a Tesla T4 measured
  `S = 0.1600920261571558` on the archive identified below.
- `[contest-CPU]`: pending on these exact archive bytes. No CPU score is
  claimed for e480b RX2 in this packet.
- A predecessor CPU receipt is runtime evidence only: the 186,269-byte MC36
  archive inflated in 831.5345 seconds on Linux x86_64 CPU, leaving 2.1647x
  headroom against the 1,800-second limit. It is not the e480b CPU score.

## Exact identity

- Archive SHA-256:
  `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3`
- Archive bytes: `183502`
- Member: `p`, `183402` bytes, SHA-256
  `30c0165ec56dd9327ca4dcda477c34c25f7664622ac37ec8ed171114267d1b58`
- Portable executable-runtime content tree SHA-256:
  `26c7d4f6a8e111c071d74208fc625bf2358e077a06dc59b54ec9421a8d198e0b`
- Evaluation source commit:
  `19dd7916eb9ab5058bbeafa885ac68d8218d0a1e`
- Source URL:
  https://github.com/adpena/comma-lab/commit/19dd7916eb9ab5058bbeafa885ac68d8218d0a1e

The source URL is the intended public pin. Its anonymous visibility must be
verified before a pull request is opened.

## Reproduction

From a clean checkout of the upstream challenge with its locked environment
and public test video installed:

```bash
sha256sum archive.zip
mkdir -p archive inflated
unzip -q archive.zip -d archive
./inflate.sh archive inflated public_test_video_names.txt
bash evaluate.sh --submission-dir . --device cuda
```

The CUDA score claim is valid only when the archive hash, receiver content
tree, upstream snapshot, 600-sample count, and reported components match the
retained authority receipt. CPU and CUDA are distinct score axes.

## Dependency closure

The receiver installs its pinned Brotli dependency into an isolated build
directory when the environment does not already provide it. This is the
existing fail-closed dependency-closure mechanism; it does not download any
score-bearing model, table, or video-derived payload. All score-bearing learned
content is inside `archive.zip`.
