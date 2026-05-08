# Track 2 Custom Decoder Identity Contract

Date: 2026-05-08
Owner: Worker D
Status: byte-custody scaffold only; non-promotable until exact eval

## Purpose

Track 2 custom-decoder work starts with an identity packet scaffold before any
optimization. The scaffold proves that a source packet can be copied into a
byte-closed runtime packet while preserving archive bytes, archive SHA-256,
member inventory, runtime file hashes, executable `inflate.sh`, and runtime
tree identity.

The implementation is:

- `tools/build_track2_identity_packet.py`
- manifest schema: `track2_custom_decoder_identity_packet_v1`
- default output manifest: `track2_identity_manifest.json`

## Contract

Input is a packet directory containing `archive.zip`, `inflate.sh`, and any
runtime files needed by that inflate path. Output is a separate packet directory
with the same archive and runtime files copied byte-for-byte, plus the identity
manifest.

The manifest always records:

- source and output archive bytes and SHA-256
- ZIP member names, sizes, compressed sizes, CRCs, payload SHA-256s, compression
  method, flags, and local-header names
- source and output runtime tree manifests, including file bytes, SHA-256s,
  modes, executable bits, and runtime tree SHA-256
- byte-closure facts: archive copy identity, runtime copy identity, runtime tree
  identity, and unchanged score-affecting payload status
- non-promotable status: `score_claim=false`, `promotion_eligible=false`,
  `dispatchable=false`, and `evidence_grade=byte_custody_only`

## Fail-Closed Guards

The scaffold refuses to build if any of these are present:

- missing `archive.zip`
- unreadable or CRC-broken ZIP
- zip-slip, absolute, backslash, control-character, Windows-drive, hidden,
  resource-fork, or `__MACOSX` archive member names
- duplicate ZIP member names
- more than one known payload container among `p`, `renderer_payload.bin`,
  `renderer_payload.bin.br`, `0.bin`, and `x`
- missing, non-executable, or bash-syntax-invalid `inflate.sh`
- hidden/resource runtime paths or runtime symlinks
- non-empty output directory unless `--force` is explicitly supplied

## Example

```bash
.venv/bin/python tools/build_track2_identity_packet.py \
  --source-packet-dir path/to/source_packet \
  --output-packet-dir experiments/results/track2_identity_packet_20260508 \
  --candidate-id track2_identity_anchor_20260508
```

This does not run `inflate.sh`, scorer networks, CUDA, or any remote job.

## Concrete identity artifact

Validated on a real local packet:

```bash
.venv/bin/python tools/build_track2_identity_packet.py \
  --source-packet-dir experiments/results/pr103_repack_pr106_standalone_20260507/exact_eval_static_release_surface \
  --output-packet-dir experiments/results/track2_identity_pr103_pr106_standalone_20260508T154753Z \
  --candidate-id track2_identity_pr103_pr106_standalone_20260508 \
  --now-utc 2026-05-08T15:50:00Z
```

Result:

- output packet:
  `experiments/results/track2_identity_pr103_pr106_standalone_20260508T154753Z`
- manifest:
  `experiments/results/track2_identity_pr103_pr106_standalone_20260508T154753Z/track2_identity_manifest.json`
- archive SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- archive bytes:
  `185,578`
- runtime tree SHA-256:
  `b021fdd49fad63f08e1dc47d14a6d67023d7ad3cb515a6d5576230d4a67fd67f`
- status:
  `score_claim=false`, `promotion_eligible=false`, `dispatchable=false`

This artifact proves byte-closed identity custody only. It does not change
charged bits and must not be used as a score-lowering claim.

## Remaining Blockers

The identity scaffold is not score evidence. A Track 2 packet remains blocked
from promotion until:

- a real custom-decoder optimization changes charged bits and records old/new
  archive SHA-256s
- exact CUDA auth eval runs through `archive.zip -> inflate.sh ->
  upstream/evaluate.py`
- adjudication and component recomputation land in structured artifacts
- `scripts/pre_submission_compliance_check.py --strict` passes on the exact
  release surface
- any future remote eval has a Level 2 dispatch claim before launch
